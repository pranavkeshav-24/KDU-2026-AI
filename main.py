"""FastAPI app entry point and API router bindings."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import settings
from db.sqlite_store import init_db, save_user_profile
from middleware.style import StyleMiddleware
from middleware.user_context import UserContextMiddleware
from models.schemas import AssistantResponse, ChatRequest, UserProfile
from memory.history import get_history, clear_history
from agents.executor import orchestrate_chat

import sys
import logging

# Basic logging config
logging.basicConfig(stream=sys.stdout, level=getattr(logging, settings.LOG_LEVEL, logging.INFO))
logger = logging.getLogger("multimodal_assistant")

limiter = Limiter(key_func=get_remote_address)
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup tasks
    logger.info("Initializing Assistant App...")
    init_db()  # Setup SQLite tables if missing
    yield
    # Shutdown tasks
    logger.info("Assistant Shutdown Complete.")


app = FastAPI(
    title="Multimodal Assistant",
    description="LangChain + OpenRouter Assistant with Context & Vision",
    version="1.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Middlewares ───────────────────────────────────────────────────────────────
app.add_middleware(StyleMiddleware)
app.add_middleware(UserContextMiddleware)

# ── Exception / Rate Limits ───────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Failed handling request {request.url}. Error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "An internal server error occurred."}
    )


# ── Core Endpoints ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
@app.get("/app", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    """Serves the main application landing UI."""
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/chat", include_in_schema=False)
@app.get("/chat/image", include_in_schema=False)
async def redirect_browser_requests_to_ui():
    """If someone opens an API route in the browser, send them to the UI."""
    return RedirectResponse(url="/")

@app.post("/chat", response_model=AssistantResponse)
@limiter.limit("20/minute")
async def chat(request: Request, payload: ChatRequest):
    """Core chat interface. Reads User state and Persona styles embedded by middleware."""
    profile = request.state.user_profile
    persona = request.state.persona
    
    response = await orchestrate_chat(
        message=payload.message,
        thread_id=payload.thread_id,
        user_profile=profile,
        persona_prompt=persona
    )
    return response


@app.post("/chat/image", response_model=AssistantResponse)
@limiter.limit("5/minute")
async def chat_image(
    request: Request, 
    message: str = Form("Analyze this image."), 
    thread_id: str = Form(...),
    file: UploadFile = File(...)
):
    """Multimodal handler accepting image uploads bypassing standard JSON requirements."""
    profile = request.state.user_profile
    persona = request.state.persona
    
    image_bytes = await file.read()
    
    response = await orchestrate_chat(
        message=message,
        thread_id=thread_id,
        user_profile=profile,
        persona_prompt=persona,
        image_bytes=image_bytes
    )
    return response


# ── Memory Endpoints ──────────────────────────────────────────────────────────

@app.get("/history/{thread_id}")
async def fetch_history(thread_id: str):
    return {"thread_id": thread_id, "messages": get_history(thread_id)}


@app.delete("/history/{thread_id}")
async def delete_history(thread_id: str):
    clear_history(thread_id)
    return {"status": "cleared", "thread_id": thread_id}


# ── Profile Endpoints ─────────────────────────────────────────────────────────

@app.get("/profile/{user_id}")
async def profile_query(request: Request, user_id: str):
    # Ideally should validate user_id vs token, simplified for template.
    profile = request.state.user_profile
    if profile and profile.user_id == user_id:
        return profile
    raise HTTPException(status_code=404, detail="Profile not found")


@app.put("/profile/{user_id}")
async def profile_update(user_id: str, profile_data: UserProfile):
    if profile_data.user_id != user_id:
        raise HTTPException(status_code=400, detail="Mismatched User IDs.")
    await save_user_profile(profile_data)
    return {"status": "saved", "user_id": user_id}


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "ok"}
