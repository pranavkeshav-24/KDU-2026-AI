from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.schemas.action import WidgetActionRequest, WidgetActionResponse
from app.schemas.chat import ChatRequest
from app.schemas.session import ChatSessionResponse
from app.services.chatkit_server import ChatKitServer
from app.services.llm_client import OpenAILLMClient
from app.services.session_store import SessionStore, store
from app.services.token_service import TokenService
from app.services.tool_registry import ToolRegistry

router = APIRouter()

token_service = TokenService(settings.session_secret, settings.session_ttl_seconds)
llm_client = OpenAILLMClient(settings.openai_api_key, settings.openai_model)
tool_registry = ToolRegistry()
chatkit_server = ChatKitServer(store, token_service, llm_client, tool_registry)


def require_bearer(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer client_secret.",
        )
    return authorization.removeprefix("Bearer ").strip()


@router.post("/session", response_model=ChatSessionResponse)
async def create_session(
    user_id: str | None = None,
    session_store: SessionStore = Depends(lambda: store),
) -> ChatSessionResponse:
    user = user_id or settings.default_user_id
    session = session_store.create_session(user)
    client_secret = token_service.issue_client_secret(
        thread_id=session.thread_id,
        user_id=session.user_id,
        session_id=session.session_id,
    )
    session_store.bind_client_secret(client_secret, session.session_id)
    return ChatSessionResponse(
        client_secret=client_secret,
        thread_id=session.thread_id,
        expires_at=session.expires_at,
        user_id=session.user_id,
    )


@router.post("/chat")
async def chat(
    request: ChatRequest,
    client_secret: str = Depends(require_bearer),
) -> StreamingResponse:
    chatkit_server.authorize_thread(client_secret, request.thread_id)
    return StreamingResponse(
        chatkit_server.stream_turn(client_secret, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/action", response_model=WidgetActionResponse)
async def widget_action(
    request: WidgetActionRequest,
    client_secret: str = Depends(require_bearer),
) -> WidgetActionResponse:
    return await chatkit_server.handle_widget_action(client_secret, request)

