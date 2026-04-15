# Multimodal AI Assistant — Design Document

**Context-Aware · Memory-Enabled · Multimodal · Multi-Model**
Built with LangChain + Python + OpenRouter | Version 1.0 | April 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Tech Stack](#2-tech-stack)
3. [System Architecture](#3-system-architecture)
4. [Component Design](#4-component-design)
   - [Step 1 — Context-Aware Responses](#step-1--context-aware-responses)
   - [Step 2 — Structured Output](#step-2--structured-output)
   - [Step 3 — Memory Management](#step-3--memory-management)
   - [Step 4 — Multimodal Capability](#step-4--multimodal-capability)
   - [Step 5 — Dynamic Behavior](#step-5--dynamic-behavior)
5. [Model Reference](#5-model-reference-openrouter-free-tier)
6. [API Reference](#6-api-reference)
7. [Environment Variables](#7-environment-variables)
8. [Implementation Plan](#8-implementation-plan)
9. [Python Dependencies](#9-python-dependencies)
10. [Open Questions & Future Enhancements](#10-open-questions--future-enhancements)

---

## 1. Executive Summary

This document describes the architecture, design decisions, and implementation plan for a context-aware, multimodal AI assistant. The system is built on the LangChain framework and uses OpenRouter as the unified LLM gateway, exposing free-tier models for cost-efficient development and testing.

The assistant integrates five core capabilities into a single coherent Python service:

- **Context-aware responses** — personalized using stored user profiles; no location re-prompting
- **Structured output enforcement** — all responses conform to a strict Pydantic/JSON schema
- **Multi-session memory** — conversation history persists across sessions via Redis
- **Image understanding** — users can upload images for description and analysis
- **Dynamic model/persona switching** — communication style and underlying LLM adapt per task

---

## 2. Tech Stack

| Component         | Technology                                                         |
| ----------------- | ------------------------------------------------------------------ |
| Language          | Python 3.11+                                                       |
| LLM Framework     | LangChain (`langchain`, `langchain-community`, `langchain-openai`) |
| LLM Gateway       | OpenRouter (openrouter.ai) — unified multi-model API               |
| Primary Model     | `google/gemma-3-27b-it` — general reasoning & multimodal           |
| Secondary Model   | `meta-llama/llama-4-scout` — fast lightweight responses            |
| Tertiary Model    | `mistralai/mistral-small-3.2-24b-instruct` — structured output     |
| Weather API       | LangChain OpenWeatherMap tool (`langchain-community`)              |
| Memory Backend    | LangChain `ConversationBufferMemory` / `RedisChatMessageHistory`   |
| Output Validation | Pydantic v2 + LangChain `PydanticOutputParser`                     |
| Image Handling    | Base64 encoding passed into vision-capable model via LangChain     |
| Persistence       | Redis (session store) / SQLite (user profiles)                     |
| Serving           | FastAPI + Uvicorn                                                  |
| Config            | `python-dotenv` + YAML config files                                |
| Testing           | `pytest` + `pytest-asyncio`                                        |

> **Why OpenRouter?** Single API key, access to 100+ models. Drop-in OpenAI-compatible endpoint — works seamlessly with `langchain-openai`. Free-tier models allow zero-cost development. Supports vision on compatible models at no extra config.

---

## 3. System Architecture

### 3.1 High-Level Request Flow

```
┌──────────────────────────────────────────────────────────────┐
│                         CLIENT                               │
│           (text / image / voice payload)                     │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│                     FastAPI Layer                            │
│   POST /chat   |   POST /chat/image   |   GET /history       │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│                  Middleware Pipeline                         │
│   1. UserContextMiddleware  (inject user profile)            │
│   2. StyleMiddleware        (set persona: expert / child)    │
│   3. ModelRouter            (select LLM by task type)        │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│               LangChain Agent Executor                       │
│   ┌─────────────────┐   ┌──────────────────────────────┐    │
│   │  Memory Store   │   │  Tool Registry               │    │
│   │  (Redis / SQLite│   │  • WeatherTool               │    │
│   │   per thread_id)│   │  • ImageAnalysisTool         │    │
│   └─────────────────┘   │  • UserProfileTool           │    │
│                         └──────────────────────────────┘    │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│               OpenRouter LLM Gateway                         │
│   google/gemma-3-27b-it       (default / vision)        │
│   meta-llama/llama-4-scout    (fast / lightweight)       │
│   mistralai/mistral-small-3.2-24b-instruct  (JSON out)  │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│               Output Parser & Validator                      │
│   PydanticOutputParser  ──►  JSON Response Schema            │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Module Map

```
multimodal_assistant/
├── main.py                    # FastAPI entrypoint
├── config.py                  # Settings (env vars, YAML)
├── models/
│   ├── router.py              # ModelRouter logic
│   └── schemas.py             # Pydantic response schemas
├── middleware/
│   ├── user_context.py        # Inject user profile
│   └── style.py               # Persona / style injection
├── memory/
│   ├── store.py               # Memory factory (buffer / Redis)
│   └── history.py             # Per-thread message history
├── tools/
│   ├── weather.py             # LangChain WeatherTool wrapper
│   ├── user_profile.py        # Profile lookup tool
│   └── image_analysis.py      # Vision tool
├── agents/
│   └── executor.py            # Agent chain assembly
└── tests/
    ├── test_weather.py
    ├── test_memory.py
    └── test_multimodal.py
```

---

## 4. Component Design

---

### Step 1 — Context-Aware Responses

#### User Profile & Context Injection

The assistant should never ask a user for information it can already derive from their stored profile. A lightweight SQLite store persists user profiles keyed by `user_id`, containing location, timezone, preferred units, and language preferences.

**User Profile Schema**

```python
class UserProfile(BaseModel):
    user_id:      str
    name:         str
    city:         str           # e.g. 'Bengaluru'
    lat:          float
    lon:          float
    timezone:     str           # e.g. 'Asia/Kolkata'
    unit_system:  Literal['metric', 'imperial']
    language:     str           # e.g. 'en'
    created_at:   datetime
```

#### Weather Tool Integration

LangChain ships an OpenWeatherMap tool in `langchain-community`. The tool is initialized once and registered in the agent's tool list. When the LLM decides it needs weather data, it calls the tool autonomously — the user never needs to specify their location because it is injected from the profile.

```python
# tools/weather.py
from langchain_community.tools.openweathermap.tool import OpenWeatherMapQueryRun
from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper

def build_weather_tool() -> OpenWeatherMapQueryRun:
    wrapper = OpenWeatherMapAPIWrapper()   # reads OPENWEATHERMAP_API_KEY from env
    return OpenWeatherMapQueryRun(api_wrapper=wrapper)
```

#### UserContextMiddleware

A FastAPI middleware reads the `X-User-ID` header, fetches the profile from SQLite, and attaches it to `request.state`. Downstream, the agent builder reads `request.state.user_profile` and prepends a system message that primes the model with user context.

```python
# middleware/user_context.py
class UserContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        uid     = request.headers.get('X-User-Id', 'anonymous')
        profile = await get_user_profile(uid)     # SQLite async query
        request.state.user_profile = profile
        return await call_next(request)
```

> **Example behavior:** User sends "What's the weather like today?" → Middleware injects `city=Bengaluru, lat=12.97, lon=77.59` → Agent calls `WeatherTool` with injected coordinates — no location prompt needed.

---

### Step 2 — Structured Output

#### JSON Response Enforcement

All responses from the assistant must conform to a strict Pydantic schema. LangChain's `PydanticOutputParser` injects format instructions into the prompt and validates the raw LLM output against the schema. If parsing fails, a retry chain with `OutputFixingParser` automatically corrects malformed output.

**Response Schema**

```python
# models/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, Literal

class WeatherResponse(BaseModel):
    temperature:  float  = Field(description='Current temperature in user units')
    feels_like:   float  = Field(description='Feels-like temperature')
    condition:    str    = Field(description='Short weather condition label')
    summary:      str    = Field(description='One-sentence natural language summary')
    location:     str    = Field(description='City and country code')
    humidity:     int    = Field(description='Relative humidity percentage')
    wind_speed:   float  = Field(description='Wind speed in user units')
    unit:         Literal['metric', 'imperial']

class AssistantResponse(BaseModel):
    intent:       str                      = Field(description='Detected user intent')
    content:      str                      = Field(description='Main assistant reply')
    data:         Optional[WeatherResponse] = None
    follow_up:    Optional[str]            = None
```

**Parser Chain**

```python
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_openai import ChatOpenAI

base_parser   = PydanticOutputParser(pydantic_object=AssistantResponse)
fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

prompt = ChatPromptTemplate.from_messages([
    ('system', SYSTEM_PROMPT + '\n' + base_parser.get_format_instructions()),
    MessagesPlaceholder('history'),
    ('human', '{input}'),
])

chain = prompt | llm | fixing_parser
```

---

### Step 3 — Memory Management

Memory is scoped to a `thread_id` (session identifier). Two memory backends are supported based on the deployment environment: an in-process `ConversationBufferMemory` for development, and a Redis-backed `RedisChatMessageHistory` for production.

| Memory Type          | Implementation                                                      |
| -------------------- | ------------------------------------------------------------------- |
| Short-term           | `ConversationBufferMemory` — full turn-by-turn history in RAM       |
| Short-term (bounded) | `ConversationTokenBufferMemory` — trims to `max_token_limit`        |
| Persistent           | `RedisChatMessageHistory` — survives restarts; keyed by `thread_id` |
| Summary              | `ConversationSummaryMemory` — compresses long histories via LLM     |

**Memory Factory**

```python
# memory/store.py
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

def get_session_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(
        session_id=session_id,
        url=settings.REDIS_URL,
        ttl=86400    # 24-hour expiry
    )

# Wrap the chain with history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='history',
)

# Invoke with a thread id
response = chain_with_history.invoke(
    {'input': user_message},
    config={'configurable': {'session_id': thread_id}}
)
```

> **Memory design decisions:**
>
> - `thread_id = user_id + ':' + session_id` enables both per-session and per-user scoping.
> - TTL of 24 hours on Redis entries prevents unbounded memory growth.
> - `ConversationTokenBufferMemory` (max 4096 tokens) is used as a fallback for very long sessions.
> - An explicit `GET /history/{thread_id}` endpoint allows the client to render conversation history.

---

### Step 4 — Multimodal Capability

Users can upload images via a dedicated `/chat/image` endpoint. The image is base64-encoded and passed as a vision message to a compatible multimodal model. LangChain's `HumanMessage` with `image_url` content type is used to construct the multimodal prompt.

**Supported Image Modalities**

| Use Case              | Example Prompt                                 |
| --------------------- | ---------------------------------------------- |
| Scene Description     | "Describe what is happening in this image"     |
| Object Detection      | "What objects can you identify in this image?" |
| OCR / Text Extraction | "Extract all text visible in this image"       |
| Diagram Analysis      | "Explain this architecture diagram"            |
| Image + Text          | "Given this chart, what is the trend in Q3?"   |

**Vision Message Construction**

```python
# tools/image_analysis.py
from langchain_core.messages import HumanMessage
import base64

def build_vision_message(image_bytes: bytes, user_text: str) -> HumanMessage:
    b64 = base64.b64encode(image_bytes).decode('utf-8')
    return HumanMessage(content=[
        {
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{b64}',
            },
        },
        {'type': 'text', 'text': user_text},
    ])

# Vision requests bypass tool routing and go directly to the multimodal model
async def analyze_image(
    image_bytes: bytes,
    question: str,
    llm
) -> AssistantResponse:
    msg    = build_vision_message(image_bytes, question)
    result = await llm.ainvoke([msg])
    return fixing_parser.parse(result.content)
```

---

### Step 5 — Dynamic Behavior

Two orthogonal dimensions of dynamic behavior are supported: **persona/style switching** (how the model communicates) and **model routing** (which underlying LLM is used). Both are resolved in the middleware pipeline before the agent executes.

#### Persona / Style Switching

The client passes an optional `X-Response-Style` header. The `StyleMiddleware` injects the appropriate system prompt prefix:

| Style Token | System Prompt Behavior                                                                                |
| ----------- | ----------------------------------------------------------------------------------------------------- |
| `expert`    | Use precise technical language. Assume the user has domain expertise. Include caveats and edge cases. |
| `child`     | Use simple words, short sentences, and friendly analogies suitable for a 10-year-old.                 |
| `casual`    | Conversational tone. Use contractions. Keep it light and friendly.                                    |
| `formal`    | Professional, structured responses. Avoid colloquialisms. Be concise and authoritative.               |

#### Model Router

The `ModelRouter` selects the LLM instance based on the classified task type. Classification is a lightweight keyword/heuristic pass — no additional LLM call required.

```python
# models/router.py
TASK_MODEL_MAP = {
    'vision':     'google/gemma-3-27b-it',                      # multimodal
    'reasoning':  'google/gemma-3-27b-it',                      # general reasoning
    'fast':       'meta-llama/llama-4-scout',                    # simple / fast queries
    'structured': 'mistralai/mistral-small-3.2-24b-instruct',   # JSON output
}

def classify_task(payload: ChatRequest) -> str:
    if payload.image:
        return 'vision'
    if any(k in payload.message.lower()
           for k in ['weather', 'temperature', 'forecast']):
        return 'structured'
    if len(payload.message.split()) < 10:
        return 'fast'
    return 'reasoning'

def get_llm(task: str) -> ChatOpenAI:
    model_id = TASK_MODEL_MAP.get(task, 'google/gemma-3-27b-it')
    return ChatOpenAI(
        base_url='https://openrouter.ai/api/v1',
        api_key=settings.OPENROUTER_API_KEY,
        model=model_id,
    )
```

---

## 5. Model Reference (OpenRouter Free Tier)

| Model ID                                   | Primary Use                         | Strengths                                   |
| ------------------------------------------ | ----------------------------------- | ------------------------------------------- |
| `google/gemma-3-27b-it`                    | Vision, reasoning, default fallback | Multimodal, strong context handling         |
| `meta-llama/llama-4-scout`                 | Fast single-turn responses          | Low latency; 10M context window             |
| `mistralai/mistral-small-3.2-24b-instruct` | Structured JSON output              | Instruction-tuned; follows schemas reliably |

> **Note on free tier limits:** OpenRouter free models are subject to rate limits (typically 20 RPM / 200K tokens/day per model). The `ModelRouter` distributes load across models, reducing the risk of hitting a single model's limit. A simple token-bucket rate limiter (`slowapi`) is recommended on the FastAPI layer for production.

---

## 6. API Reference

| Method   | Path                   | Description                                                    |
| -------- | ---------------------- | -------------------------------------------------------------- |
| `POST`   | `/chat`                | Send a text message; returns `AssistantResponse` JSON          |
| `POST`   | `/chat/image`          | Upload image + optional text; returns `AssistantResponse` JSON |
| `GET`    | `/history/{thread_id}` | Retrieve full conversation history for a thread                |
| `DELETE` | `/history/{thread_id}` | Clear conversation history for a thread                        |
| `GET`    | `/profile/{user_id}`   | Fetch stored user profile                                      |
| `PUT`    | `/profile/{user_id}`   | Update user profile (location, preferences, etc.)              |
| `GET`    | `/health`              | Liveness check — returns `{"status": "ok"}`                    |

### Example Request / Response

```json
// POST /chat
// Headers: X-User-Id: user_42 | X-Response-Style: expert
{
  "message": "What's the weather today?",
  "thread_id": "user_42:session_7"
}

// 200 OK
{
  "intent": "weather_query",
  "content": "Current conditions in Bengaluru: 28°C, partly cloudy.",
  "data": {
    "temperature": 28.0,
    "feels_like": 31.2,
    "condition": "Partly Cloudy",
    "summary": "Warm and partly cloudy in Bengaluru with moderate humidity.",
    "location": "Bengaluru, IN",
    "humidity": 72,
    "wind_speed": 14.4,
    "unit": "metric"
  },
  "follow_up": "Would you like an hourly forecast or packing suggestions?"
}
```

---

## 7. Environment Variables

| Variable                 | Purpose                                                            |
| ------------------------ | ------------------------------------------------------------------ |
| `OPENROUTER_API_KEY`     | OpenRouter API key — used as the OpenAI-compatible API key         |
| `OPENWEATHERMAP_API_KEY` | OpenWeatherMap API key for the LangChain weather tool              |
| `REDIS_URL`              | Redis connection string (e.g. `redis://localhost:6379`)            |
| `DATABASE_URL`           | SQLite or Postgres URL for user profile persistence                |
| `DEFAULT_STYLE`          | Default persona style: `expert` \| `casual` \| `child` \| `formal` |
| `LOG_LEVEL`              | Logging verbosity: `DEBUG` \| `INFO` \| `WARNING`                  |

---

## 8. Implementation Plan

| Phase             | Timeline | Deliverables                                                                                                |
| ----------------- | -------- | ----------------------------------------------------------------------------------------------------------- |
| Phase 1 — Core    | Week 1   | FastAPI skeleton, OpenRouter LLM integration, basic chat endpoint, UserProfile SQLite store                 |
| Phase 2 — Tools   | Week 1–2 | WeatherTool integration, `UserContextMiddleware`, `PydanticOutputParser` chain with `OutputFixingParser`    |
| Phase 3 — Memory  | Week 2   | `RunnableWithMessageHistory`, Redis backend, `/history` endpoints, `ConversationTokenBufferMemory` fallback |
| Phase 4 — Vision  | Week 2–3 | Image upload endpoint, base64 vision message builder, multimodal model routing                              |
| Phase 5 — Dynamic | Week 3   | `StyleMiddleware`, `ModelRouter` classification, persona system prompts, rate limiter                       |
| Phase 6 — Tests   | Week 3–4 | `pytest` suite for all modules, integration test with mock OpenRouter, load test                            |

---

## 9. Python Dependencies

```txt
# requirements.txt
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
langchain>=0.2.0
langchain-community>=0.2.0
langchain-openai>=0.1.0
langchain-core>=0.2.0
pydantic>=2.7.0
python-dotenv>=1.0.0
redis>=5.0.0
aiofiles>=23.0.0
python-multipart>=0.0.9    # for image upload
pyowm>=3.3.0               # OpenWeatherMap (used by langchain-community)
slowapi>=0.1.9             # rate limiting
pytest>=8.0.0
pytest-asyncio>=0.23.0
httpx>=0.27.0              # async test client
```

---

## 10. Open Questions & Future Enhancements

- Should user profiles be editable via the API or only via an admin interface?
- Long-term: migrate from SQLite to PostgreSQL for user profiles to support horizontal scaling.
- Audio input modality — Whisper API for speech-to-text could be added as a Phase 7.
- Streaming responses via Server-Sent Events (SSE) for improved perceived latency.
- LangSmith tracing integration for observability and prompt debugging in production.
- Consider adding a tool-call confidence threshold — if the model is uncertain whether to call a tool, it should ask rather than hallucinate tool arguments.

---

_End of Document — Multimodal AI Assistant Design Doc v1.0_
