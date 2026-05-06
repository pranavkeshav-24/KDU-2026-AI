# ChatKit Integration and Server-Driven UI

This implementation follows the lab design with a self-managed FastAPI backend and a React/Next.js frontend.

## Runtime Shape

```text
User Browser (Next.js)
        <-> HTTPS / POST streaming
FastAPI Backend (ChatKitServer)
        <-> OpenAI Responses API, or local mock when no API key is configured
```

The browser never calls OpenAI directly. It calls the backend for `/api/session`, `/api/chat`, and `/api/action`.

## Session Security

- `POST /api/session` creates a server-side session record with `session_id`, `thread_id`, `user_id`, and expiry.
- The backend returns a signed opaque `client_secret`.
- Every chat and action call requires `Authorization: Bearer <client_secret>`.
- `ChatKitServer.authorize_thread` validates token signature, expiry, token thread binding, session binding, and thread owner.
- A token issued for one thread receives `403 Forbidden` if it is used against another thread.

## Streaming Protocol

The backend normalizes all provider output into Server-Sent Events:

- `event: text_delta`
- `event: widget`
- `event: done`
- `event: error`

The frontend uses `fetch` and a manual SSE parser rather than `EventSource` so it can send `POST` bodies and bearer auth headers.

## Server-Driven Widgets

The backend owns widget emission through `ToolRegistry`. Widget payloads include:

- `widget_id`
- `type`
- `props`
- `action_endpoint`
- `idempotency_key`

The frontend renderer maps known widget types to React components and shows a fallback for unknown types.

## Hidden Actions

Widget clicks call `POST /api/action` directly. They are hidden events, not visible chat messages. Buttons enter `loading` immediately, then transition to `confirmed`, `duplicate`, or `error`. The backend stores idempotency keys per thread and processes each action once.

## Human Handoff

The backend includes session mode flags (`ai`, `handoff_pending`, `human`) and a `handoff_notice` widget path. This demonstrates the state model from the design doc. Production human handoff would still need a broker or WebSocket channel to push unsolicited human-agent messages while no chat stream is active.

## Provider Portability

The frontend only depends on normalized backend events. To switch providers, update `OpenAILLMClient` or replace it with another adapter that yields text deltas and maps tool calls to the same widget schema.

