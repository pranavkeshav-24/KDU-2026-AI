"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { Send, ShieldCheck } from "lucide-react";

import { API_BASE, createSession, postWidgetAction } from "@/lib/api";
import { streamChatTurn } from "@/lib/sse";
import type { ChatMessage, ChatSession, WidgetDefinition, WidgetState } from "@/lib/types";
import { WidgetRenderer } from "@/components/WidgetRenderer";

type StreamEvent =
  | { delta: string }
  | WidgetDefinition
  | { message: string }
  | { thread_id: string };

export function ChatWorkspace() {
  const [session, setSession] = useState<ChatSession | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "Tell me where you want to go, and I will prepare a travel option."
    }
  ]);
  const [widgets, setWidgets] = useState<WidgetDefinition[]>([]);
  const [widgetStates, setWidgetStates] = useState<Record<string, WidgetState>>({});
  const [input, setInput] = useState("Book a flight to Paris next week");
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    createSession()
      .then(setSession)
      .catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    messagesRef.current?.scrollTo({ top: messagesRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, widgets]);

  const expiresAt = useMemo(() => {
    if (!session?.expires_at) return "pending";
    return new Date(session.expires_at * 1000).toLocaleTimeString();
  }, [session?.expires_at]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!session || !input.trim() || isStreaming) return;

    const userMessage = input.trim();
    const assistantId = crypto.randomUUID();
    setInput("");
    setError(null);
    setIsStreaming(true);
    setMessages((current) => [
      ...current,
      { id: crypto.randomUUID(), role: "user", content: userMessage },
      { id: assistantId, role: "assistant", content: "" }
    ]);

    try {
      await streamChatTurn({
        apiBase: API_BASE,
        clientSecret: session.client_secret,
        threadId: session.thread_id,
        message: userMessage,
        onEvent: (eventName, payload) => {
          handleStreamEvent(eventName, payload as StreamEvent, assistantId);
        }
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Chat stream failed");
    } finally {
      setIsStreaming(false);
    }
  }

  function handleStreamEvent(eventName: string, payload: StreamEvent, assistantId: string) {
    if (eventName === "text_delta" && "delta" in payload) {
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? { ...message, content: message.content + payload.delta }
            : message
        )
      );
    }

    if (eventName === "widget" && "widget_id" in payload) {
      setWidgets((current) =>
        current.some((widget) => widget.widget_id === payload.widget_id)
          ? current
          : [...current, payload]
      );
      setWidgetStates((current) => ({ ...current, [payload.widget_id]: "idle" }));
    }

    if (eventName === "error" && "message" in payload) {
      setError(payload.message);
    }
  }

  async function handleWidgetAction(
    widget: WidgetDefinition,
    actionType: string,
    payload: Record<string, unknown> = {}
  ) {
    if (!session || widgetStates[widget.widget_id] !== "idle") return;

    setWidgetStates((current) => ({ ...current, [widget.widget_id]: "loading" }));
    try {
      const response = await postWidgetAction({
        clientSecret: session.client_secret,
        endpoint: widget.action_endpoint,
        threadId: session.thread_id,
        widgetId: widget.widget_id,
        actionType,
        payload,
        idempotencyKey: widget.idempotency_key
      });
      setWidgetStates((current) => ({ ...current, [widget.widget_id]: response.status }));
    } catch (err) {
      setWidgetStates((current) => ({ ...current, [widget.widget_id]: "error" }));
      setError(err instanceof Error ? err.message : "Widget action failed");
    }
  }

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <ShieldCheck size={22} />
          </div>
          <div>
            <h1>Travel Booking AI</h1>
            <p>Secure ChatKit lab</p>
          </div>
        </div>

        <section className="detail">
          <dl>
            <div>
              <dt>Session</dt>
              <dd>{session ? "Active" : "Initializing"}</dd>
            </div>
            <div>
              <dt>Thread</dt>
              <dd>{session?.thread_id ?? "pending"}</dd>
            </div>
            <div>
              <dt>Token Expires</dt>
              <dd>{expiresAt}</dd>
            </div>
          </dl>
        </section>
      </aside>

      <section className="chat-panel" aria-label="Travel booking chat">
        <header className="chat-header">
          <div>
            <h2>Booking Conversation</h2>
            <span className="status">
              <span className="dot" />
              {isStreaming ? "Streaming" : "Ready"}
            </span>
          </div>
          {error ? <span className="error-text">{error}</span> : null}
        </header>

        <div className="messages" ref={messagesRef}>
          {messages.map((message) => (
            <div className={`message ${message.role}`} key={message.id}>
              {message.content || " "}
            </div>
          ))}

          {widgets.map((widget) => (
            <WidgetRenderer
              key={widget.widget_id}
              widget={widget}
              state={widgetStates[widget.widget_id] ?? "idle"}
              onAction={handleWidgetAction}
            />
          ))}
        </div>

        <form className="composer" onSubmit={handleSubmit}>
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Ask for flights, hotels, or handoff to a human"
            disabled={!session || isStreaming}
          />
          <button className="primary-button" disabled={!session || isStreaming || !input.trim()} title="Send message">
            <Send size={18} />
            Send
          </button>
        </form>
      </section>
    </main>
  );
}

