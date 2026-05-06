import type { ChatSession } from "@/lib/types";

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export async function createSession(): Promise<ChatSession> {
  const response = await fetch(`${API_BASE}/api/session`, { method: "POST" });
  if (!response.ok) {
    throw new Error(`Session failed: ${response.status}`);
  }
  return response.json();
}

export async function postWidgetAction(params: {
  clientSecret: string;
  endpoint: string;
  threadId: string;
  widgetId: string;
  actionType: string;
  payload: Record<string, unknown>;
  idempotencyKey: string;
}) {
  const response = await fetch(`${API_BASE}${params.endpoint}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${params.clientSecret}`
    },
    body: JSON.stringify({
      thread_id: params.threadId,
      widget_id: params.widgetId,
      action_type: params.actionType,
      payload: params.payload,
      idempotency_key: params.idempotencyKey
    })
  });
  if (!response.ok) {
    throw new Error(`Action failed: ${response.status}`);
  }
  return response.json();
}

