export type ChatSession = {
  client_secret: string;
  thread_id: string;
  expires_at: number;
  user_id: string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

export type WidgetDefinition = {
  widget_id: string;
  type: "flight_card" | "book_now_button" | "date_picker" | "handoff_notice";
  props: Record<string, string>;
  action_endpoint: string;
  idempotency_key: string;
};

export type WidgetState = "idle" | "loading" | "confirmed" | "duplicate" | "error";

