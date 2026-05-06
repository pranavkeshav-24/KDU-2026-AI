type SseHandler = (eventName: string, payload: unknown) => void;

export async function streamChatTurn(params: {
  apiBase: string;
  clientSecret: string;
  threadId: string;
  message: string;
  onEvent: SseHandler;
}) {
  const response = await fetch(`${params.apiBase}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${params.clientSecret}`
    },
    body: JSON.stringify({
      thread_id: params.threadId,
      message: params.message
    })
  });

  if (!response.ok || !response.body) {
    throw new Error(`Chat stream failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";
    for (const rawEvent of events) {
      dispatchSseEvent(rawEvent, params.onEvent);
    }
  }

  if (buffer.trim()) {
    dispatchSseEvent(buffer, params.onEvent);
  }
}

function dispatchSseEvent(rawEvent: string, onEvent: SseHandler) {
  const lines = rawEvent.split("\n");
  const eventName = lines
    .find((line) => line.startsWith("event:"))
    ?.replace("event:", "")
    .trim();
  const data = lines
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.replace("data:", "").trim())
    .join("");

  if (!eventName || !data) return;
  onEvent(eventName, JSON.parse(data));
}

