"use client";

import { Check, Headphones, Loader2 } from "lucide-react";

import type { WidgetDefinition, WidgetState } from "@/lib/types";

type HandoffNoticeProps = {
  widget: WidgetDefinition;
  state: WidgetState;
  onAction: (actionType: string, payload?: Record<string, unknown>) => void;
};

export function HandoffNotice({ widget, state, onAction }: HandoffNoticeProps) {
  const disabled = state !== "idle";

  return (
    <article className="widget handoff">
      <h3>{widget.props.title}</h3>
      <p>{widget.props.body}</p>
      <div className="widget-footer">
        <span>{state === "confirmed" ? "Handoff active" : "Waiting for acknowledgement"}</span>
        <button
          className="secondary-button"
          disabled={disabled}
          title="Acknowledge human handoff"
          onClick={() => onAction("acknowledge_handoff")}
        >
          {state === "loading" ? <Loader2 size={18} /> : state === "confirmed" ? <Check size={18} /> : <Headphones size={18} />}
          Acknowledge
        </button>
      </div>
    </article>
  );
}

