"use client";

import type { WidgetDefinition, WidgetState } from "@/lib/types";
import { FlightCard } from "@/components/FlightCard";
import { HandoffNotice } from "@/components/HandoffNotice";

type WidgetRendererProps = {
  widget: WidgetDefinition;
  state: WidgetState;
  onAction: (widget: WidgetDefinition, actionType: string, payload?: Record<string, unknown>) => void;
};

export function WidgetRenderer({ widget, state, onAction }: WidgetRendererProps) {
  const handleAction = (actionType: string, payload?: Record<string, unknown>) => {
    onAction(widget, actionType, payload);
  };

  if (widget.type === "flight_card") {
    return <FlightCard widget={widget} state={state} onAction={handleAction} />;
  }

  if (widget.type === "handoff_notice") {
    return <HandoffNotice widget={widget} state={state} onAction={handleAction} />;
  }

  return (
    <article className="widget fallback">
      <strong>Unsupported widget</strong>
      <p>{widget.type}</p>
    </article>
  );
}

