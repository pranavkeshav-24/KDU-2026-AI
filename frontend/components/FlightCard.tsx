"use client";

import { Check, Loader2, PlaneTakeoff } from "lucide-react";

import type { WidgetDefinition, WidgetState } from "@/lib/types";

type FlightCardProps = {
  widget: WidgetDefinition;
  state: WidgetState;
  onAction: (actionType: string, payload?: Record<string, unknown>) => void;
};

export function FlightCard({ widget, state, onAction }: FlightCardProps) {
  const disabled = state !== "idle";
  const label =
    state === "loading"
      ? "Booking"
      : state === "confirmed" || state === "duplicate"
        ? "Booked"
        : state === "error"
          ? "Unavailable"
          : "Book";

  return (
    <article className="widget flight-card">
      <div className="flight-top">
        <div>
          <h3>{widget.props.airline}</h3>
          <p>{widget.props.fare}</p>
        </div>
        <div className="price">{widget.props.price}</div>
      </div>

      <div className="itinerary">
        <div>
          <div className="time">{widget.props.departure}</div>
          <div>{widget.props.route?.split(" -> ")[0]}</div>
        </div>
        <div className="route-line" />
        <div>
          <div className="time">{widget.props.arrival}</div>
          <div>{widget.props.route?.split(" -> ")[1]}</div>
        </div>
      </div>

      <div className="widget-footer">
        <span>{widget.props.summary}</span>
        <button
          className="primary-button"
          disabled={disabled}
          title="Book this itinerary"
          onClick={() => onAction("book_now", { fare: widget.props.fare })}
        >
          {state === "loading" ? <Loader2 size={18} /> : state === "idle" ? <PlaneTakeoff size={18} /> : <Check size={18} />}
          {label}
        </button>
      </div>
    </article>
  );
}

