from typing import Any, Literal

from pydantic import BaseModel, Field


WidgetType = Literal["flight_card", "book_now_button", "date_picker", "handoff_notice"]


class WidgetDefinition(BaseModel):
    widget_id: str = Field(min_length=8)
    type: WidgetType
    props: dict[str, Any]
    action_endpoint: str
    idempotency_key: str

