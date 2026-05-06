from typing import Any, Literal

from pydantic import BaseModel, Field


class WidgetActionRequest(BaseModel):
    thread_id: str = Field(min_length=8)
    widget_id: str = Field(min_length=8)
    action_type: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = Field(min_length=8)


class WidgetActionResponse(BaseModel):
    widget_id: str
    status: Literal["confirmed", "duplicate", "error"]
    message: str

