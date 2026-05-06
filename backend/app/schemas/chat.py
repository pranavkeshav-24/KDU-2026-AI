from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    thread_id: str = Field(min_length=8)
    message: str = Field(min_length=1, max_length=4000)

