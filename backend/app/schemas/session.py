from pydantic import BaseModel


class ChatSessionResponse(BaseModel):
    client_secret: str
    thread_id: str
    expires_at: int
    user_id: str

