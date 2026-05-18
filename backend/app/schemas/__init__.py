import uuid
from datetime import datetime
from pydantic import BaseModel, Field


# ── Session ──────────────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    camera_count: int = Field(default=3, ge=1, le=32)
    sync_strategy: str = Field(default="multividsynch")
    layout: str = Field(default="hstack")


class SessionOut(BaseModel):
    id: uuid.UUID
    name: str
    camera_count: int
    status: str
    sync_strategy: str
    layout: str
    created_at: datetime
    updated_at: datetime
    master_url: str | None = None

    model_config = {"from_attributes": True}


# ── Upload ────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    message: str
    session_id: uuid.UUID
    chunk_index: int
    cam_id: str
    processing_triggered: bool = False


# ── Offset ────────────────────────────────────────────────────────────────────

class OffsetOut(BaseModel):
    cam_id: str
    offset_seconds: float
    computed_at: datetime

    model_config = {"from_attributes": True}


# ── WebSocket Events ──────────────────────────────────────────────────────────

class WSEvent(BaseModel):
    type: str  # chunk_uploaded | chunk_ready | processing_started | chunk_done | error
    session_id: str
    chunk_index: int | None = None
    cam_id: str | None = None
    url: str | None = None
    message: str | None = None
