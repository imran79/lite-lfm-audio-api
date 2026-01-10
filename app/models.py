from pydantic import BaseModel, Field
from enum import Enum


class VoiceType(str, Enum):
    """Available TTS voice types."""
    US_MALE = "us_male"
    US_FEMALE = "us_female"
    UK_MALE = "uk_male"
    UK_FEMALE = "uk_female"
    DEFAULT = "default"


class TTSRequest(BaseModel):
    """Request model for TTS endpoint."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to convert to speech")
    voice: VoiceType = Field(default=VoiceType.DEFAULT, description="Voice type to use")


class VoiceInfo(BaseModel):
    """Voice information model."""
    id: str
    name: str
    language: str


class VoicesResponse(BaseModel):
    """Response model for voices endpoint."""
    voices: list[VoiceInfo]


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_dir: str | None = None
    error: str | None = None


class AudioChunkEvent(BaseModel):
    """SSE event data for audio chunk."""
    audio: str  # base64 encoded
    format: str = "wav"
    sample_rate: int = 24000


class CompleteEvent(BaseModel):
    """SSE event data for completion."""
    success: bool = True


class STTResponse(BaseModel):
    """Response model for STT endpoint."""
    text: str
    language: str | None = None
    segments: list[dict] | None = None

class ErrorEvent(BaseModel):
    """SSE event data for errors."""
    message: str
