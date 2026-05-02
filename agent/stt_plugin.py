"""
Custom LiveKit STT Plugin — wraps Faster-Whisper for local speech-to-text.

Implements the livekit.agents.stt.STT interface so the AgentSession
can feed it real-time audio frames from the WebRTC track.
"""

from __future__ import annotations

import asyncio
import io
import logging
import tempfile
import wave
from typing import Optional

from livekit.agents import stt, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from agent.config import (
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_DOWNLOAD_ROOT,
    WHISPER_MODEL_SIZE,
)

logger = logging.getLogger(__name__)


class WhisperSTT(stt.STT):
    """
    Speech-to-Text plugin using CTranslate2's faster-whisper.

    This keeps the model loaded in memory and transcribes audio frames
    passed in by the LiveKit AgentSession pipeline.
    """

    def __init__(
        self,
        *,
        model_size: str = WHISPER_MODEL_SIZE,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
        download_root: str = WHISPER_DOWNLOAD_ROOT,
        language: str = "en",
        beam_size: int = 5,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False),
        )
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._download_root = download_root
        self._language = language
        self._beam_size = beam_size
        self._model: Optional[object] = None

    def _ensure_model(self):
        """Lazy-load the Whisper model on first use."""
        if self._model is None:
            from faster_whisper import WhisperModel

            logger.info(
                "Loading Whisper model: %s (device=%s, compute=%s)",
                self._model_size,
                self._device,
                self._compute_type,
            )
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
                download_root=self._download_root,
            )
        return self._model

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: str | None = None,
        conn_options: object = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        """
        Transcribe a complete audio buffer using faster-whisper.

        This is called by the AgentSession after the VAD detects the end
        of a speech segment.
        """
        # Merge all audio frames into a single buffer
        frame = utils.merge_frames(buffer)
        sample_rate = frame.sample_rate
        num_channels = frame.num_channels
        audio_bytes = frame.data.tobytes()

        # Run transcription in a thread pool to avoid blocking the event loop
        transcript = await asyncio.get_event_loop().run_in_executor(
            None,
            self._transcribe_sync,
            audio_bytes,
            sample_rate,
            num_channels,
            language,
        )

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    text=transcript,
                    language=language or self._language,
                    confidence=1.0,
                ),
            ],
        )

    def _transcribe_sync(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        num_channels: int,
        language: str | None,
    ) -> str:
        """Synchronous transcription (runs in executor thread)."""
        model = self._ensure_model()

        # Write audio to a temporary WAV file (faster-whisper needs a file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            with wave.open(tmp, "wb") as wf:
                wf.setnchannels(num_channels)
                wf.setsampwidth(2)  # 16-bit PCM
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)

        try:
            segments, info = model.transcribe(
                tmp_path,
                beam_size=self._beam_size,
                language=language or self._language,
            )
            text = "".join(segment.text for segment in segments).strip()
            logger.debug("STT result: %s (lang=%s)", text, info.language)
            return text
        finally:
            import os
            os.unlink(tmp_path)
