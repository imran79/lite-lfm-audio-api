"""
Custom LiveKit TTS Plugin — wraps the LFM 2.5 Audio CLI for text-to-speech.

Implements the livekit.agents.tts.TTS interface so the AgentSession
can send LLM output text and receive synthesized audio frames.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from livekit import rtc
from livekit.agents import tts, utils

from agent.config import MODEL_DIR, RUNNER_PATH, TTS_VOICE, VOICE_PROMPTS

logger = logging.getLogger(__name__)


def _find_model_files() -> dict[str, Path]:
    """
    Locate the LFM 2.5 Audio GGUF model files on disk.

    Returns:
        Dictionary of model file paths.

    Raises:
        ValueError: If any required model file is missing.
    """
    searches = {
        "main": ("LFM2.5-Audio-1.5B-Q4_0.gguf", "LFM2.5-Audio-1.5B-*.gguf"),
        "mmproj": ("mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf", "mmproj-LFM2.5-Audio-1.5B-*.gguf"),
        "vocoder": ("vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf", "vocoder-LFM2.5-Audio-1.5B-*.gguf"),
        "tokenizer": ("tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf", "tokenizer-LFM2.5-Audio-1.5B-*.gguf"),
    }

    result = {}
    missing = []

    for key, (preferred, fallback) in searches.items():
        found = list(MODEL_DIR.glob(preferred))
        if not found:
            found = list(MODEL_DIR.glob(fallback))
        if found:
            result[key] = found[0]
        else:
            missing.append(f"{key} ({fallback})")

    if missing:
        raise ValueError(f"Missing LFM model files in {MODEL_DIR}: {', '.join(missing)}")

    return result


class LFMAudioTTS(tts.TTS):
    """
    Text-to-Speech plugin using Liquid AI's LFM 2.5 Audio model.

    Invokes the llama-liquid-audio-cli binary as a subprocess to generate
    WAV audio, then streams the result back into the LiveKit pipeline.
    """

    SAMPLE_RATE = 24000
    NUM_CHANNELS = 1

    def __init__(self, *, voice: str = TTS_VOICE) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=self.SAMPLE_RATE,
            num_channels=self.NUM_CHANNELS,
        )
        self._voice = voice
        self._models: Optional[dict[str, Path]] = None

    def _ensure_models(self) -> dict[str, Path]:
        """Lazy-load model file paths."""
        if self._models is None:
            self._models = _find_model_files()
            logger.info("LFM Audio models found: %s", {k: str(v) for k, v in self._models.items()})
        return self._models

    def synthesize(self, text: str, **kwargs) -> "LFMAudioSynthesizeStream":
        """Create a synthesis handle for the given text."""
        return LFMAudioSynthesizeStream(
            tts=self,
            text=text,
            voice=self._voice,
            conn_options=kwargs.get("conn_options"),
        )


class LFMAudioSynthesizeStream(tts.ChunkedStream):
    """
    Handles a single TTS synthesis request.

    Runs the llama-liquid-audio-cli as a subprocess, reads the output WAV,
    and yields audio frames back to the AgentSession.
    """

    def __init__(self, *, tts: LFMAudioTTS, text: str, voice: str, conn_options: Optional[object] = None) -> None:
        super().__init__(tts=tts, input_text=text, conn_options=conn_options)
        self._voice = voice

    async def _to_pcm16_bytes(self, wav_path: str, sample_rate: int, num_channels: int) -> bytes:
        """
        Normalize model output to PCM16 bytes for LiveKit.

        The LFM runner can emit floating-point WAV (format tag 3), which the
        stdlib `wave` reader in Python 3.12 cannot parse.
        """
        cmd = [
            "ffmpeg",
            "-v", "error",
            "-i", wav_path,
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", str(num_channels),
            "pipe:1",
        ]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown ffmpeg decode error"
            raise RuntimeError(f"Failed to decode TTS output to PCM16: {error_msg}")
        return stdout

    async def _run(self, output_emitter: tts.AudioEmitter = None) -> None:
        """Execute the TTS CLI and push synthesized audio frames."""
        tts_instance: LFMAudioTTS = self._tts
        models = tts_instance._ensure_models()
        system_prompt = VOICE_PROMPTS.get(self._voice, VOICE_PROMPTS["default"])

        # Create a temporary output file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name

        try:
            # Build CLI command
            cmd = [
                str(RUNNER_PATH),
                "-m", str(models["main"]),
                "-mm", str(models["mmproj"]),
                "-mv", str(models["vocoder"]),
                "--tts-speaker-file", str(models["tokenizer"]),
                "-sys", system_prompt,
                "-p", self._input_text,
                "--output", output_path,
            ]

            logger.debug("Running TTS command: %s", " ".join(cmd))

            # Run the CLI tool
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown TTS error"
                logger.error("TTS generation failed: %s", error_msg)
                raise RuntimeError(f"TTS generation failed: {error_msg}")

            # Decode to PCM16 so frames are compatible with LiveKit.
            sample_rate = tts_instance.SAMPLE_RATE
            num_channels = tts_instance.NUM_CHANNELS
            sample_width = 2
            raw_data = await self._to_pcm16_bytes(
                output_path,
                sample_rate=sample_rate,
                num_channels=num_channels,
            )

            # Send audio in chunks (~100ms each for smooth streaming)
            samples_per_chunk = sample_rate // 10  # 100ms chunks
            bytes_per_sample = sample_width * num_channels
            chunk_size = samples_per_chunk * bytes_per_sample

            request_id = utils.shortuuid()
            if output_emitter is not None:
                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=sample_rate,
                    num_channels=num_channels,
                    mime_type="audio/pcm",
                    stream=False,
                )

            for offset in range(0, len(raw_data), chunk_size):
                chunk_data = raw_data[offset : offset + chunk_size]
                if output_emitter:
                    output_emitter.push(chunk_data)
                else:
                    frame = rtc.AudioFrame(
                        data=chunk_data,
                        sample_rate=sample_rate,
                        num_channels=num_channels,
                        samples_per_channel=len(chunk_data) // bytes_per_sample,
                    )
                    self._event_ch.send_nowait(
                        tts.SynthesizedAudio(
                            request_id=request_id,
                            frame=frame,
                        )
                    )

            if output_emitter is not None:
                output_emitter.flush()

            logger.info(
                "TTS synthesized %d bytes of audio for: %.50s...",
                len(raw_data),
                self._input_text,
            )

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
