"""
Centralized configuration for the LiveKit Voice AI Agent.
All values are read from environment variables with sensible defaults.
"""

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# LiveKit
# ---------------------------------------------------------------------------
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://livekit:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")

# ---------------------------------------------------------------------------
# Ollama (LLM)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))

# ---------------------------------------------------------------------------
# LFM 2.5 Audio (TTS)
# ---------------------------------------------------------------------------
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))
RUNNER_PATH = Path(os.getenv("RUNNER_PATH", "/app/runners/llama-liquid-audio-cli"))
TTS_VOICE = os.getenv("TTS_VOICE", "us_male")

VOICE_PROMPTS = {
    "default": "Perform TTS.",
    "us_male": "Perform TTS. Use the US male voice.",
    "us_female": "Perform TTS. Use the US female voice.",
    "uk_male": "Perform TTS. Use the UK male voice.",
    "uk_female": "Perform TTS. Use the UK female voice.",
}

# ---------------------------------------------------------------------------
# Faster-Whisper (STT)
# ---------------------------------------------------------------------------
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_DOWNLOAD_ROOT = str(MODEL_DIR / "whisper")

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
AGENT_NAME = os.getenv("AGENT_NAME", "lfm-audio-agent")
AGENT_INSTRUCTIONS = os.getenv(
    "AGENT_INSTRUCTIONS",
    "You are a helpful, friendly voice assistant powered by Liquid AI. "
    "Keep your responses concise and conversational. "
    "Do not use markdown, emojis, or special formatting in your replies.",
)

# Voice turn-handling tuning to reduce false cutoffs/interruptions.
MIN_ENDPOINTING_DELAY = float(os.getenv("MIN_ENDPOINTING_DELAY", "0.6"))
MAX_ENDPOINTING_DELAY = float(os.getenv("MAX_ENDPOINTING_DELAY", "2.0"))
MIN_CONSECUTIVE_SPEECH_DELAY = float(os.getenv("MIN_CONSECUTIVE_SPEECH_DELAY", "0.15"))
MIN_INTERRUPTION_DURATION = float(os.getenv("MIN_INTERRUPTION_DURATION", "0.6"))
MIN_INTERRUPTION_WORDS = int(os.getenv("MIN_INTERRUPTION_WORDS", "2"))
ALLOW_INTERRUPTIONS = os.getenv("ALLOW_INTERRUPTIONS", "true").lower() == "true"
RESUME_FALSE_INTERRUPTION = os.getenv("RESUME_FALSE_INTERRUPTION", "true").lower() == "true"
FALSE_INTERRUPTION_TIMEOUT = float(os.getenv("FALSE_INTERRUPTION_TIMEOUT", "2.0"))
