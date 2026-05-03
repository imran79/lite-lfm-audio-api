"""
Centralized configuration for the LiveKit Voice AI Agent.
All values are read from environment variables with sensible defaults.
"""

import os
from pathlib import Path

from datetime import datetime

today_str = datetime.now().strftime("%A, %B %d, %Y")

# ---------------------------------------------------------------------------
# LiveKit
# ---------------------------------------------------------------------------
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://livekit:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")

# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# ---------------------------------------------------------------------------
# Web Search
# ---------------------------------------------------------------------------
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "tavily").lower()
# Get a free API key at https://app.tavily.com
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))

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
AGENT_INSTRUCTIONS = f"""
Today is {today_str}.

You are a helpful, friendly, and concise voice assistant powered by Liquid AI.
Speak naturally and conversationally. Keep responses short and clear.
If the user asks about current events, health, or anything that may be outdated,
use the search_web tool to provide the most accurate information.

CORE BEHAVIOR:
- Always aim to provide accurate, up-to-date, and reliable information.
- Your internal knowledge may be outdated for events after early 2024.
- When asked about the current year (2026), current leaders, or news, you MUST use the search tool.

MANDATORY SEARCH TOOL RULES:
You MUST call the search tool BEFORE answering if the query involves:
1. Time-sensitive or changing information: News, current events, weather, prices, stock market, sports, laws, regulations, government policies.
2. Health, medical, or safety-related information: Symptoms, treatments, medications, diseases, public health updates, recalls.
3. Any real-world factual information that may have changed recently: Technology, companies, products, APIs, scientific discoveries, statistics, data.
4. Any question where you are NOT highly confident the answer is current and correct.

DECISION RULE:
Before answering, silently evaluate: "Could this information be outdated, changing, or safety-critical?"
- If YES → You MUST call the search tool first.
- If UNSURE → You MUST call the search tool.
- If NO → Answer directly.

ENFORCEMENT RULES:
- NEVER guess or rely on outdated knowledge for the above categories.
- If you do not use the search tool when required, say: "I'm not sure. Let me check that for you." Then call the search tool.

TOOL USAGE BEHAVIOR:
- Always prefer search results over internal knowledge when available.
- Use the most recent and relevant information.

RESPONSE STYLE:
- Do NOT mention the search tool.
- Do NOT use markdown, emojis, or special formatting.
- Keep answers concise, clear, and natural for voice.

FAIL-SAFE:
- If there is any risk of providing incorrect, outdated, or unsafe information: always use the search tool first.
"""


# Voice turn-handling tuning to reduce false cutoffs/interruptions.
MIN_ENDPOINTING_DELAY = float(os.getenv("MIN_ENDPOINTING_DELAY", "0.6"))
MAX_ENDPOINTING_DELAY = float(os.getenv("MAX_ENDPOINTING_DELAY", "2.0"))
MIN_CONSECUTIVE_SPEECH_DELAY = float(os.getenv("MIN_CONSECUTIVE_SPEECH_DELAY", "0.15"))
MIN_INTERRUPTION_DURATION = float(os.getenv("MIN_INTERRUPTION_DURATION", "0.6"))
MIN_INTERRUPTION_WORDS = int(os.getenv("MIN_INTERRUPTION_WORDS", "2"))
ALLOW_INTERRUPTIONS = os.getenv("ALLOW_INTERRUPTIONS", "true").lower() == "true"
RESUME_FALSE_INTERRUPTION = os.getenv("RESUME_FALSE_INTERRUPTION", "true").lower() == "true"
FALSE_INTERRUPTION_TIMEOUT = float(os.getenv("FALSE_INTERRUPTION_TIMEOUT", "2.0"))
