"""
LiveKit Voice AI Agent — Entry Point

Registers a LiveKit agent that orchestrates:
  Audio In → Silero VAD → Faster-Whisper STT → Ollama LLM → LFM Audio TTS → Audio Out

Run in development mode:
    python agent/main.py dev

Run in production mode:
    python agent/main.py start
"""

from dotenv import load_dotenv

load_dotenv(".env")

import json
import logging
import re
import httpx
from typing import Annotated
from livekit import agents
from livekit.agents import APIConnectOptions, AgentSession, Agent, AgentServer, llm
from livekit.agents.voice.agent_session import SessionConnectOptions
from livekit.plugins import openai, silero
import asyncio

from agent.config import (
    AGENT_INSTRUCTIONS,
    AGENT_NAME,
    ALLOW_INTERRUPTIONS,
    FALSE_INTERRUPTION_TIMEOUT,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    MAX_ENDPOINTING_DELAY,
    MIN_CONSECUTIVE_SPEECH_DELAY,
    MIN_ENDPOINTING_DELAY,
    MIN_INTERRUPTION_DURATION,
    MIN_INTERRUPTION_WORDS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    RESUME_FALSE_INTERRUPTION,
    TTS_VOICE,
    VOICE_PROMPTS,
    WHISPER_MODEL_SIZE,
)
from agent.search import search_web as search_web_fn
from agent.stt_plugin import WhisperSTT
from agent.tts_plugin import LFMAudioTTS

logger = logging.getLogger(__name__)




class VoiceAssistant(Agent):
    """The AI persona that responds to users in the LiveKit room."""

    def __init__(self) -> None:
        super().__init__(instructions=AGENT_INSTRUCTIONS)

    @llm.function_tool( description=(
        "Use this tool to get real-time, up-to-date information from the web. "
        "You MUST use this tool for any question about current events, health, "
        "recent information, or anything that may be outdated."
    ))
    async def search_web(
        self,
        query: Annotated[str, "The topic or question to search the web for"]
    ) -> str:
        return await search_web_fn(query)


server = AgentServer()


@server.rtc_session(agent_name=AGENT_NAME)
async def voice_agent(ctx: agents.JobContext):
    """
    Called when a user joins a LiveKit room.
    Sets up the full voice pipeline and starts the conversation.
    """

    # -- STT: local Faster-Whisper --
    stt_engine = WhisperSTT(model_size=WHISPER_MODEL_SIZE)

    # -- LLM: Configurable provider (Ollama, Groq, or OpenAI) --
    if LLM_PROVIDER == "groq":
        logger.info("Using Groq LLM: %s", GROQ_MODEL)
        llm_engine = openai.LLM(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            tool_choice="auto"
        )
    elif LLM_PROVIDER == "openai":
        logger.info("Using OpenAI LLM: %s", OPENAI_MODEL)
        llm_engine = openai.LLM(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
        )
    else:
        logger.info("Using Ollama LLM: %s at %s", OLLAMA_MODEL, OLLAMA_BASE_URL)
        llm_engine = openai.LLM(
            model=OLLAMA_MODEL,
            api_key="ollama",
            base_url=OLLAMA_BASE_URL,
            tool_choice="auto",
            max_retries=1,
            # Keep read/write open long enough for first-model load on cold start.
            timeout=httpx.Timeout(
                connect=15.0,
                read=OLLAMA_TIMEOUT_SECONDS,
                write=OLLAMA_TIMEOUT_SECONDS,
                pool=OLLAMA_TIMEOUT_SECONDS,
            ),
        )

    # -- TTS voice selection from room metadata (fallback to default env) --
    selected_voice = TTS_VOICE
    try:
        if ctx.room.metadata:
            metadata = json.loads(ctx.room.metadata)
            requested_voice = metadata.get("voice")
            if requested_voice in VOICE_PROMPTS:
                selected_voice = requested_voice
            else:
                logger.warning("Unsupported voice in room metadata: %s", requested_voice)
    except Exception as e:
        logger.warning("Failed to parse room metadata for voice selection: %s", e)

    # Fallback: parse voice from room name, e.g. lfm-audio-room-us_female-<ts>
    if selected_voice == TTS_VOICE and ctx.room.name:
        match = re.search(r"lfm-audio-room-([a-z_]+)-\d+$", ctx.room.name)
        if match:
            candidate_voice = match.group(1)
            if candidate_voice in VOICE_PROMPTS:
                selected_voice = candidate_voice
            else:
                logger.warning("Unsupported voice from room name: %s", candidate_voice)

    logger.info(
        "Voice selection for room=%s metadata=%s resolved_voice=%s",
        ctx.room.name,
        ctx.room.metadata,
        selected_voice,
    )

    # -- TTS: local LFM 2.5 Audio CLI --
    tts_engine = LFMAudioTTS(voice=selected_voice)

    # -- VAD: Silero (lightweight, CPU-friendly) --
    vad_engine = silero.VAD.load()

    # -- Assemble the pipeline --
    session = AgentSession(
        stt=stt_engine,
        llm=llm_engine,
        tts=tts_engine,
        vad=vad_engine,
        min_endpointing_delay=MIN_ENDPOINTING_DELAY,
        max_endpointing_delay=MAX_ENDPOINTING_DELAY,
        min_consecutive_speech_delay=MIN_CONSECUTIVE_SPEECH_DELAY,
        min_interruption_duration=MIN_INTERRUPTION_DURATION,
        min_interruption_words=MIN_INTERRUPTION_WORDS,
        allow_interruptions=ALLOW_INTERRUPTIONS,
        resume_false_interruption=RESUME_FALSE_INTERRUPTION,
        false_interruption_timeout=FALSE_INTERRUPTION_TIMEOUT,
        conn_options=SessionConnectOptions(
            llm_conn_options=APIConnectOptions(
                max_retry=3,
                retry_interval=2.0,
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )
        ),
    )

    # Start the session (connects to the room's audio tracks)
    await session.start(
        room=ctx.room,
        agent=VoiceAssistant(),
    )

    # Greet the user on connect
    await session.generate_reply(
        instructions="Greet the user warmly and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
