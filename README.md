# Lite LFM Audio API 🎙️

A real-time voice AI agent powered by **Liquid AI's LFM 2.5 Audio** model for TTS,
**Faster-Whisper** for STT, **Ollama / Groq / OpenAI** for LLM reasoning, and
**LiveKit** for WebRTC transport. All self-hosted with Docker.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Browser (WebRTC)                             │
│              http://localhost:3000 (or :8080 via proxy)                 │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │  WebRTC (media) + WS (signal)
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         Nginx Proxy (:8080)                              │
│  / → frontend  │  /api/ → FastAPI  │  /livekit/ → LiveKit WS signal     │
└─────────┬──────────────────────────┬─────────────────────────────────────┘
          │                          │
          ▼                          ▼
┌─────────────────┐    ┌──────────────────────────┐
│   LiveKit Server│    │  FastAPI REST API (:8000) │
│   (WebRTC)      │    │  /token  /tts  /stt       │
│   Ports: 7880   │    └──────────────────────────┘
│   RTC: 7881     │
│   UDP: 50000+   │
└────────┬────────┘
         │  Audio frames in / out
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     LiveKit Voice Agent (agent/main.py)                  │
│                                                                         │
│   ┌──────────┐    ┌───────────┐    ┌────────────┐    ┌──────────────┐  │
│   │  Silero  │───▶│ Faster-   │───▶│  Ollama /  │───▶│   LFM 2.5    │  │
│   │  VAD     │    │ Whisper   │    │  Groq /    │    │   Audio TTS  │  │
│   │ (voice   │    │ STT       │    │  OpenAI    │    │              │  │
│   │  detect) │    │ (speech→  │    │  LLM       │    │ (text→speech)│  │
│   └──────────┘    │  text)    │    │ (reasoning)│    └──────────────┘  │
│                   └───────────┘    └─────┬──────┘                       │
│                                          │                              │
│                                          ▼                              │
│                               ┌──────────────────────┐                  │
│                               │  Web Search (LLM     │                  │
│                               │  function tool)      │                  │
│                               │  ┌────────┐ ┌──────┐ │                  │
│                               │  │ Tavily │ │Duck  │ │                  │
│                               │  │(API)   │ │DuckGo│ │                  │
│                               │  └────────┘ └──────┘ │                  │
│                               └──────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                              Backing Services                            │
│                                                                          │
│  ┌────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────────────┐  │
│  │ Redis  │   │   Ollama     │   │   LFM      │   │   Whisper        │  │
│  │(state) │   │  (qwen2.5    │   │   Audio    │   │   Models         │  │
│  │        │   │   :0.5b)     │   │   GGUF     │   │   (base.en)      │  │
│  └────────┘   └──────────────┘   │   Models   │   └──────────────────┘  │
│                                  └────────────┘                         │
└──────────────────────────────────────────────────────────────────────────┘
```

### Voice Pipeline (detail)

```
    Audio In (mic)
         │
         ▼
 ┌─────────────────┐
 │  Silero VAD      │  Detects when the user is speaking
 │  (voice activity  │  → segments audio into utterances
 │   detection)     │
 └────────┬────────┘
          │ speech segment complete
          ▼
 ┌─────────────────┐
 │ Faster-Whisper   │  Converts speech → text (local, CPU-optimized)
 │ STT              │  Returns: transcript + confidence score
 └────────┬────────┘
          │ text
          ▼
 ┌─────────────────┐
 │ Ollama / Groq /  │  Llm reasoning with tool-calling
 │ OpenAI LLM       │  → If question needs current info:
 │                  │      calls search_web(query) tool
 │                  │       ├── Tavily (primary, API key)
 │                  │       └── DuckDuckGo (fallback, no key)
 │                  │  → Returns response text
 └────────┬────────┘
          │ response text
          ▼
 ┌─────────────────┐
 │ LFM 2.5 Audio   │  Converts text → speech (local GGUF model)
 │ TTS              │  Supports 4 voices: us_male, us_female,
 │                  │  uk_male, uk_female
 └────────┬────────┘
          │ 16-bit PCM audio (24kHz, mono)
          ▼
    Audio Out (speakers)
```

---

## Features

| Feature | Description |
|---------|-------------|
| 🎤 **Real-time Voice** | Full-duplex conversation via WebRTC — low latency |
| 🧠 **Local LLM** | Ollama with `qwen2.5:0.5b` + tool-calling (or swap to Groq/OpenAI) |
| 🔊 **Premium TTS** | Liquid AI's LFM 2.5 Audio with 4 voice options |
| 👂 **Fast STT** | Faster-Whisper (`base.en`) — CPU-optimized, ~150MB model |
| 🔍 **Web Search** | Tavily API (primary) + DuckDuckGo fallback — keeps answers current |
| 🐳 **Fully Dockerized** | `docker compose up` starts everything |
| 🔒 **Self-hosted** | All models + data stay on your machine |
| 🎮 **GPU Support** | Optional NVIDIA acceleration via `docker-compose.gpu.yml` |
| 📡 **Legacy REST API** | Original HTTP endpoints still work for TTS/STT |

---

## Quick Start

### Prerequisites

- [Docker](https://www.docker.com/) and Docker Compose (v2+) installed
- 8 GB+ RAM recommended (16 GB with GPU for best performance)
- ~2 GB free disk space for model downloads

### 1. Configure

```bash
cp .env.example .env
# Edit .env to set your API keys (optional — defaults work for local-only)
```

### 2. Start the stack (CPU-only)

```bash
docker compose up --build
```

First run will download models (~2 GB total). This takes a few minutes.

### 2b. Start with GPU acceleration (optional)

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### 3. Open the voice client

Navigate to **http://localhost:8080** in your browser.

Click **Connect**, allow microphone access, and start talking!

---

## Services

| Service | Internal Port | External Port | Description |
|---------|--------------|---------------|-------------|
| **Proxy** (Nginx) | `80` | `8080` | Single entry point for all web services |
| **Frontend** | `80` | — | Browser voice client (served by Nginx) |
| **FastAPI** | `8000` | — | REST API + LiveKit token provider |
| **LiveKit** | `7880` | `7881` (TCP), `50000-50100` (UDP) | WebRTC media server |
| **Redis** | `6379` | — | State backend for LiveKit |
| **Ollama** | `11434` | — | LLM inference server |
| **Agent** | — | — | Voice AI pipeline (VAD → STT → LLM → TTS) |

---

## Web Search

The agent has a `search_web` tool that the LLM calls automatically when it needs
current information (news, weather, people, facts, health, etc.).

### Search Providers

| Provider | Type | API Key Required | Reliability | Free Tier |
|----------|------|:----------------:|:-----------:|:---------:|
| **Tavily** | Dedicated search API | ✅ Yes | ★★★ High | 1,000 queries/month |
| **DuckDuckGo** | Web scraping (library) | ❌ No | ★★ Medium | Unlimited |

**Tavily** is the recommended provider — it's built specifically for AI agents,
returns clean structured results, and is very reliable. Get a free API key at
[https://app.tavily.com](https://app.tavily.com).

**DuckDuckGo** is the built-in fallback — no API key needed, but may be
rate-limited or blocked in some environments (containers, cloud VMs).

### Configuration

```env
# In .env:
SEARCH_PROVIDER=tavily       # "tavily" (default) or "duckduckgo"
TAVILY_API_KEY=tvly-xxxx     # Your Tavily API key
TAVILY_MAX_RESULTS=5         # Number of results per query (default: 5)
```

**Behavior:**
- If `SEARCH_PROVIDER=tavily` and `TAVILY_API_KEY` is set → uses Tavily, falls back to DuckDuckGo on error
- If `SEARCH_PROVIDER=tavily` but no API key → automatically uses DuckDuckGo
- If `SEARCH_PROVIDER=duckduckgo` → uses DuckDuckGo only

---

## API Endpoints

### Real-time Voice (Primary)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/livekit/token` | POST | Get a LiveKit room token for WebRTC connection |

### Legacy REST API (Still works)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and model status |
| `/voices` | GET | List available voices |
| `/tts/stream` | POST | TTS with SSE streaming (base64 audio) |
| `/tts` | POST | TTS with direct WAV response |
| `/stt` | POST | Speech-to-text (upload audio file) |

### API Docs (when running)

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Available Voices

| Voice ID | Name | Language |
|----------|------|----------|
| `us_male` | US Male | en-US |
| `us_female` | US Female | en-US |
| `uk_male` | UK Male | en-GB |
| `uk_female` | UK Female | en-GB |

Voice can be selected:
- **Per room**: Pass `{"voice": "us_female"}` in the LiveKit room metadata
- **Per room name**: Name your room `lfm-audio-room-us_female-123456`
- **Global default**: Set `TTS_VOICE=us_female` in `.env`

---

## Configuration Reference

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

### LLM

| Variable | Default | Options | Description |
|----------|---------|---------|-------------|
| `LLM_PROVIDER` | `groq` | `ollama`, `groq`, `openai` | Which LLM backend to use |
| `OLLAMA_BASE_URL` | `http://ollama:11434/v1` | — | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen2.5:0.5b` | any Ollama model | Model for local inference |
| `OLLAMA_TIMEOUT_SECONDS` | `90` | — | Timeout for Ollama requests |
| `GROQ_API_KEY` | — | — | Groq API key |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | any Groq model | Model for cloud inference |
| `OPENAI_API_KEY` | — | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | any OpenAI model | Model for cloud inference |

### TTS (LFM 2.5 Audio)

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_VOICE` | `us_male` | Default voice: `us_male`, `us_female`, `uk_male`, `uk_female` |
| `MODEL_DIR` | `/app/models` | Path to model files |
| `RUNNER_PATH` | `/app/runners/llama-liquid-audio-cli` | Path to the LFM CLI binary |

### STT (Faster-Whisper)

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL_SIZE` | `base.en` | Model size: `tiny`, `base`, `small`, `medium`, `large` |
| `WHISPER_DEVICE` | `cpu` | Compute device: `cpu` or `cuda` |
| `WHISPER_COMPUTE_TYPE` | `int8` | Precision: `int8`, `float16`, `float32` |

### Web Search

| Variable | Default | Description |
|----------|---------|-------------|
| `SEARCH_PROVIDER` | `tavily` | `tavily` or `duckduckgo` |
| `TAVILY_API_KEY` | — | Get yours at https://app.tavily.com |
| `TAVILY_MAX_RESULTS` | `5` | Results per search query |

### Voice Turn-Handling

| Variable | Default | Description |
|----------|---------|-------------|
| `MIN_ENDPOINTING_DELAY` | `0.6` | Minimum silence before end of turn (seconds) |
| `MAX_ENDPOINTING_DELAY` | `2.0` | Maximum silence before forced end of turn (seconds) |
| `ALLOW_INTERRUPTIONS` | `true` | Allow the user to interrupt the agent mid-speech |
| `MIN_INTERRUPTION_DURATION` | `0.6` | Minimum speech duration to count as interruption |
| `MIN_INTERRUPTION_WORDS` | `2` | Minimum words to count as interruption |
| `RESUME_FALSE_INTERRUPTION` | `true` | Resume agent after a false interruption |
| `FALSE_INTERRUPTION_TIMEOUT` | `2.0` | Timeout before resuming after false interruption |

---

## Project Structure

```
lite_lfm_audio_api/
│
├── agent/                         # LiveKit Voice Agent (Python)
│   ├── __init__.py               # Package marker
│   ├── main.py                   # Entry point — agent session + tool defs
│   ├── config.py                 # Centralised configuration (all env vars)
│   ├── search.py                 # Web search module (Tavily + DuckDuckGo)
│   ├── stt_plugin.py             # Faster-Whisper STT plugin
│   ├── tts_plugin.py             # LFM 2.5 Audio TTS plugin
│   └── requirements.txt          # Agent Python dependencies
│
├── app/                           # FastAPI REST API (legacy)
│   ├── __init__.py
│   ├── main.py                   # API endpoints and routes
│   └── models.py                 # Pydantic request/response models
│
├── frontend/                      # Browser voice client
│   ├── index.html                # Voice UI (vanilla JS + WebRTC)
│   ├── nginx.conf                # Frontend Nginx config
│   └── Dockerfile                # Nginx container
│
├── proxy/                         # Reverse proxy
│   └── nginx.conf                # Routes /, /api/, /livekit/
│
├── runners/                       # LFM Audio CLI binaries
│   ├── llama-liquid-audio-cli    # TTS binary
│   ├── libggml*.so               # GGML runtime libraries
│   ├── libllama*.so              # Llama.cpp libraries
│   └── libliquid-audio.so        # LFM Audio runtime
│
├── docker-compose.yml             # Full service stack (CPU)
├── docker-compose.gpu.yml         # GPU acceleration override
├── Dockerfile                    # FastAPI container
├── Dockerfile.agent              # Agent container
├── livekit.yaml                  # LiveKit server config
├── download_models.sh            # Model download helper
├── client_test.py                # Legacy REST client test
├── .env.example                  # Template environment file
└── README.md                     # This file
```

### Agent Module Details

| File | Role | Key Details |
|------|------|-------------|
| `agent/main.py` | Entry point, session setup, tool definitions | Registers `search_web` function tool; configures STT/LLM/TTS/VAD pipeline |
| `agent/config.py` | All env-var configuration | LLM provider, search provider, voice tuning parameters |
| `agent/search.py` | Web search module | Tavily (primary) + DuckDuckGo (fallback) with auto-failover |
| `agent/stt_plugin.py` | Custom LiveKit STT plugin | Wraps Faster-Whisper; lazy model load; returns real confidence scores |
| `agent/tts_plugin.py` | Custom LiveKit TTS plugin | Wraps LFM Audio CLI; pure-Python WAV parser (no ffmpeg needed); supports abort/cancellation |

---

## LLM Provider Comparison

| Provider | Type | Speed | Quality | Tool Calling | Cost |
|----------|------|:-----:|:-------:|:------------:|:----:|
| **Ollama** (qwen2.5:0.5b) | Local | ★★ | ★★ | ✅ | Free |
| **Groq** (gpt-oss-120b) | Cloud | ★★★ | ★★★ | ✅ | Free tier |
| **OpenAI** (gpt-4o) | Cloud | ★★★ | ★★★ | ✅ | Paid |

Set via `LLM_PROVIDER` in `.env`. The agent's LLM backend is hot-swappable — no
code changes needed.

---

## Legacy REST Client

The original HTTP test client still works for quick testing without the browser:

```bash
# Install deps
pip install -r requirements.txt

# Run all tests
python client_test.py --mode all

# Or specific mode
python client_test.py --mode tts --text "Hello world"
```

---

## Troubleshooting

### Models fail to download
```bash
# Manually trigger model downloads
docker compose run --rm agent
```

### No audio in browser
- Check microphone permissions in browser settings
- Ensure `http://localhost:8080` uses HTTP (not HTTPS) for local microphone access
- Restart the agent: `docker compose restart agent`

### Search returning empty results
- For Tavily: verify `TAVILY_API_KEY` is set in `.env`
- Check agent logs: `docker compose logs agent | grep search`
- Try switching providers: `SEARCH_PROVIDER=duckduckgo`

### Slow response
- The first LLM response after startup is slower (model loading)
- Consider enabling GPU acceleration with `docker-compose.gpu.yml`
- Reduce `WHISPER_MODEL_SIZE` to `tiny.en` for faster STT

---

## License

This project uses the **LFM 2.5 Audio** model which is subject to the
[Liquid AI License](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF/blob/main/LICENSE).

All other code is provided under the MIT license (see `LICENSE` file).
