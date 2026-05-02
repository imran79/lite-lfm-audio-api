# Lite LFM Audio API 🎙️
A real-time voice AI agent powered by **Liquid AI's LFM 2.5 Audio** model, **Ollama** for LLM inference, and **LiveKit** for WebRTC transport.

## What is this?

A self-hosted voice AI platform that lets you have real-time conversations with an AI agent — all running locally on your machine.

### Architecture

```
Browser (WebRTC) → LiveKit Server → Voice Agent Pipeline → LiveKit Server → Browser
                                    ├── Silero VAD (voice detection)
                                    ├── Faster-Whisper STT (speech → text)
                                    ├── Ollama LLM (text reasoning)
                                    └── LFM 2.5 Audio TTS (text → speech)
```

### Features

| Feature | Description |
|---------|-------------|
| 🎤 **Real-time Voice** | Talk to the AI agent in real-time via WebRTC |
| 🧠 **Local LLM** | Ollama runs `qwen2.5:0.5b` with tool-calling support |
| 🔊 **Premium TTS** | Liquid AI LFM 2.5 Audio with multiple voice options |
| 👂 **Fast STT** | Faster-Whisper for CPU-optimized speech recognition |
| 🐳 **Fully Dockerized** | One command to start the entire stack |
| 🔒 **Self-hosted** | All data stays on your machine |
| 🎮 **GPU Support** | Optional NVIDIA GPU acceleration for Ollama |
| 📡 **Legacy REST API** | Original HTTP endpoints still work |

## Quick Start

### Prerequisites

- [Docker](https://www.docker.com/) and Docker Compose installed
- 8GB+ RAM recommended (16GB with GPU for best performance)

### 1. Start the stack (CPU-only)

```bash
docker compose up --build
```

### 1b. Start with GPU acceleration (optional)

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed on your host.

### 2. Wait for model downloads

On first run, the system will:
- Download LFM 2.5 Audio GGUF models (~1GB)
- Download the Whisper STT model (~150MB)
- Pull `qwen2.5:0.5b` into Ollama

### 3. Open the voice client

Navigate to **http://localhost:3000** in your browser.

Click **Connect**, allow microphone access, and start talking!

## Services

| Service | Port | Description |
|---------|------|-------------|
| **Frontend** | `3000` | Browser voice client |
| **FastAPI** | `8000` | REST API + LiveKit token provider |
| **LiveKit** | `7880` | WebRTC media server |
| **Ollama** | `11434` | LLM inference |

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

### API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Available Voices

| Voice ID | Name | Language |
|----------|------|----------|
| `us_male` | US Male | en-US |
| `us_female` | US Female | en-US |
| `uk_male` | UK Male | en-GB |
| `uk_female` | UK Female | en-GB |

## Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `LIVEKIT_URL` | `ws://localhost:7880` | LiveKit server URL |
| `LIVEKIT_API_KEY` | `devkey` | LiveKit API key |
| `LIVEKIT_API_SECRET` | `secret` | LiveKit API secret |
| `OLLAMA_BASE_URL` | `http://ollama:11434/v1` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen2.5:0.5b` | LLM model for the agent |
| `TTS_VOICE` | `us_male` | Default TTS voice |
| `WHISPER_MODEL_SIZE` | `base.en` | Whisper model variant |

## Project Structure

```
lite_lfm_audio_api/
├── agent/                    # LiveKit Voice Agent
│   ├── main.py              # Agent entry point
│   ├── config.py            # Configuration
│   ├── stt_plugin.py        # Faster-Whisper STT plugin
│   ├── tts_plugin.py        # LFM Audio TTS plugin
│   └── requirements.txt     # Agent dependencies
├── app/                      # FastAPI REST API
│   ├── main.py              # API endpoints
│   └── models.py            # Pydantic models
├── frontend/                 # Browser voice client
│   ├── index.html           # Voice UI
│   └── Dockerfile           # Nginx container
├── runners/                  # LFM Audio CLI binaries
├── docker-compose.yml        # Full service stack
├── docker-compose.gpu.yml    # GPU override
├── Dockerfile                # API container
├── Dockerfile.agent          # Agent container
├── download_models.sh        # Model download script
└── client_test.py            # Legacy REST client test
```

## Legacy REST Client

The original HTTP test client still works:

```bash
python client_test.py --mode all
```

## License

This project uses the LFM2.5-Audio model which is subject to the [Liquid AI license](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF/blob/main/LICENSE).
