# LFM Audio TTS API

A FastAPI-based Text-to-Speech API using the [LiquidAI LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) model with GGUF quantization for CPU inference.

## Features

- 🎙️ **Text-to-Speech**: Convert text to natural-sounding speech
- 🌊 **SSE Streaming**: Real-time audio delivery via Server-Sent Events
- 🗣️ **Multiple Voices**: Support for US/UK male/female voices
- 🐳 **Docker Ready**: Easy deployment with Docker Compose
- 💻 **CPU Only**: No GPU required - runs on GGUF quantized models

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and model status |
| `/voices` | GET | List available voices |
| `/tts/stream` | POST | TTS with SSE streaming (base64 audio) |
| `/tts` | POST | TTS with direct WAV response |

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- GGUF model files (already included):
  - `LFM2.5-Audio-1.5B-Q4_0.gguf` (main model)
  - `mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf` (multimodal projector)
  - `vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf` (audio vocoder)
  - `tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf` (tokenizer)

### Run with Docker

```bash
# Build and start the container
docker-compose up --build

# The API will be available at http://localhost:8000
```

### API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Usage Examples

### SSE Streaming (JavaScript)

```javascript
const eventSource = new EventSource('/tts/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: 'Hello, this is a test.',
    voice: 'us_male'
  })
});

eventSource.addEventListener('audio', (event) => {
  const data = JSON.parse(event.data);
  const audioBlob = base64ToBlob(data.audio, 'audio/wav');
  // Play or save the audio
});

eventSource.addEventListener('complete', () => {
  eventSource.close();
});
```

### Synchronous Request (curl)

```bash
curl -X POST "http://localhost:8000/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test.", "voice": "default"}' \
  --output speech.wav
```

### Python Client

```python
import requests

response = requests.post(
    "http://localhost:8000/tts",
    json={"text": "Hello, world!", "voice": "uk_female"}
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

## Available Voices

| Voice ID | Name | Language |
|----------|------|----------|
| `default` | Default | English |
| `us_male` | US Male | en-US |
| `us_female` | US Female | en-US |
| `uk_male` | UK Male | en-GB |
| `uk_female` | UK Female | en-GB |

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `/app/models` | Directory containing GGUF model files |
| `RUNNER_PATH` | `/app/runners/llama-liquid-audio-cli` | Path to the llama runner binary |

## Model Files

The GGUF model files were downloaded from [LiquidAI/LFM2.5-Audio-1.5B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF).

Total size: ~1GB (Q4_0 quantization)

## License

This project uses the LFM2.5-Audio model which is subject to the [Liquid AI license](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF/blob/main/LICENSE).
