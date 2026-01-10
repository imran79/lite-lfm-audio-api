"""
FastAPI SSE Text-to-Speech API using LFM2.5-Audio-1.5B GGUF model.

This API provides text-to-speech conversion using the Liquid AI LFM2.5 model
with Server-Sent Events (SSE) for streaming audio output.
"""
import os
import base64
import json
import asyncio
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from faster_whisper import WhisperModel
import shutil

from app.models import (
    TTSRequest,
    VoiceType,
    VoicesResponse,
    VoiceInfo,
    HealthResponse,
    STTResponse,
)


app = FastAPI(
    title="LFM Audio TTS API",
    description="Text-to-Speech API using LFM2.5-Audio-1.5B GGUF with SSE streaming",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment variables
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))
RUNNER_PATH = Path(os.getenv("RUNNER_PATH", "/app/runners/llama-liquid-audio-cli"))

# Voice prompts mapping
VOICE_PROMPTS = {
    "default": "Perform TTS.",
    "us_male": "Perform TTS. Use the US male voice.",
    "us_female": "Perform TTS. Use the US female voice.",
    "uk_male": "Perform TTS. Use the UK male voice.",
    "uk_female": "Perform TTS. Use the UK female voice.",
}


# Global STT Model
stt_model = None

def get_stt_model():
    """Lazy load the STT model."""
    global stt_model
    if stt_model is None:
        model_path = MODEL_DIR / "whisper"
        print(f"Loading Whisper model from {model_path}...")
        stt_model = WhisperModel("base.en", device="cpu", compute_type="int8", download_root=str(model_path))
    return stt_model

def get_model_files() -> dict[str, Path]:
    """
    Get paths to required GGUF model files.
    
    Returns:
        Dictionary with paths to main model, mmproj, vocoder, and tokenizer.
        
    Raises:
        ValueError: If any required model files are missing.
    """
    # Find model files - prefer Q4_0 quantization
    main_model = list(MODEL_DIR.glob("LFM2.5-Audio-1.5B-Q4_0.gguf"))
    if not main_model:
        main_model = list(MODEL_DIR.glob("LFM2.5-Audio-1.5B-*.gguf"))
    
    mmproj = list(MODEL_DIR.glob("mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf"))
    if not mmproj:
        mmproj = list(MODEL_DIR.glob("mmproj-LFM2.5-Audio-1.5B-*.gguf"))
    
    vocoder = list(MODEL_DIR.glob("vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf"))
    if not vocoder:
        vocoder = list(MODEL_DIR.glob("vocoder-LFM2.5-Audio-1.5B-*.gguf"))
    
    tokenizer = list(MODEL_DIR.glob("tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf"))
    if not tokenizer:
        tokenizer = list(MODEL_DIR.glob("tokenizer-LFM2.5-Audio-1.5B-*.gguf"))
    
    # Check for missing files
    missing = []
    if not main_model:
        missing.append("main model (LFM2.5-Audio-1.5B-*.gguf)")
    if not mmproj:
        missing.append("mmproj (mmproj-LFM2.5-Audio-1.5B-*.gguf)")
    if not vocoder:
        missing.append("vocoder (vocoder-LFM2.5-Audio-1.5B-*.gguf)")
    if not tokenizer:
        missing.append("tokenizer (tokenizer-LFM2.5-Audio-1.5B-*.gguf)")
    
    if missing:
        raise ValueError(f"Missing model files: {', '.join(missing)}")
    
    return {
        "main": main_model[0],
        "mmproj": mmproj[0],
        "vocoder": vocoder[0],
        "tokenizer": tokenizer[0],
    }


async def run_tts_generation(
    text: str,
    voice: str,
    output_path: str,
) -> tuple[bool, str]:
    """
    Run TTS generation using the llama-liquid-audio-cli runner.
    
    Args:
        text: Input text to convert to speech.
        voice: Voice type to use.
        output_path: Path to save the output WAV file.
        
    Returns:
        Tuple of (success, error_message).
    """
    try:
        models = get_model_files()
        system_prompt = VOICE_PROMPTS.get(voice, VOICE_PROMPTS["default"])
        
        # Build CLI command
        cmd = [
            str(RUNNER_PATH),
            "-m", str(models["main"]),
            "-mm", str(models["mmproj"]),
            "-mv", str(models["vocoder"]),
            "--tts-speaker-file", str(models["tokenizer"]),
            "-sys", system_prompt,
            "-p", text,
            "--output", output_path,
        ]
        
        # Run the CLI tool
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            return False, f"TTS generation failed: {error_msg}"
        
        return True, ""
        
    except ValueError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the health status of the API and verifies model files are present.
    """
    try:
        models = get_model_files()
        return HealthResponse(
            status="healthy",
            model_dir=str(MODEL_DIR),
        )
    except ValueError as e:
        return HealthResponse(
            status="unhealthy",
            model_dir=str(MODEL_DIR),
            error=str(e),
        )


@app.get("/voices", response_model=VoicesResponse)
async def list_voices():
    """
    List available TTS voices.
    
    Returns information about all supported voice types.
    """
    return VoicesResponse(
        voices=[
            VoiceInfo(id="default", name="Default", language="en"),
            VoiceInfo(id="us_male", name="US Male", language="en-US"),
            VoiceInfo(id="us_female", name="US Female", language="en-US"),
            VoiceInfo(id="uk_male", name="UK Male", language="en-GB"),
            VoiceInfo(id="uk_female", name="UK Female", language="en-GB"),
        ]
    )


@app.post("/tts/stream")
async def tts_stream(request: TTSRequest):
    """
    Convert text to speech with SSE streaming.
    
    Returns base64-encoded WAV audio via Server-Sent Events.
    
    Event types:
    - **audio**: Contains the generated audio data
        - `audio`: base64-encoded WAV data
        - `format`: "wav"
        - `sample_rate`: 24000
    - **complete**: Indicates successful completion
        - `success`: true
    - **error**: Indicates an error occurred
        - `message`: Error description
    """
    
    async def generate():
        output_path = None
        try:
            # Create temp file for output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                output_path = tmp.name
            
            # Run TTS generation
            success, error = await run_tts_generation(
                text=request.text,
                voice=request.voice.value,
                output_path=output_path,
            )
            
            if not success:
                yield {
                    "event": "error",
                    "data": json.dumps({"message": error}),
                }
                return
            
            # Read and encode the output audio
            with open(output_path, "rb") as f:
                audio_data = f.read()
            
            encoded = base64.b64encode(audio_data).decode("utf-8")
            
            yield {
                "event": "audio",
                "data": json.dumps({
                    "audio": encoded,
                    "format": "wav",
                    "sample_rate": 24000,
                }),
            }
            
            yield {
                "event": "complete",
                "data": json.dumps({"success": True}),
            }
            
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)}),
            }
        finally:
            # Cleanup temp file
            if output_path and os.path.exists(output_path):
                os.unlink(output_path)
    
    return EventSourceResponse(generate())


@app.post("/tts")
async def tts_sync(request: TTSRequest):
    """
    Convert text to speech (synchronous).
    
    Returns WAV audio directly as a binary response.
    This is useful for clients that don't support SSE.
    """
    output_path = None
    try:
        # Create temp file for output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name
        
        # Run TTS generation
        success, error = await run_tts_generation(
            text=request.text,
            voice=request.voice.value,
            output_path=output_path,
        )
        
        if not success:
            raise HTTPException(status_code=500, detail=error)
        
        # Read the output audio
        with open(output_path, "rb") as f:
            audio_data = f.read()
        
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
            },
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp file
        if output_path and os.path.exists(output_path):
            os.unlink(output_path)

        if output_path and os.path.exists(output_path):
            os.unlink(output_path)


@app.post("/stt", response_model=STTResponse)
async def speech_to_text(file: UploadFile = File(...)):
    """
    Convert speech to text.
    
    Accepts an audio file upload (wav, mp3, etc.) and returns the transcription.
    """
    temp_filename = None
    try:
        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_filename = tmp.name
            
        # Get model and transcribe
        model = get_stt_model()
        segments, info = model.transcribe(temp_filename, beam_size=5)
        
        # Collect results
        result_text = ""
        segment_list = []
        
        for segment in segments:
            result_text += segment.text
            segment_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })
            
        return STTResponse(
            text=result_text.strip(),
            language=info.language,
            segments=segment_list
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {str(e)}")
    finally:
        if temp_filename and os.path.exists(temp_filename):
            os.unlink(temp_filename)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
