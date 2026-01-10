import json
import base64
import argparse
import requests
import sseclient  # pip install sseclient-py
import time
import os
from typing import Optional

BASE_URL = "http://localhost:8000"

def test_health():
    """Just checking if the server is awake."""
    print("Checking API health...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        print(f"✅ API is healthy: {json.dumps(response.json(), indent=2)}")
        return True
    except Exception as e:
        print(f"❌ API health check failed: {e}")
        return False

def test_list_voices():
    """Ask the server what voices it has."""
    print("\nListing voices...")
    try:
        response = requests.get(f"{BASE_URL}/voices")
        response.raise_for_status()
        voices = response.json()["voices"]
        print(f"✅ Found {len(voices)} voices:")
        for v in voices:
            print(f"  - {v['id']}: {v['name']} ({v['language']})")
    except Exception as e:
        print(f"❌ Failed to list voices: {e}")

def test_sync_tts(text: str, voice: str, output: str):
    """
    Test the basic TTS. 
    It waits for the full audio file to be ready before downloading it.
    """
    print(f"\nTesting Sync TTS (Voice: {voice})...")
    print(f"Input: '{text}'")
    
    start_time = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/tts",
            json={"text": text, "voice": voice},
            stream=True
        )
        response.raise_for_status()
        
        with open(output, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        duration = time.time() - start_time
        print(f"✅ Saved audio to {output} (took {duration:.2f}s)")
    except Exception as e:
        print(f"❌ Sync TTS failed: {e}")

def test_stream_tts(text: str, voice: str, output: str):
    """
    Test the Streaming TTS. 
    Downloads chunks of audio as they are generated (faster time-to-first-byte).
    """
    print(f"\nTesting Stream TTS (Voice: {voice})...")
    print(f"Input: '{text}'")
    
    start_time = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/tts/stream",
            json={"text": text, "voice": voice},
            stream=True
        )
        response.raise_for_status()
        
        client = sseclient.SSEClient(response)
        
        with open(output, "wb") as f:
            chunk_count = 0
            for event in client.events():
                if event.event == "audio":
                    data = json.loads(event.data)
                    audio_bytes = base64.b64decode(data["audio"])
                    f.write(audio_bytes)
                    chunk_count += 1
                    print(f"\rReceived {chunk_count} chunks...", end="", flush=True)
                elif event.event == "complete":
                    print("\n✅ Generation complete.")
                    break
                elif event.event == "error":
                    data = json.loads(event.data)
                    print(f"\n❌ Server error: {data['message']}")
                    return
        
        duration = time.time() - start_time
        print(f"✅ Saved stream to {output} (took {duration:.2f}s)")
        
    except Exception as e:
        print(f"\n❌ Stream TTS failed: {e}")


def test_all_combinations(text: str):
    """
    The 'Kitchen Sink' test.
    Tries every voice with every method to make sure nothing is broken.
    """
    print("\n" + "="*50)
    print("🚀 STARTING EXHAUSTIVE TEST")
    print("="*50)
    
    try:
        # Get voices first
        response = requests.get(f"{BASE_URL}/voices")
        response.raise_for_status()
        voices = response.json()["voices"]
        
        print(f"Found {len(voices)} voices. Testing each...")
        
        results = {"success": 0, "failed": 0}
        
        for v in voices:
            voice_id = v['id']
            print(f"\n🎤 Testing Voice: {v['name']} ({voice_id})")
            
            # Sync Test
            try:
                test_sync_tts(text, voice_id, f"output_{voice_id}_sync.wav")
                results["success"] += 1
            except Exception:
                results["failed"] += 1
                
            # Stream Test
            try:
                test_stream_tts(text, voice_id, f"output_{voice_id}_stream.wav")
                results["success"] += 1
            except Exception:
                results["failed"] += 1
        
        print("\n" + "="*50)
        print(f"📊 TEST SUMMARY: {results['success']} Passed, {results['failed']} Failed")
        print("="*50)
        
    except Exception as e:
        print(f"❌ Exhaustive test failed: {e}")


def test_stt(audio_file: str):
    """
    Test Speech-to-Text.
    Uploads an audio file and prints what the AI heard.
    """
    print(f"\nTesting STT with {audio_file}...")
    if not os.path.exists(audio_file):
        print("s❌ Audio file not found. Run TTS test first locally to generate one.")
        return

    try:
        start_time = time.time()
        with open(audio_file, "rb") as f:
            files = {"file": (audio_file, f, "audio/wav")}
            response = requests.post(f"{BASE_URL}/stt", files=files)
        
        duration = time.time() - start_time
        response.raise_for_status()
        data = response.json()
        print(f"✅ Transcription Success (took {duration:.2f}s):")
        print(f"   Detected Language: {data['language']}")
        print(f"   Text: {data['text']}")
        
    except Exception as e:
        print(f"❌ STT failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LFM TTS API")
    parser.add_argument("--mode", choices=["sync", "stream", "stt", "all", "exhaustive"], default="all", help="Test mode (exhaustive tests all voices)")

    parser.add_argument("--text", default="Human rights attorney Amal Clooney has it all: a successful career, one of the most coveted husbands of Hollywood, and an impeccable fashion sense. Amal is always dressed for the occasion and surely has one or two things to teach us about fashion.", help="Text to speak")
    parser.add_argument("--voice", default="us_male", help="Voice ID to use (ignored in exhaustive mode)")
    
    args = parser.parse_args()
    
    if test_health():
        if args.mode == "exhaustive":
            test_all_combinations(args.text)
        else:
            test_list_voices()
            
            if args.mode in ["sync", "all"]:
                test_sync_tts(args.text, args.voice, "output_sync.wav")
                
            if args.mode in ["stream", "all"]:
                test_stream_tts(args.text, args.voice, "output_stream.wav")

            if args.mode in ["stt", "all"]:
                # Use one of the generated files if available, or warn
                input_file = "output_sync.wav" 
                test_stt(input_file)
