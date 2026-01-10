#!/bin/bash
set -e

MODEL_DIR=${MODEL_DIR:-/app/models}
RUNNER_DIR=${RUNNER_DIR:-/app/runners}

echo "Checking model files in $MODEL_DIR..."

# Function to download file if not exists
download_if_missing() {
    local filename=$1
    local url=$2
    local filepath="$MODEL_DIR/$filename"

    if [ ! -f "$filepath" ]; then
        echo "Downloading $filename..."
        # Use curl with location following (-L) and show progress
        curl -L -o "$filepath" "$url"
    else
        echo "$filename already exists."
    fi
}

# Ensure model directory exists
mkdir -p "$MODEL_DIR"

# Download GGUF files (Q4_0 quantization)
BASE_URL="https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF/resolve/main"

download_if_missing "LFM2.5-Audio-1.5B-Q4_0.gguf" "$BASE_URL/LFM2.5-Audio-1.5B-Q4_0.gguf"
download_if_missing "mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf" "$BASE_URL/mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf"
download_if_missing "vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf" "$BASE_URL/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf"
download_if_missing "tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf" "$BASE_URL/tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf"

# Also check for runner
if [ ! -f "$RUNNER_DIR/llama-liquid-audio-cli" ]; then
    echo "Checking for runner binary..."
    # If the runner isn't there (because it wasn't copied from host), try to download it
    # Note: Automation of runner download is tricky due to unzipping, but we'll attempt it
    RUNNER_URL="$BASE_URL/runners/llama-liquid-audio-ubuntu-x64.zip"
    ZIP_PATH="/tmp/runner.zip"
    
    echo "Downloading runner..."
    curl -L -o "$ZIP_PATH" "$RUNNER_URL"
    
    echo "Extracting runner..."
    unzip -o "$ZIP_PATH" -d "/tmp/runners-extracted"
    mv /tmp/runners-extracted/llama-liquid-audio-ubuntu-x64/* "$RUNNER_DIR/"
    rm -rf "/tmp/runners-extracted"
    
    chmod +x "$RUNNER_DIR/llama-liquid-audio-cli"
    rm "$ZIP_PATH"
fi

echo "All required files are present."
exec "$@"
