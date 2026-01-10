FROM ubuntu:24.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    unzip \
    curl \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Create virtual environment to satisfy PEP 668
ENV VIRTUAL_ENV=/app/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Create directories for models and runners
RUN mkdir -p /app/models /app/runners

# Copy the runner zip and extract it (handling subdirectory)
COPY runners/llama-liquid-audio-ubuntu-x64.zip /tmp/
RUN unzip /tmp/llama-liquid-audio-ubuntu-x64.zip -d /tmp/extracted \
    && mv /tmp/extracted/llama-liquid-audio-ubuntu-x64/* /app/runners/ \
    && rm -rf /tmp/extracted \
    && chmod +x /app/runners/llama-liquid-audio-cli \
    && chmod +x /app/runners/llama-liquid-audio-server 2>/dev/null || true \
    && rm /tmp/llama-liquid-audio-ubuntu-x64.zip

# Copy application code
COPY app/ ./app/
COPY download_models.sh .
RUN chmod +x download_models.sh

# Expose port
EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app/models
ENV RUNNER_PATH=/app/runners/llama-liquid-audio-cli

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run download script then the application
ENTRYPOINT ["./download_models.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
