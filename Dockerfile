# RunPod Serverless Chatterbox TTS
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    runpod \
    torchaudio \
    chatterbox-tts

# Copy handler
COPY handler.py /app/handler.py

# Pre-download model during build (faster cold starts)
RUN python -c "from chatterbox.tts import ChatterboxTTS; ChatterboxTTS.from_pretrained(device='cpu')"

# Start handler
CMD ["python", "-u", "/app/handler.py"]

