# Chatterbox TTS on RunPod

## Option 1: Use Pre-built Image (Easiest)

1. Go to RunPod → Serverless → New Endpoint
2. Settings:
   - Name: `chatterbox-tts`
   - GPU: `A10G 24GB`
   - Container Image: `mikububu/chatterbox-tts:latest` (I'll build this)
   - Min Workers: 0
   - Max Workers: 1

## Option 2: Build Your Own

```bash
# Build
docker build -t your-dockerhub/chatterbox-tts:latest .

# Push to Docker Hub
docker push your-dockerhub/chatterbox-tts:latest
```

## API Usage

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "Hello, this is a test of the Chatterbox text to speech system."
    }
  }'
```

## Response

```json
{
  "output": {
    "audio_base64": "UklGRi...",
    "duration_seconds": 3.5
  }
}
```

## Cost Estimate

- A10G: ~$0.30/hour
- Average generation: ~5 seconds per minute of audio
- Nuclear Package (60 min audio): ~$0.50-1.00 total

