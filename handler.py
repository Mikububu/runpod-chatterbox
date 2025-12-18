"""
RunPod Serverless Handler for Chatterbox TTS
Generates audio from text using the Chatterbox model
"""

import runpod
import torch
import base64
import io
import os
from chatterbox.tts import ChatterboxTTS

# Initialize model on startup (cold start ~30s)
print("🔊 Loading Chatterbox TTS model...")
model = ChatterboxTTS.from_pretrained(device="cuda")
print("✅ Model loaded!")


def handler(job):
    """
    RunPod handler function
    
    Input:
    {
        "input": {
            "text": "Hello, this is a test.",
            "voice": "neutral"  # optional
        }
    }
    
    Output:
    {
        "audio_base64": "...",
        "duration_seconds": 3.5
    }
    """
    try:
        job_input = job.get("input", {})
        text = job_input.get("text", "")
        
        if not text:
            return {"error": "No text provided"}
        
        print(f"🎤 Generating audio for: {text[:100]}...")
        
        # Generate audio
        wav = model.generate(text)
        
        # Convert to bytes
        import torchaudio
        buffer = io.BytesIO()
        torchaudio.save(buffer, wav, 24000, format="wav")
        buffer.seek(0)
        
        # Encode as base64
        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        
        # Calculate duration
        duration = wav.shape[1] / 24000
        
        print(f"✅ Generated {duration:.2f}s of audio")
        
        return {
            "audio_base64": audio_base64,
            "duration_seconds": round(duration, 2)
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"error": str(e)}


# Start the serverless handler
runpod.serverless.start({"handler": handler})

