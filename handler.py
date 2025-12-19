"""
RunPod Serverless Handler for Chatterbox TTS
Generates audio from text using the Chatterbox model with voice cloning

IMPORTANT PARAMETERS (from HuggingFace docs):
- exaggeration (default 0.5): Emotion intensity (0-1)
- cfg (default 0.5): Classifier-free guidance weight
  - Lower cfg (~0.3) = slower, more deliberate pacing
  - Higher cfg = faster speech
  - For voice cloning with fast speakers, use cfg=0.3
  
Tips from docs:
- General use: exaggeration=0.5, cfg=0.5
- Expressive/dramatic: exaggeration=0.7, cfg=0.3
- Voice cloning: cfg=0.3 helps match reference better
"""

import runpod
import torch
import torchaudio
import base64
import io
import os
import requests
from chatterbox.tts import ChatterboxTTS

# Initialize model on startup (cold start ~30s)
print("🔊 Loading Chatterbox TTS model...")
model = ChatterboxTTS.from_pretrained(device="cuda")
print("✅ Model loaded!")

# Cache for voice samples to avoid re-downloading
voice_cache = {}


def download_voice_sample(url: str) -> torch.Tensor:
    """Download and cache voice sample from URL"""
    if url in voice_cache:
        print(f"📦 Using cached voice sample")
        return voice_cache[url]
    
    print(f"⬇️ Downloading voice sample from {url[:50]}...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    # Load audio from bytes
    audio_bytes = io.BytesIO(response.content)
    waveform, sample_rate = torchaudio.load(audio_bytes)
    
    # Resample to 24000Hz if needed (Chatterbox expects 24kHz)
    if sample_rate != 24000:
        resampler = torchaudio.transforms.Resample(sample_rate, 24000)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Cache it
    voice_cache[url] = waveform
    print(f"✅ Voice sample loaded and cached ({waveform.shape[1]/24000:.1f}s)")
    
    return waveform


def handler(job):
    """
    RunPod handler function
    
    Input:
    {
        "input": {
            "text": "Hello, this is a test.",
            "audio_url": "https://example.com/voice_sample.mp3",  # optional - for voice cloning
            "exaggeration": 0.5,  # emotion intensity (0-1), default 0.5
            "cfg": 0.5  # classifier-free guidance (0-1), default 0.5, lower=slower/clearer
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
        audio_url = job_input.get("audio_url", None)
        
        # Get parameters with proper defaults from HuggingFace docs
        exaggeration = float(job_input.get("exaggeration", 0.5))  # Default 0.5
        cfg = float(job_input.get("cfg", 0.5))  # Default 0.5 (CRITICAL for quality!)
        
        if not text:
            return {"error": "No text provided"}
        
        print(f"🎤 Generating audio for: {text[:100]}...")
        print(f"   exaggeration: {exaggeration}")
        print(f"   cfg: {cfg}")
        print(f"   Voice cloning: {'Yes' if audio_url else 'No (default voice)'}")
        
        # Load voice sample if URL provided
        audio_prompt = None
        if audio_url:
            try:
                audio_prompt = download_voice_sample(audio_url)
            except Exception as e:
                print(f"⚠️ Failed to load voice sample: {e}, using default voice")
                audio_prompt = None
        
        # Generate audio with proper parameters
        if audio_prompt is not None:
            wav = model.generate(
                text,
                audio_prompt=audio_prompt,
                exaggeration=exaggeration,
                cfg=cfg  # CRITICAL: This was missing!
            )
        else:
            wav = model.generate(
                text,
                exaggeration=exaggeration,
                cfg=cfg
            )
        
        # Convert to bytes
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
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# Start the serverless handler
runpod.serverless.start({"handler": handler})
