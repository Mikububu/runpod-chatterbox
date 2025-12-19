"""
RunPod Serverless Handler for Chatterbox TTS
Version: 2.1.1 - WAV output for backend stitching (backend converts to MP3)

Parameters:
- text: Text to synthesize  
- audio_url: URL to voice sample for cloning (WAV format recommended)
- exaggeration: Emotion intensity (0-1), default 0.5
"""

HANDLER_VERSION = "2.1.1"

import runpod
import torch
import torchaudio
import base64
import io
import os
import tempfile
import requests
from chatterbox.tts import ChatterboxTTS

# Initialize model on startup
print(f"🔊 Loading Chatterbox TTS model... (Handler v{HANDLER_VERSION})")
model = ChatterboxTTS.from_pretrained(device="cuda")
print(f"✅ Model loaded! Handler version {HANDLER_VERSION}")

# Cache downloaded voice samples
voice_file_cache = {}


def download_voice_to_file(url: str) -> str:
    """Download voice sample to temp file, return path"""
    if url in voice_file_cache and os.path.exists(voice_file_cache[url]):
        print(f"📦 Using cached voice file")
        return voice_file_cache[url]
    
    print(f"⬇️ Downloading voice from {url[:60]}...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    # Save to temp file - always use .wav for compatibility
    # Chatterbox works best with WAV input
    if ".wav" in url:
        ext = ".wav"
    elif ".mp3" in url:
        ext = ".mp3"
    else:
        ext = ".wav"
    
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    temp_file.write(response.content)
    temp_file.close()
    
    voice_file_cache[url] = temp_file.name
    print(f"✅ Saved to {temp_file.name}")
    return temp_file.name


def handler(job):
    """
    RunPod handler - outputs WAV for backend stitching
    Backend will convert final stitched audio to MP3
    """
    try:
        job_input = job.get("input", {})
        text = job_input.get("text", "")
        audio_url = job_input.get("audio_url")
        exaggeration = float(job_input.get("exaggeration", 0.5))
        
        if not text:
            return {"error": "No text provided"}
        
        print(f"🎤 Text: {text[:80]}...")
        print(f"   exaggeration={exaggeration}")
        print(f"   Voice URL: {audio_url[:60] if audio_url else 'None'}...")
        
        # Generate with or without voice cloning
        if audio_url:
            voice_path = download_voice_to_file(audio_url)
            print(f"🎯 Using audio_prompt_path: {voice_path}")
            wav = model.generate(
                text,
                audio_prompt_path=voice_path,
                exaggeration=exaggeration
            )
        else:
            wav = model.generate(
                text,
                exaggeration=exaggeration
            )
        
        # Output as WAV (backend will stitch and convert to MP3)
        buffer = io.BytesIO()
        torchaudio.save(buffer, wav, 24000, format="wav")
        buffer.seek(0)
        
        audio_bytes = buffer.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        duration = wav.shape[1] / 24000
        size_kb = len(audio_bytes) / 1024
        
        print(f"✅ Generated {duration:.2f}s audio ({size_kb:.1f} KB WAV)")
        
        return {
            "audio_base64": audio_base64,
            "duration_seconds": round(duration, 2),
            "format": "wav",
            "size_kb": round(size_kb, 1)
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
