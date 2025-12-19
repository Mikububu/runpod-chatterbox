"""
RunPod Serverless Handler for Chatterbox TTS
Version: 2.2.0 - Correct parameters from HuggingFace Space

Parameters (from official HF Space):
- text: Text to synthesize (max 300 chars per chunk)
- audio_url: URL to voice sample for cloning
- exaggeration: 0.25-2.0 (default 0.5, neutral)
- temperature: 0.05-5.0 (default 0.8)
- cfg_weight: 0.2-1.0 (default 0.5) - CFG/Pace control

Output: WAV at 24kHz
"""

HANDLER_VERSION = "2.2.0"

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
print(f"   Sample rate: {model.sr}Hz")

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
    
    # Determine extension from URL
    if ".wav" in url.lower():
        ext = ".wav"
    elif ".mp3" in url.lower():
        ext = ".mp3"
    elif ".flac" in url.lower():
        ext = ".flac"
    elif ".m4a" in url.lower():
        ext = ".m4a"
    elif ".ogg" in url.lower():
        ext = ".ogg"
    else:
        ext = ".wav"
    
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    temp_file.write(response.content)
    temp_file.close()
    
    voice_file_cache[url] = temp_file.name
    print(f"✅ Saved to {temp_file.name} ({len(response.content)} bytes)")
    return temp_file.name


def handler(job):
    """
    RunPod handler with correct Chatterbox parameters
    """
    try:
        job_input = job.get("input", {})
        text = job_input.get("text", "")
        audio_url = job_input.get("audio_url")
        
        # Parameters with correct defaults from HF Space
        exaggeration = float(job_input.get("exaggeration", 0.5))
        temperature = float(job_input.get("temperature", 0.8))
        cfg_weight = float(job_input.get("cfg_weight", 0.5))
        
        if not text:
            return {"error": "No text provided"}
        
        # Truncate text to 300 chars (Chatterbox limit)
        if len(text) > 300:
            print(f"⚠️ Text truncated from {len(text)} to 300 chars")
            text = text[:300]
        
        print(f"🎤 Text: {text[:80]}...")
        print(f"   exaggeration={exaggeration}, temperature={temperature}, cfg_weight={cfg_weight}")
        
        # Build generate kwargs
        generate_kwargs = {
            "exaggeration": exaggeration,
            "temperature": temperature,
            "cfg_weight": cfg_weight,
        }
        
        # Add voice cloning if URL provided
        if audio_url:
            voice_path = download_voice_to_file(audio_url)
            generate_kwargs["audio_prompt_path"] = voice_path
            print(f"🎯 Voice cloning: {voice_path}")
        
        # Generate audio
        wav = model.generate(text, **generate_kwargs)
        
        # Convert to 16-bit PCM (required for backend stitching)
        # Chatterbox outputs float32, we need int16
        wav_16bit = (wav * 32767).to(torch.int16)
        
        # Save as WAV (24kHz, 16-bit PCM mono)
        buffer = io.BytesIO()
        torchaudio.save(buffer, wav_16bit, model.sr, format="wav", encoding="PCM_S", bits_per_sample=16)
        buffer.seek(0)
        
        audio_bytes = buffer.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        duration = wav.shape[1] / model.sr
        size_kb = len(audio_bytes) / 1024
        
        print(f"✅ Generated {duration:.2f}s audio ({size_kb:.1f} KB WAV @ {model.sr}Hz)")
        
        return {
            "audio_base64": audio_base64,
            "duration_seconds": round(duration, 2),
            "format": "wav",
            "sample_rate": model.sr,
            "size_kb": round(size_kb, 1)
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
