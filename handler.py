"""
RunPod Serverless Handler for Chatterbox TTS
Version: 2.2.2 - Fixed 16-bit PCM WAV output

Parameters:
- text: Text to synthesize  
- audio_url: URL to voice sample for cloning
- exaggeration: Emotion intensity (0-1), default 0.5
- temperature: Controls randomness (0.05-5.0), default 0.8
- cfg_weight: CFG/Pace weight (0.2-1.0), default 0.5
"""

HANDLER_VERSION = "2.2.2"

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
    
    # Save to temp file with correct extension
    ext = ".mp3" if ".mp3" in url else ".wav" if ".wav" in url else ".m4a"
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    temp_file.write(response.content)
    temp_file.close()
    
    voice_file_cache[url] = temp_file.name
    print(f"✅ Saved to {temp_file.name}")
    return temp_file.name


def handler(job):
    """
    RunPod handler - uses audio_prompt_path for voice cloning
    """
    try:
        job_input = job.get("input", {})
        text = job_input.get("text", "")
        audio_url = job_input.get("audio_url")
        exaggeration = float(job_input.get("exaggeration", 0.5))
        temperature = float(job_input.get("temperature", 0.8))
        cfg_weight = float(job_input.get("cfg_weight", 0.5))
        
        if not text:
            return {"error": "No text provided"}
        
        # Truncate text to 300 chars (Chatterbox limit)
        if len(text) > 300:
            print(f"⚠️ Text truncated from {len(text)} to 300 chars")
            text = text[:300]
        
        print(f"🎤 Handler v{HANDLER_VERSION} processing: {text[:80]}...")
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
        
        # Generate audio - returns float32 tensor
        wav = model.generate(text, **generate_kwargs)
        print(f"   Raw wav shape: {wav.shape}, dtype: {wav.dtype}")
        
        # Convert float32 to int16 for backend compatibility
        # The backend's concatenateWavBuffers expects 16-bit PCM
        wav_float = wav.squeeze(0) if wav.dim() > 1 else wav
        wav_normalized = wav_float / wav_float.abs().max()  # Normalize to [-1, 1]
        wav_int16 = (wav_normalized * 32767).to(torch.int16)
        
        # Save as 16-bit PCM WAV
        buffer = io.BytesIO()
        torchaudio.save(buffer, wav_int16.unsqueeze(0), model.sr, format="wav", encoding="PCM_S", bits_per_sample=16)
        buffer.seek(0)
        
        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        duration = wav.shape[-1] / model.sr
        
        print(f"✅ Generated {duration:.2f}s 16-bit PCM audio")
        
        return {
            "audio_base64": audio_base64,
            "duration_seconds": round(duration, 2)
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
