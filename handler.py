"""
RunPod Serverless Handler for Chatterbox TTS
Version: 2.1.0 - MP3 output for smaller file sizes

Parameters:
- text: Text to synthesize  
- audio_url: URL to voice sample for cloning (WAV format recommended)
- exaggeration: Emotion intensity (0-1), default 0.5
- output_format: "mp3" (default) or "wav"
"""

HANDLER_VERSION = "2.1.0"

import runpod
import torch
import torchaudio
import base64
import io
import os
import tempfile
import subprocess
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
    
    # Save to temp file with correct extension
    ext = ".mp3" if ".mp3" in url else ".wav" if ".wav" in url else ".m4a"
    temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    temp_file.write(response.content)
    temp_file.close()
    
    voice_file_cache[url] = temp_file.name
    print(f"✅ Saved to {temp_file.name}")
    return temp_file.name


def wav_to_mp3(wav_tensor, sample_rate=24000):
    """Convert WAV tensor to MP3 bytes using ffmpeg"""
    # Save WAV to temp file
    wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    torchaudio.save(wav_file.name, wav_tensor, sample_rate, format="wav")
    wav_file.close()
    
    # Convert to MP3 using ffmpeg
    mp3_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    mp3_file.close()
    
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", wav_file.name,
            "-codec:a", "libmp3lame", "-b:a", "128k",
            mp3_file.name
        ], check=True, capture_output=True)
        
        with open(mp3_file.name, "rb") as f:
            mp3_bytes = f.read()
        
        return mp3_bytes
    finally:
        # Cleanup temp files
        os.unlink(wav_file.name)
        os.unlink(mp3_file.name)


def handler(job):
    """
    RunPod handler - outputs MP3 for smaller file sizes
    """
    try:
        job_input = job.get("input", {})
        text = job_input.get("text", "")
        audio_url = job_input.get("audio_url")
        exaggeration = float(job_input.get("exaggeration", 0.5))
        output_format = job_input.get("output_format", "mp3")  # mp3 or wav
        
        if not text:
            return {"error": "No text provided"}
        
        print(f"🎤 Text: {text[:80]}...")
        print(f"   exaggeration={exaggeration}, output={output_format}")
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
        
        duration = wav.shape[1] / 24000
        
        # Convert to requested format
        if output_format == "mp3":
            print(f"🔄 Converting to MP3...")
            audio_bytes = wav_to_mp3(wav, 24000)
            content_type = "audio/mpeg"
        else:
            buffer = io.BytesIO()
            torchaudio.save(buffer, wav, 24000, format="wav")
            buffer.seek(0)
            audio_bytes = buffer.read()
            content_type = "audio/wav"
        
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        size_kb = len(audio_bytes) / 1024
        print(f"✅ Generated {duration:.2f}s audio ({size_kb:.1f} KB {output_format})")
        
        return {
            "audio_base64": audio_base64,
            "duration_seconds": round(duration, 2),
            "format": output_format,
            "size_kb": round(size_kb, 1)
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
