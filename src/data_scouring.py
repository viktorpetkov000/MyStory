import os
import yt_dlp
import numpy as np
import scipy.io.wavfile as wav
import noisereduce as nr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from .utils import setup_logging, ensure_dir, get_ffmpeg_path

logger = setup_logging("DataScouring")

def clean_audio(audio_path):
    """
    Removes background noise from the audio file using spectral gating.
    Overwrites the input file or returns path to cleaned version.
    """
    logger.info(f"Cleaning background noise from: {audio_path}")
    try:
        # Load audio using scipy (faster for arrays)
        rate, data = wav.read(audio_path)
        
        # Check shape: scipy returns (samples, channels) but noisereduce wants (channels, samples)
        if len(data.shape) > 1:
            data = data.T
        
        # Perform noise reduction
        reduced_noise = nr.reduce_noise(
            y=data, 
            sr=rate, 
            stationary=True,
            chunk_size=600000 # Use safer chunk size (~13s)
        )
        
        # Transpose back to (samples, channels) for writing
        if len(reduced_noise.shape) > 1:
            reduced_noise = reduced_noise.T
        
        # Save back
        cleaned_path = audio_path.replace(".wav", "_clean.wav")
        wav.write(cleaned_path, rate, reduced_noise)
        
        logger.info(f"Cleaned audio saved to: {cleaned_path}")
        return cleaned_path
    
    except Exception as e:
        logger.error(f"Failed to clean audio: {e}")
        return audio_path # Return original if failure

def download_audio(youtube_url, output_dir):
    """
    Downloads audio from a YouTube video using yt-dlp.
    Returns the path to the downloaded wav file.
    """
    ensure_dir(output_dir)
    
    # Configure yt-dlp
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'ffmpeg_location': get_ffmpeg_path()
    }
    
    try:
        logger.info(f"Downloading audio from {youtube_url}...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            filename = ydl.prepare_filename(info)
            # yt-dlp changes extension after post-processing
            wav_filename = os.path.splitext(filename)[0] + ".wav"
            
        if os.path.exists(wav_filename):
            logger.info(f"Audio downloaded to {wav_filename}")
            return wav_filename
        else:
            logger.error("Downloaded file not found (extension mismatch?)")
            return None
            
    except Exception as e:
        logger.error(f"Failed to download audio: {e}")
        return None

def segment_audio(audio_path, output_dir, min_silence_len=500, silence_thresh=-40, keep_silence=200, remove_noise=False):
    """
    Splits audio file into segments based on silence.
    Optional: remove_noise (bool) to clean audio before splitting.
    """
    ensure_dir(output_dir)
    
    if remove_noise:
        # clear audio before loading into pydub
        cleaned_path = clean_audio(audio_path)
        if cleaned_path != audio_path:
            audio_path = cleaned_path
            
    logger.info(f"Segmenting audio: {audio_path}")
    
    try:
        # Load audio
        sound = AudioSegment.from_wav(audio_path)
        
        # Split on silence
        chunks = split_on_silence(
            sound,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence
        )
        
        logger.info(f"Found {len(chunks)} segments.")
        
        saved_paths = []
        for i, chunk in enumerate(chunks):
            # Filter out very short chunks (< 1 sec)
            if len(chunk) < 1000:
                continue
                
            chunk_name = f"segment_{i:03d}.wav"
            chunk_path = os.path.join(output_dir, chunk_name)
            chunk.export(chunk_path, format="wav")
            saved_paths.append(chunk_path)
            
        logger.info(f"Saved {len(saved_paths)} valid segments to {output_dir}")
        return saved_paths
        
    except Exception as e:
        logger.error(f"Failed to segment audio: {e}")
        return []
