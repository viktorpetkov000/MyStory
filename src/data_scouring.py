import os
import yt_dlp
from pydub import AudioSegment
from pydub.silence import split_on_silence
from .utils import setup_logging, ensure_dir, get_ffmpeg_path

logger = setup_logging("DataScouring")

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

def segment_audio(audio_path, output_dir, min_silence_len=500, silence_thresh=-40, keep_silence=200):
    """
    Splits audio file into segments based on silence.
    """
    ensure_dir(output_dir)
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
