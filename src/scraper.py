import os
import yt_dlp
from duckduckgo_search import DDGS
from .utils import setup_logging, ensure_dir, get_ffmpeg_path

logger = setup_logging("Scraper")

def search_videos(query, max_results=1):
    """Searches for videos using yt-dlp built-in search."""
    logger.info(f"Searching for: {query}")
    results = []
    try:
        # yt-dlp search syntax: ytsearchN:query
        search_query = f"ytsearch{max_results}:{query}"
        
        ffmpeg_path = get_ffmpeg_path()
        ydl_opts = {
            'quiet': True,
            'extract_flat': True, # Don't download, just extract info
            'force_generic_extractor': False,
        }
        if ffmpeg_path:
            ydl_opts['ffmpeg_location'] = os.path.dirname(ffmpeg_path)
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_query, download=False)
            if 'entries' in info:
                for entry in info['entries']:
                    results.append(entry['url'])
                    
    except Exception as e:
        logger.error(f"Search failed: {e}")
        
    logger.info(f"Found {len(results)} videos.")
    return results

def download_audio(url, output_dir):
    """Downloads audio from a YouTube URL using yt-dlp."""
    ensure_dir(output_dir)
    
    ffmpeg_path = get_ffmpeg_path()
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
    }
    
    if ffmpeg_path:
        ydl_opts['ffmpeg_location'] = os.path.dirname(ffmpeg_path)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Downloading audio from {url}...")
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            # The actual file will have the extension from preferredcodec (wav)
            final_filename = os.path.splitext(filename)[0] + ".wav"
            logger.info(f"Downloaded: {final_filename}")
            return final_filename
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return None

import librosa
import soundfile as sf
import numpy as np

def clean_and_select_segment(input_path, output_path, min_duration=5, max_duration=15):
    """
    Cleans the audio by removing silence and selects the best segment for cloning.
    Criteria: Longest continuous speech segment within min/max duration.
    """
    try:
        logger.info(f"Processing audio for voice isolation: {input_path}")
        
        # Load audio
        y, sr = librosa.load(input_path, sr=22050)
        
        # Split into non-silent intervals (top_db=20 is conservative for voice)
        intervals = librosa.effects.split(y, top_db=20)
        
        best_segment = None
        max_len = 0
        
        for start, end in intervals:
            duration = (end - start) / sr
            
            # We want a segment that is long enough but not too long
            if min_duration <= duration <= max_duration:
                if duration > max_len:
                    max_len = duration
                    best_segment = y[start:end]
            
            # If we have a very long segment, we can just take a slice of it
            elif duration > max_duration:
                # Take the middle part or just the first max_duration
                # Let's take the first max_duration chunk
                end_sample = start + int(max_duration * sr)
                if max_duration > max_len:
                    max_len = max_duration
                    best_segment = y[start:end_sample]

        # If no segment met the criteria, try to find the longest segment > 2s
        if best_segment is None:
            logger.info("No ideal segment found. Searching for longest continuous speech segment...")
            longest_duration = 0
            for start, end in intervals:
                duration = (end - start) / sr
                if duration > longest_duration:
                    longest_duration = duration
                    best_segment = y[start:end]
            
            # If the longest segment is still very short (< 1s), this audio might be bad.
            if longest_duration < 1.0:
                 logger.warning("Longest segment is < 1s. Audio might be mostly silence/noise.")
            else:
                 logger.info(f"Selected fallback segment of {longest_duration:.2f}s")
        
        # If still nothing (e.g. total silence), use trimmed
        if best_segment is None:
            logger.warning("No usable segment found, using trimmed audio.")
            best_segment, _ = librosa.effects.trim(y, top_db=20)

        # Save the selected segment
        sf.write(output_path, best_segment, sr)
        logger.info(f"Saved best audio segment to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        return input_path # Fallback to original

def scour_voice_samples(character_name, output_dir="temp/voice_samples", video_url=None):
    """
    Main function to find and download voice samples.
    If video_url is provided, it uses that specific video instead of searching.
    """
    ensure_dir(output_dir)
    
    downloaded_files = []
    
    if video_url:
        logger.info(f"Using provided video URL: {video_url}")
        urls = [video_url]
    else:
        # Search queries to try
        queries = [
            f"{character_name} voice lines",
            f"{character_name} quotes",
            f"{character_name} interview"
        ]
        urls = []
        for q in queries:
            found = search_videos(q, max_results=1)
            if found:
                urls.extend(found)
                break # Found something, let's try it
    
    for url in urls:
        raw_file_path = download_audio(url, output_dir)
        if raw_file_path:
            # Process the file to get a clean sample
            clean_filename = f"clean_{os.path.basename(raw_file_path)}"
            clean_path = os.path.join(output_dir, clean_filename)
            
            final_path = clean_and_select_segment(raw_file_path, clean_path)
            downloaded_files.append(final_path)
            
            # Stop after finding one good sample to save time/resources
            if final_path:
                return downloaded_files
                
    return downloaded_files
