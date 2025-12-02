import logging
import os
import shutil

def setup_logging(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def clean_temp(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    ensure_dir(path)

def get_ffmpeg_path():
    """
    Returns the path to the ffmpeg executable.
    Prioritizes the local project bin directory.
    """
    # 1. Check local project bin
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    local_ffmpeg = os.path.join(project_root, 'bin', 'ffmpeg', 'bin', 'ffmpeg.exe')
    if os.path.exists(local_ffmpeg):
        return local_ffmpeg

    # 2. Check system PATH
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
        
    # 3. Check Winget location (fallback)
    possible_paths = [
        r"C:\Users\vikto\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
            
    return None

def split_text(text):
    """
    Splits text into sentences using simple regex.
    Avoids NLTK dependency.
    """
    import re
    # Split by . ! ? followed by space or end of string
    # Keep the punctuation with the sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]
