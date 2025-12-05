import logging
import os
import shutil
import subprocess
import time
import socket

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

def get_ollama_host_port():
    """Parses OLLAMA_HOST env var or returns defaults."""
    ollama_host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    if ":" in ollama_host:
        host, port = ollama_host.split(":")
        return host, int(port)
    return ollama_host, 11434

def is_ollama_running():
    """
    Checks if Ollama is running by attempting to connect to its API port.
    Respects OLLAMA_HOST environment variable.
    """
    host, port = get_ollama_host_port()
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False

def start_ollama(max_wait=30):
    """
    Starts Ollama if it's not already running.
    Returns True if Ollama is running (either started or was already running).
    """
    logger = setup_logging("Ollama")
    
    if is_ollama_running():
        logger.info("Ollama is already running.")
        return True
    
    logger.info("Ollama is not running. Starting it...")
    
    try:
        # Start ollama serve in the background
        # It inherits environment variables (OLLAMA_HOST, OLLAMA_MODELS) from this process
        
        # Use CREATE_NO_WINDOW on Windows to prevent a console window
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0  # SW_HIDE
        
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            startupinfo=startupinfo
        )
        
        # Wait for Ollama to be ready
        logger.info("Waiting for Ollama to start...")
        start_time = time.time()
        while time.time() - start_time < max_wait:
            if is_ollama_running():
                logger.info("Ollama started successfully.")
                return True
            time.sleep(0.5)
        
        logger.error(f"Ollama did not start within {max_wait} seconds.")
        return False
        
    except FileNotFoundError:
        logger.error("Ollama executable not found. Please install Ollama from https://ollama.com/")
        return False
    except Exception as e:
        logger.error(f"Failed to start Ollama: {e}")
        return False
