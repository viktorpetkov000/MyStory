import os
import torch
import time
import numpy as np
import torchaudio
import soundfile as sf
import huggingface_hub
# Patch for Coqui TTS compatibility with huggingface_hub >= 0.25.0
if not hasattr(huggingface_hub, 'cached_download'):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

from TTS.api import TTS
from .utils import setup_logging, ensure_dir, split_text

# Monkey-patch torch.load to handle weights_only=True default in newer PyTorch versions
_original_load = torch.load
def safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load

# Monkey-patch torchaudio.load and save to use soundfile directly
def safe_audio_load(filepath, **kwargs):
    try:
        data, sr = sf.read(filepath)
        # soundfile returns (frames, channels), torch expects (channels, frames)
        if data.ndim == 2:
            data = data.T
        else:
            data = data[None, :] # Add channel dim for mono
        return torch.from_numpy(data).float(), sr
    except Exception as e:
        logger.error(f"Soundfile load failed: {e}")
        raise e

def safe_audio_save(filepath, src, sample_rate, **kwargs):
    try:
        # src is (channels, frames), soundfile wants (frames, channels)
        if src.ndim == 2:
            src = src.t()
        # Convert to numpy
        if isinstance(src, torch.Tensor):
            src = src.detach().cpu().numpy()
        sf.write(filepath, src, sample_rate)
    except Exception as e:
        logger.error(f"Soundfile save failed: {e}")
        raise e

torchaudio.load = safe_audio_load
torchaudio.save = safe_audio_save

logger = setup_logging("AudioGen")
_TTS_MODEL = None

def load_tts_model():
    global _TTS_MODEL
    if _TTS_MODEL is not None:
        return _TTS_MODEL

    logger.info("load_tts_model called. Initializing TTS...")
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device for loading: {device}")
    
    if device == "cuda":
        # Disable specific optimizations that might hang on RTX 50-series
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        logger.info("Disabled CuDNN benchmark and TF32 for stability.")

    try:
        # Initialize TTS
        model_dir = os.path.join(os.environ["TTS_HOME"], "tts", "tts_models--multilingual--multi-dataset--xtts_v2")
        model_path = os.path.join(model_dir, "model.pth")
        config_path = os.path.join(model_dir, "config.json")
        
        logger.info(f"Checking for TTS model at: {model_path}")
        
        tts = None
        if os.path.exists(model_path):
             logger.info("Local model file found. Instantiating TTS object...")
             start_time = time.time()
             tts = TTS(model_path=model_dir, config_path=config_path).to(device)
             logger.info(f"TTS object instantiated and moved to {device} in {time.time() - start_time:.2f}s")
        else:
             logger.warning(f"Local model file not found at {model_path}, falling back to download/load by name...")
             tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        
        _TTS_MODEL = tts
        
        # Verify device placement
        try:
            param_device = next(tts.synthesizer.tts_model.parameters()).device
            logger.info(f"Model loaded successfully. Verifying device placement: {param_device}")
        except:
            logger.warning("Could not verify model device placement.")
            
        return _TTS_MODEL
        
    except Exception as e:
        logger.error(f"Failed to load TTS model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def generate_narration(text, reference_audio_path, output_path, language="en", temperature=0.75, repetition_penalty=2.0, top_k=50, top_p=0.85):
    """
    Generates audio narration using Coqui TTS (XTTS v2) with voice cloning.
    """
    ensure_dir(os.path.dirname(output_path))
    
    import threading
    
    # Deep Diagnostics
    pid = os.getpid()
    tid = threading.get_ident()
    logger.info(f"generate_narration called. PID: {pid}, TID: {tid}")
    
    # Check for GPU
    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1024**2
        vram_reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"CUDA Available. Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"Current VRAM Usage: {vram:.2f} MB (Allocated), {vram_reserved:.2f} MB (Reserved)")
    else:
        logger.warning("CUDA NOT AVAILABLE. Using CPU!")

    try:
        tts = load_tts_model()
        if not tts:
            logger.error("TTS model failed to load.")
            return None
        
        logger.info(f"Generating audio for text (length: {len(text)})... Params: T={temperature}, RP={repetition_penalty}, TK={top_k}, TP={top_p}")
        
        # Manual sentence splitting
        sentences = split_text(text)
        logger.info(f"Split text into {len(sentences)} sentences.")
        
        # Compute latents
        logger.info("Computing speaker latents...")
        start_latents = time.time()
            
        with torch.no_grad():
            gpt_cond_latent, speaker_embedding = tts.synthesizer.tts_model.get_conditioning_latents(audio_path=reference_audio_path)
            
        logger.info(f"Latents computed in {time.time() - start_latents:.2f}s")
        
        all_audio = []
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            logger.info(f"Generating sentence {i+1}/{len(sentences)}...")
            
            start_inf = time.time()
            
            with torch.no_grad():
                out = tts.synthesizer.tts_model.inference(
                    text=sentence,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    top_k=top_k,
                    top_p=top_p
                )
            
            logger.info(f"Inference took {time.time() - start_inf:.2f}s")

            wav = out['wav']
            if isinstance(wav, torch.Tensor):
                wav = wav.squeeze().cpu().numpy()
            else:
                wav = np.array(wav)
            
            all_audio.append(wav)
            
        # Convert to numpy and save
        if all_audio:
            final_wav = np.concatenate(all_audio)
        else:
            final_wav = np.array([])
        
        # XTTS usually outputs at 24000Hz
        try:
            out_sr = tts.synthesizer.output_sample_rate
        except:
            out_sr = 24000
            
        sf.write(output_path, final_wav, out_sr)
        
        logger.info(f"Audio generated at: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to generate audio: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
