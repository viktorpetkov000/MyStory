import os
import torch
# Ensure HF_HOME is set before importing diffusers
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.join(project_root, "models", "huggingface")

from diffusers import AutoPipelineForText2Image
try:
    from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
except ImportError:
    # Fallback for MoviePy v2.0+
    from moviepy import ImageClip, AudioFileClip, concatenate_videoclips
from .utils import setup_logging, ensure_dir

logger = setup_logging("VisualGen")

def load_pipeline():
    """Loads the Stable Diffusion pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Stable Diffusion pipeline on {device}...")
    
    try:
        # Using SDXL Turbo for fast generation
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", 
            torch_dtype=torch.float16, 
            variant="fp16"
        )
        pipe.to(device)
        return pipe
    except Exception as e:
        logger.error(f"Failed to load SD pipeline: {e}")
        return None

# Global pipeline to avoid reloading
_pipe = None

def generate_image(prompt, output_path):
    """Generates an image for the given prompt."""
    global _pipe
    if _pipe is None:
        _pipe = load_pipeline()
        if _pipe is None:
            return None
            
    ensure_dir(os.path.dirname(output_path))
    logger.info(f"Generating image for prompt: {prompt}")
    
    try:
        # SDXL Turbo needs only 1-4 steps
        image = _pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0.0).images[0]
        image.save(output_path)
        logger.info(f"Image saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to generate image: {e}")
        return None

def create_video(audio_path, image_paths, output_path):
    """
    Assembles images and audio into a video.
    Images are displayed for equal duration to match audio length.
    """
    logger.info("Assembling video...")
    
    try:
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        
        if not image_paths:
            logger.error("No images provided for video.")
            return None
            
        # Simple logic: divide duration equally among images
        image_duration = duration / len(image_paths)
        
        clips = []
        for img_path in image_paths:
            # MoviePy v2 uses with_duration instead of set_duration
            try:
                clip = ImageClip(img_path).with_duration(image_duration)
            except AttributeError:
                # Fallback for older versions
                clip = ImageClip(img_path).set_duration(image_duration)
            
            # Resize to something standard if needed, e.g. 1024x1024 or 1920x1080
            # clip = clip.resize(height=1080) 
            clips.append(clip)
            
        video = concatenate_videoclips(clips, method="compose")
        
        # MoviePy v2 uses with_audio instead of set_audio
        try:
            video = video.with_audio(audio_clip)
        except AttributeError:
            video = video.set_audio(audio_clip)
        
        video.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac')
        logger.info(f"Video saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to create video: {e}")
        return None
