import gradio as gr
import os
import sys
import time

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Set environment variables
os.environ["OLLAMA_MODELS"] = os.path.join(project_root, "models", "ollama")
os.environ["HF_HOME"] = os.path.join(project_root, "models", "huggingface")
os.environ["TTS_HOME"] = os.path.join(project_root, "models", "tts")

# Add local bin to PATH
local_bin = os.path.join(project_root, "bin")
os.environ["PATH"] = os.path.join(local_bin, "ffmpeg", "bin") + os.pathsep + \
                     os.path.join(local_bin, "ollama") + os.pathsep + \
                     os.environ["PATH"]

from src.utils import setup_logging, clean_temp, get_ffmpeg_path, ensure_dir
from src.audio_gen import generate_narration, load_tts_model
from src.visual_gen import generate_image, create_video
from src.story_gen import split_story_into_segments, generate_image_prompts

# Set ffmpeg path for moviepy
ffmpeg_path = get_ffmpeg_path()
if ffmpeg_path:
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path

logger = setup_logging("App")

def process_story(text, ref_audio):
    if not text or not ref_audio:
        return "Error: Please provide both text and reference audio.", None

    logger.info("Starting processing...")
    run_id = f"run_{int(time.time())}"
    temp_dir = os.path.join("temp", run_id)
    output_dir = os.path.join("output", run_id)
    clean_temp(temp_dir)
    clean_temp(output_dir)
    ensure_dir(output_dir)

    # 1. Generate Audio
    logger.info("Generating audio...")
    audio_path = os.path.join(temp_dir, "narration.wav")
    # No progress updates to avoid GPU blocking
    generated_audio = generate_narration(text, ref_audio, audio_path)
    
    if not generated_audio:
        return "Error: Audio generation failed.", None

    # 2. Generate Visuals
    logger.info("Generating visuals...")
    segments = split_story_into_segments(text)
    # Use a generic character name for now since we removed the input
    prompts = generate_image_prompts(segments, "cinematic scene")
    
    image_paths = []
    for i, prompt in enumerate(prompts):
        img_path = os.path.join(temp_dir, "images", f"img_{i}.png")
        saved_path = generate_image(prompt, img_path)
        if saved_path:
            image_paths.append(saved_path)
            
    if not image_paths:
        return "Error: Image generation failed.", None

    # 3. Assemble Video
    logger.info("Assembling video...")
    final_video_path = os.path.join(output_dir, "final_story.mp4")
    video_path = create_video(generated_audio, image_paths, final_video_path)
    
    if video_path:
        return "Success! Video generated.", video_path
    else:
        return "Error: Video assembly failed.", None

from src.data_scouring import download_audio, segment_audio

# ... (process_story function remains)

def process_scraper(url, char_name):
    if not url or not char_name:
        return "Error: Please provide both URL and Character Name."
        
    logger.info(f"Starting scraper for {char_name}...")
    temp_dir = os.path.join("temp", "scraper", char_name)
    output_dir = os.path.join("output", "voices", char_name)
    clean_temp(temp_dir)
    ensure_dir(output_dir)
    
    # 1. Download
    wav_path = download_audio(url, temp_dir)
    if not wav_path:
        return "Error: Download failed."
        
    # 2. Segment
    segments = segment_audio(wav_path, output_dir)
    
    if segments:
        return f"Success! Saved {len(segments)} voice clips to {output_dir}"
    else:
        return "Error: No segments found (audio might be too quiet or short)."

# Minimal UI
with gr.Blocks(title="Simple Storyteller") as demo:
    gr.Markdown("# ⚡ Fast Storyteller")
    
    with gr.Tabs():
        with gr.Tab("Story Mode"):
            with gr.Row():
                with gr.Column():
                    story_input = gr.Textbox(label="Story Text", lines=5, placeholder="Paste your story here...")
                    audio_input = gr.Audio(label="Reference Voice", type="filepath")
                    generate_btn = gr.Button("Generate Video", variant="primary")
                
                with gr.Column():
                    status_output = gr.Textbox(label="Status")
                    video_output = gr.Video(label="Result")
                    
            generate_btn.click(
                fn=process_story,
                inputs=[story_input, audio_input],
                outputs=[status_output, video_output]
            )
            
        with gr.Tab("Voice Scraper"):
            gr.Markdown("Download and split voice clips from YouTube.")
            with gr.Row():
                url_input = gr.Textbox(label="YouTube URL")
                name_input = gr.Textbox(label="Character Name (for folder)")
                scrape_btn = gr.Button("Scrape & Segment")
            
            scraper_status = gr.Textbox(label="Status")
            
            scrape_btn.click(
                fn=process_scraper,
                inputs=[url_input, name_input],
                outputs=[scraper_status]
            )

if __name__ == "__main__":
    logger.info("Preloading TTS model...")
    load_tts_model()
    logger.info("Model loaded. Launching UI...")
    
    # Launch without queue to prevent async interference
    demo.launch(inbrowser=True)
