import gradio as gr
import os
import sys
import time
import shutil
import multiprocessing

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Set environment variables
os.environ["OLLAMA_MODELS"] = os.path.join(project_root, "models", "ollama")
os.environ["OLLAMA_HOST"] = "127.0.0.1:11435" # Use custom port to avoid conflicts
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" # suppress windows symlink warning
os.environ["HF_HOME"] = os.path.join(project_root, "models", "huggingface")
os.environ["TTS_HOME"] = os.path.join(project_root, "models", "tts")

# Add local bin to PATH
local_bin = os.path.join(project_root, "bin")
os.environ["PATH"] = os.path.join(local_bin, "ffmpeg", "bin") + os.pathsep + \
                     os.path.join(local_bin, "ollama") + os.pathsep + \
                     os.environ["PATH"]

from src.utils import setup_logging, clean_temp, get_ffmpeg_path, ensure_dir, start_ollama

# Set ffmpeg path for moviepy
ffmpeg_path = get_ffmpeg_path()
if ffmpeg_path:
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path

logger = setup_logging("App")

# --- Multiprocessing Workers ---

def _run_audio_generation(text, ref_audio, audio_path, result_queue):
    """Run TTS in a separate process to isolate CUDA context."""
    try:
        from src.audio_gen import generate_narration
        result = generate_narration(text, ref_audio, audio_path)
        result_queue.put(("success", result))
    except Exception as e:
        result_queue.put(("error", str(e)))

def _run_visual_generation(text, audio_path, output_dir, result_queue):
    """Run Visuals + Video Assembly in a separate process."""
    try:
        from src.story_gen import split_story_into_segments, generate_image_prompts
        from src.visual_gen import generate_image, create_video
        
        # 1. Prompts
        segments = split_story_into_segments(text)
        prompts = generate_image_prompts(segments, "cinematic scene")
        
        # 2. Images
        temp_dir = os.path.join(output_dir, "temp_images")
        ensure_dir(temp_dir)
        
        image_paths = []
        for i, prompt in enumerate(prompts):
            img_path = os.path.join(temp_dir, f"img_{i}.png")
            saved = generate_image(prompt, img_path)
            if saved:
                image_paths.append(saved)
        
        if not image_paths:
            result_queue.put(("error", "No images generated"))
            return

        # 3. Video
        final_video_path = os.path.join(output_dir, "final_story.mp4")
        video_path = create_video(audio_path, image_paths, final_video_path)
        
        if video_path:
             result_queue.put(("success", video_path))
        else:
             result_queue.put(("error", "Video assembly failed"))
            
    except Exception as e:
        result_queue.put(("error", str(e)))

# --- Main Processing Logic ---

def process_story(text, ref_audio, progress=gr.Progress()):
    if not text or not ref_audio:
        return "Error: Please provide both text and reference audio.", None

    logger.info("Starting processing...")
    progress(0, desc="Initializing...")
    
    run_id = f"run_{int(time.time())}"
    temp_dir = os.path.join("temp", run_id)
    output_dir = os.path.join("output", run_id)
    clean_temp(temp_dir)
    ensure_dir(output_dir)

    # 1. Generate Audio
    logger.info("Generating audio (in subprocess)...")
    progress(0.1, desc="Generating Audio...")
    audio_path = os.path.join(temp_dir, "narration.wav")
    
    queue_audio = multiprocessing.Queue()
    p_audio = multiprocessing.Process(
        target=_run_audio_generation,
        args=(text, ref_audio, audio_path, queue_audio)
    )
    p_audio.start()
    
    # Wait for completion while keeping UI responsive (Gradio yields)
    while p_audio.is_alive():
        time.sleep(0.5)
        # progress(0.2, desc="Generating Audio (Running)...") 
    
    p_audio.join() # Ensure cleanup
    
    if not queue_audio.empty():
        status, result = queue_audio.get()
        if status == "error":
            return f"Error: Audio generation failed - {result}", None
        generated_audio = result
    else:
        return "Error: Audio process died unexpectedly.", None

    if not generated_audio:
        return "Error: Audio generation returned None.", None

    # 2. Visuals & Video
    logger.info("Generating visuals & video (in subprocess)...")
    progress(0.4, desc="Generating Visuals & Video...")
    
    queue_video = multiprocessing.Queue()
    p_video = multiprocessing.Process(
        target=_run_visual_generation,
        args=(text, generated_audio, output_dir, queue_video)
    )
    p_video.start()
    
    while p_video.is_alive():
        time.sleep(0.5)
        # progress(0.6, desc="Working on Visuals...")
        
    p_video.join()

    if not queue_video.empty():
        status, result = queue_video.get()
        if status == "error":
            return f"Error: Video generation failed - {result}", None
        video_path = result
    else:
        return "Error: Video process died unexpectedly.", None
    
    progress(1.0, desc="Done!")
    return "Success! Video generated.", video_path

def process_scraper(url, char_name):
    if not url or not char_name:
        return "Error: Please provide both URL and Character Name."
    
    # Simple synchronous call is fine here as it's mostly network/CPU IO
    from src.data_scouring import download_audio, segment_audio
    
    try:
        temp_dir = os.path.join("temp", "scraper", char_name)
        output_dir = os.path.join("output", "voices", char_name)
        clean_temp(temp_dir)
        ensure_dir(output_dir)
        
        wav_path = download_audio(url, temp_dir)
        if not wav_path:
            return "Error: Download failed."
            
        segments = segment_audio(wav_path, output_dir)
        return f"Success! Saved {len(segments)} segments to {output_dir}"
    except Exception as e:
        return f"Error: {e}"

# --- UI Setup ---

with gr.Blocks(title="MyStory AI") as demo:
    gr.Markdown("# 🎬 MyStory AI")
    
    with gr.Tabs():
        with gr.Tab("Story Mode"):
            with gr.Row():
                with gr.Column():
                    story_input = gr.Textbox(label="Story Text", lines=5, placeholder="Enter the story text here...")
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
            with gr.Row():
                url_input = gr.Textbox(label="YouTube URL")
                name_input = gr.Textbox(label="Character Name")
                scrape_btn = gr.Button("Scrape & Segment")
            
            scraper_status = gr.Textbox(label="Status")
            
            scrape_btn.click(
                fn=process_scraper,
                inputs=[url_input, name_input],
                outputs=[scraper_status]
            )

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    # Ensure Ollama is running
    if not start_ollama():
        logger.warning("Ollama failed to start!")
        
    logger.info("Launching Gradio UI...")
    demo.queue().launch(inbrowser=True)
