import gradio as gr
import os
import sys
import time
import shutil
import multiprocessing
import queue

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Set environment variables
os.environ["OLLAMA_MODELS"] = os.path.join(project_root, "models", "ollama")
os.environ["OLLAMA_HOST"] = "127.0.0.1:11435" # Custom port
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
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

# --- Persistent Workers ---

def audio_worker(task_queue, result_queue):
    """
    Persistent worker for Audio Generation.
    Loads model ONCE, then waits for tasks.
    """
    print("[AudioWorker] Starting process...")
    try:
        from src.audio_gen import generate_narration
        print("[AudioWorker] Ready for tasks.")
        
        while True:
            try:
                task = task_queue.get(timeout=1)
                if task == "STOP":
                    break
                
                # Unwrap task
                text, ref_audio, audio_path, temp, rep_pen, top_k, top_p = task
                print(f"[AudioWorker] Processing: {text[:20]}... params: T={temp}, RP={rep_pen}")
                
                res = generate_narration(
                    text, ref_audio, audio_path, 
                    temperature=float(temp), 
                    repetition_penalty=float(rep_pen), 
                    top_k=int(top_k), 
                    top_p=float(top_p)
                )
                result_queue.put(("success", res))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[AudioWorker] Error: {e}")
                result_queue.put(("error", str(e)))
                
    except Exception as e:
        print(f"[AudioWorker] Critical Fail: {e}")
        # Signal death
        result_queue.put(("CRITICAL", str(e)))

def visual_worker(task_queue, result_queue):
    """
    Persistent worker for Visual Generation.
    Loads SDXL model ONCE, then waits for tasks.
    """
    print("[VisualWorker] Starting process...")
    try:
        from src.story_gen import split_story_into_segments, generate_image_prompts
        from src.visual_gen import generate_image, create_video, load_pipeline
        
        # Preload Model
        print("[VisualWorker] Loading SDXL Model...")
        pipe = load_pipeline() # This caches the pipe in visual_gen's global _pipe
        print("[VisualWorker] Model Loaded. Ready.")
        
        while True:
            try:
                task = task_queue.get(timeout=1)
                if task == "STOP":
                    break
                
                # Unwrap task
                text, audio_path, output_dir = task
                print("[VisualWorker] Generating Visuals...")
                
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
                    continue

                # 3. Video
                final_video_path = os.path.join(output_dir, "final_story.mp4")
                video_path = create_video(audio_path, image_paths, final_video_path)
                
                if video_path:
                     result_queue.put(("success", video_path))
                else:
                     result_queue.put(("error", "Video assembly failed"))
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VisualWorker] Error: {e}")
                result_queue.put(("error", str(e)))

    except Exception as e:
        print(f"[VisualWorker] Critical Fail: {e}")
        result_queue.put(("CRITICAL", str(e)))

def scraper_worker(task_queue, result_queue):
    """
    Persistent worker for Voice Scraping.
    Downloads and segments audio in a background process.
    """
    print("[ScraperWorker] Starting process...")
    try:
        from src.data_scouring import download_audio, segment_audio
        print("[ScraperWorker] Ready for tasks.")
        
        while True:
            try:
                task = task_queue.get(timeout=1)
                if task == "STOP":
                    break
                
                # Unwrap task
                url, char_name, silence_thresh, min_silence_len, remove_noise = task
                print(f"[ScraperWorker] Processing: {char_name} ({url})")
                
                temp_dir = os.path.join("temp", "scraper", char_name)
                output_dir = os.path.join("output", "voices", char_name)
                clean_temp(temp_dir)
                ensure_dir(output_dir)
                
                wav_path = download_audio(url, temp_dir)
                if not wav_path:
                    result_queue.put(("error", "Download failed"))
                    continue
                    
                segments = segment_audio(
                    wav_path, 
                    output_dir, 
                    min_silence_len=int(min_silence_len), 
                    silence_thresh=int(silence_thresh),
                    remove_noise=remove_noise
                )
                
                result_queue.put(("success", f"Success! Saved {len(segments)} segments to {output_dir}"))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ScraperWorker] Error: {e}")
                result_queue.put(("error", str(e)))

    except Exception as e:
        print(f"[ScraperWorker] Critical Fail: {e}")
        result_queue.put(("CRITICAL", str(e)))


# --- Global State ---
g_audio_task = None
g_audio_result = None
g_visual_task = None
g_visual_result = None
g_scraper_task = None
g_scraper_result = None

# --- Main Logic ---

def process_story(text, ref_audio, skip_visuals, temp, rep_pen, top_k, top_p, progress=gr.Progress()):
    if not text or not ref_audio:
        return "Error: Input missing", None

    progress(0, desc="Queuing Task...")
    
    run_id = f"run_{int(time.time())}"
    temp_dir = os.path.join("temp", run_id)
    output_dir = os.path.join("output", run_id)
    clean_temp(temp_dir)
    ensure_dir(output_dir)

    # 1. Audio
    audio_path = os.path.join(temp_dir, "narration.wav")
    
    # Send task
    progress(0.1, desc="Generating Audio...")
    g_audio_task.put((text, ref_audio, audio_path, temp, rep_pen, top_k, top_p))
    
    # Wait for result
    try:
        status, result = g_audio_result.get(timeout=300) # 5 min timeout
        if status != "success":
            return f"Audio Error: {result}", None
        generated_audio = result
    except queue.Empty:
        return "Error: Audio Worker Timed Out", None

    if skip_visuals:
        # Copy audio to output to make it accessible
        final_audio_path = os.path.join(output_dir, "narrated_story.wav")
        shutil.copy(generated_audio, final_audio_path)
        progress(1.0, desc="Done (Audio Only)")
        return "Success! Audio generated.", final_audio_path

    # 2. Visuals
    progress(0.4, desc="Generating Video...")
    g_visual_task.put((text, generated_audio, output_dir))
    
    try:
        status, result = g_visual_result.get(timeout=600) # 10 min timeout
        if status != "success":
            return f"Visual Error: {result}", None
        video_path = result
    except queue.Empty:
        return "Error: Visual Worker Timed Out", None
    
    progress(1.0, desc="Done!")
    return "Success! Video generated.", video_path

def process_scraper(url, char_name, silence_thresh, min_silence_len, remove_noise, progress=gr.Progress()):
    if not url or not char_name:
        return "Error: Missing URL or Name"
    
    progress(0, desc="Queuing Scraper Task...")
    g_scraper_task.put((url, char_name, silence_thresh, min_silence_len, remove_noise))
    
    progress(0.1, desc="Processing (Download/Clean/Segment)...")
    try:
        # Give it plenty of time (download + noise removal can be slow)
        status, result = g_scraper_result.get(timeout=1200) # 20 min timeout
        if status != "success":
            return f"Error: {result}"
        return result
    except queue.Empty:
        return "Error: Scraper Worker Timed Out"

# --- UI Setup ---

# Initialize Queues
g_audio_task = multiprocessing.Queue()
g_audio_result = multiprocessing.Queue()
g_visual_task = multiprocessing.Queue()
g_visual_result = multiprocessing.Queue()
g_scraper_task = multiprocessing.Queue()
g_scraper_result = multiprocessing.Queue()

def start_workers():
    p_audio = multiprocessing.Process(target=audio_worker, args=(g_audio_task, g_audio_result))
    p_visual = multiprocessing.Process(target=visual_worker, args=(g_visual_task, g_visual_result))
    p_scraper = multiprocessing.Process(target=scraper_worker, args=(g_scraper_task, g_scraper_result))
    
    p_audio.daemon = True
    p_visual.daemon = True
    p_scraper.daemon = True
    
    p_audio.start()
    p_visual.start()
    p_scraper.start()
    
    return p_audio, p_visual, p_scraper

with gr.Blocks(title="MyStory AI") as demo:
    gr.Markdown("# 🎬 MyStory AI")
    
    with gr.Tabs():
        with gr.Tab("Story Mode"):
            with gr.Row():
                with gr.Column():
                    story_input = gr.Textbox(label="Story Text", lines=5, placeholder="Enter text...")
                    audio_input = gr.Audio(label="Reference Voice", type="filepath")
                    
                    with gr.Accordion("Advanced Voice Settings", open=False):
                        temp_slider = gr.Slider(minimum=0.01, maximum=1.0, value=0.75, label="Temperature", info="Lower = More stable/similar to ref. Higher = More expressive/random.")
                        rep_pen_slider = gr.Slider(minimum=1.0, maximum=10.0, value=2.0, label="Repetition Penalty", info="Increase if speech repeats or stutters.")
                        top_p_slider = gr.Slider(minimum=0.01, maximum=1.0, value=0.85, label="Top P", info="Nucleus sampling probability.")
                        top_k_slider = gr.Slider(minimum=1, maximum=100, step=1, value=50, label="Top K", info="Top-K sampling.")
                        
                    skip_vis = gr.Checkbox(label="Generate Audio Only (Skip Video)", value=False)
                    generate_btn = gr.Button("Generate", variant="primary")
                
                with gr.Column():
                    status_output = gr.Textbox(label="Status")
                    # Use File output because it can show Video OR Audio players automatically
                    final_output = gr.File(label="Result")
            
            generate_btn.click(
                fn=process_story,
                inputs=[story_input, audio_input, skip_vis, temp_slider, rep_pen_slider, top_k_slider, top_p_slider],
                outputs=[status_output, final_output]
            )
            
        with gr.Tab("Voice Scraper"):
            gr.Markdown("## YouTube Audio Scraper")
            with gr.Row():
                url_input = gr.Textbox(label="YouTube URL")
                name_input = gr.Textbox(label="Character Name")
            
            with gr.Accordion("Advanced Tuning", open=True):
                thresh_slider = gr.Slider(
                    minimum=-60, maximum=-10, value=-40, step=1, 
                    label="Silence Threshold (dB)", 
                    info="Lower = Checks for quieter silence. Higher = Less sensitive."
                )
                min_len_slider = gr.Slider(
                    minimum=100, maximum=3000, value=500, step=100, 
                    label="Min Silence Length (ms)",
                    info="Minimum duration of silence to trigger a split."
                )
                noise_checkbox = gr.Checkbox(
                    label="Remove Background Noise",
                    value=False,
                    info="Cleans audio using spectral gating before splitting (may be slow)."
                )
            
            scrape_btn = gr.Button("Scrape & Segment")
            scraper_status = gr.Textbox(label="Status")
            
            scrape_btn.click(
                fn=process_scraper,
                inputs=[url_input, name_input, thresh_slider, min_len_slider, noise_checkbox],
                outputs=[scraper_status]
            )

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    if not start_ollama():
        logger.warning("Ollama failed to start!")
        
    # Start Persistent Workers
    logger.info("Starting Background Workers...")
    p_a, p_v, p_s = start_workers()
    logger.info("Workers Started. UI Launching...")
    
    try:
        demo.queue().launch(inbrowser=True)
    finally:
        # Cleanup on exit
        g_audio_task.put("STOP")
        g_visual_task.put("STOP")
        g_scraper_task.put("STOP")
        p_a.terminate()
        p_v.terminate()
        p_s.terminate()
