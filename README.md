# Simple Storyteller

**Simple Storyteller** is an AI-powered application designed to generate short, narrated video stories and create voice datasets from YouTube videos. It leverages local Large Language Models (LLMs), Text-to-Speech (TTS) engines, and Image Generation models to create multimedia content entirely offline (after initial model downloads).

## Features

### 📖 Story Mode
Turn text prompts into fully narrated videos.
*   **Story Generation**: Uses **Ollama (Llama 3)** to generate engaging first-person life stories for characters.
*   **Narration**: Uses **Coqui TTS (XTTS v2)** for high-quality, emotive voice cloning and narration.
*   **Visuals**: Uses **Stable Diffusion XL (SDXL) Turbo** to generate cinematic images for each story segment.
*   **Video Assembly**: Automatically stitches audio and images into a final `.mp4` video.

### 🎙️ Voice Scraper
Build voice datasets for cloning.
*   **YouTube Download**: Downloads audio from YouTube videos.
*   **Segmentation**: Automatically splits audio into clean segments suitable for training or reference audio.

## Technology Stack

*   **Frontend**: [Gradio](https://www.gradio.app/)
*   **LLM Backend**: [Ollama](https://ollama.com/)
*   **TTS Engine**: [Coqui TTS](https://github.com/coqui-ai/TTS)
*   **Image Generation**: [Diffusers](https://huggingface.co/docs/diffusers/index) (SDXL Turbo)
*   **Audio/Video Processing**: `ffmpeg`, `moviepy`, `librosa`, `soundfile`

## Prerequisites

*   **OS**: Windows (tested), Linux, or macOS.
*   **Python**: 3.10 or higher.
*   **GPU**: NVIDIA GPU with at least 8GB VRAM is highly recommended for reasonable generation speeds.
*   **Ollama**: Must be installed and running. [Download Ollama](https://ollama.com/).
    *   Pull the Llama 3 model: `ollama pull llama3`
*   **FFmpeg**: The application looks for a local `bin/ffmpeg` folder, but having FFmpeg installed system-wide is recommended.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd MyStory
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You may need to install PyTorch with CUDA support separately if the default installation doesn't pick it up correctly. Visit [pytorch.org](https://pytorch.org/) for instructions.*

## Usage

1.  **Start the Ollama server** (if not already running).

2.  **Run the application**:
    ```bash
    python src/app.py
    ```

3.  **Open the Web UI**:
    The application will launch a local web server. Open the URL provided in the terminal (usually `http://127.0.0.1:7860`) in your browser.

### Generating a Story
1.  Go to the **Story Mode** tab.
2.  Enter a **Story Text** (or let the system generate one if that feature is enabled in your version).
3.  Upload a **Reference Voice** audio file (WAV/MP3) to clone the narrator's voice.
4.  Click **Generate Video**.
5.  The result will appear in the "Result" video player.

### Scraping Voices
1.  Go to the **Voice Scraper** tab.
2.  Paste a **YouTube URL**.
3.  Enter a **Character Name** (this will be the folder name).
4.  Click **Scrape & Segment**.
5.  Segments will be saved to `output/voices/<Character Name>`.

## Project Structure

```
MyStory/
├── models/             # Stores downloaded models (Ollama, HF, TTS)
├── output/             # Generated videos and voice clips
├── src/                # Source code
│   ├── app.py          # Main Gradio application
│   ├── audio_gen.py    # TTS logic
│   ├── visual_gen.py   # Image generation logic
│   ├── story_gen.py    # LLM story generation logic
│   └── ...
├── temp/               # Temporary processing files
└── requirements.txt    # Python dependencies
```

## License

[MIT License](LICENSE)
