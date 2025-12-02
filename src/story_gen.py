import ollama
from .utils import setup_logging, split_text

logger = setup_logging("StoryGen")

def check_and_pull_model(model_name):
    """Checks if the model exists locally, and pulls it if not."""
    try:
        # List available models
        response = ollama.list()
        
        installed_models = []
        # Handle object-based response (newer ollama client)
        if hasattr(response, 'models'):
            for m in response.models:
                # Access 'model' attribute if it exists, else try 'name'
                if hasattr(m, 'model'):
                    installed_models.append(m.model)
                elif hasattr(m, 'name'):
                    installed_models.append(m.name)
        # Handle dict-based response (older versions)
        elif isinstance(response, dict) and 'models' in response:
            installed_models = [m.get('name', m.get('model')) for m in response['models']]
        else:
            # Fallback for unexpected structure
            logger.warning(f"Unexpected ollama.list() format: {type(response)}")
            installed_models = str(response)
            
        # Check if model is installed (ignoring tag if not specified, or exact match)
        is_installed = any(model_name in m for m in installed_models)
        
        if not is_installed:
            logger.info(f"Model '{model_name}' not found locally. Pulling it now (this may take a while)...")
            ollama.pull(model_name)
            logger.info(f"Model '{model_name}' pulled successfully.")
        else:
            logger.info(f"Model '{model_name}' is ready.")
            
    except Exception as e:
        logger.error(f"Failed to check/pull model '{model_name}': {e}")

def generate_story(character_name, model="llama3"):
    """
    Generates a life story for the character using Ollama.
    """
    check_and_pull_model(model)
    logger.info(f"Generating story for {character_name} using {model}...")
    
    prompt = f"""
    You are {character_name}. 
    Write your life story in the first person. 
    Focus on the key events that defined who you are. 
    The tone should match your personality. 
    Keep it engaging and suitable for a short video narration (approx 300-500 words).
    Do not include any introductory text like "Here is my story", just start telling the story.
    """

    try:
        response = ollama.chat(model=model, messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])
        story = response['message']['content']
        logger.info("Story generated successfully.")
        return story
    except Exception as e:
        logger.error(f"Failed to generate story: {e}")
        return f"I am {character_name}, and this is my story... (Generation Failed: {e})"

def split_story_into_segments(story, sentences_per_segment=2):
    """
    Splits the story into segments for visual generation.
    """
    sentences = split_text(story)
    segments = []
    current_segment = ""
    count = 0
    
    for sentence in sentences:
        current_segment += sentence + " "
        count += 1
        if count >= sentences_per_segment:
            segments.append(current_segment.strip())
            current_segment = ""
            count = 0
            
    if current_segment:
        segments.append(current_segment.strip())
        
    return segments

def generate_image_prompts(segments, character_name, model="llama3"):
    """
    Generates image prompts for each story segment using Ollama.
    """
    prompts = []
    logger.info("Generating image prompts for segments...")
    
    check_and_pull_model(model)
    
    for i, segment in enumerate(segments):
        logger.info(f"Generating prompt for segment {i+1}/{len(segments)}...")
        prompt_request = f"""
        You are an AI image prompt generator.
        Based on the following story segment narrated by {character_name}, create a detailed, visual image prompt for Stable Diffusion.
        Focus on the setting, action, and atmosphere.
        Keep it under 40 words.
        
        Story Segment: "{segment}"
        
        Output ONLY the prompt.
        """
        
        try:
            response = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt_request}])
            image_prompt = response['message']['content'].strip()
            # Clean up if the model is chatty
            if "Prompt:" in image_prompt:
                image_prompt = image_prompt.split("Prompt:")[-1].strip()
            prompts.append(image_prompt)
        except Exception as e:
            logger.error(f"Failed to generate prompt for segment: {e}")
            prompts.append(f"A cinematic shot of {character_name}, atmospheric lighting, high quality")
            
    return prompts
