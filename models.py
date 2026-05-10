# models.py

import torch
from transformers import Pipeline, pipeline, GenerationConfig
from diffusers import DiffusionPipeline, StableDiffusionInpaintPipelineLegacy
from PIL import Image
import markdown2


prompt = "How to set up a FastAPI project?"
system_prompt = """
Your name is FastAPI bot and you are a helpful
chatbot responsible for teaching FastAPI to your users.
Always respond in markdown.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def load_text_model():
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
        dtype=torch.bfloat16,
        device=device 
    )
    pipe.model.generation_config.max_length = None
    pipe.model.generation_config.max_new_tokens = None
    return pipe


def generate_text(pipe: Pipeline, prompt: str, temperature: float = 0.7) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ] 
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) 
    
    # Create a GenerationConfig object to avoid deprecation warnings
    # We explicitly set max_length to None to avoid conflicts with max_new_tokens
    generation_config = GenerationConfig(
        temperature=temperature,
        max_new_tokens=256,
        max_length=None,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )
    
    predictions = pipe(
        prompt,
        generation_config=generation_config,
        clean_up_tokenization_spaces=True,  # Changed to True to properly clean up tokens
    ) 
    generated_text = predictions[0]["generated_text"]
    
    # Extract assistant response
    assistant_start = generated_text.rfind("<|assistant|>")
    if assistant_start != -1:
        output = generated_text[assistant_start + len("<|assistant|>"):].strip()
    else:
        output = generated_text.strip()
    
    # Convert markdown to HTML for web display
    output = markdown2.markdown(output)
    
    return output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image_model() -> StableDiffusionInpaintPipelineLegacy:
    pipe = DiffusionPipeline.from_pretrained(
        "segmind/tiny-sd", torch_dtype=torch.float32,
        device=device
    ) 
    return pipe

def generate_image(
    pipe: StableDiffusionInpaintPipelineLegacy, prompt: str
) -> Image.Image:
    output = pipe(prompt, num_inference_steps=10).images[0]  
    return output