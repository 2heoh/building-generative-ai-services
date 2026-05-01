# main.py

from dotenv import load_dotenv
import os

load_dotenv()

from fastapi import FastAPI
from openai import OpenAI
from models import load_text_model, generate_text

app = FastAPI() 
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY')) 

@app.get("/")
def root_controller():
    return {"status": "healthy"}

@app.get("/generate/text") 
def serve_language_model_controller(prompt: str) -> str: 
    pipe = load_text_model() 
    output = generate_text(pipe, prompt) 
    return output 

@app.get("/chat") 
def chat_controller(prompt: str = "Inspire me"): 
    response = openai_client.chat.completions.create( 
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    statement = response.choices[0].message.content
    return {"statement": statement} 
