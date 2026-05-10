# main.py

from dotenv import load_dotenv
import os
load_dotenv()

import markdown2
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator
from openai import OpenAI
from fastapi import (
    Body,
    BackgroundTasks,
    FastAPI,
    File,
    UploadFile,
    status,
    HTTPException,
    Request,
    Depends,
)
from models import generate_text, load_text_model, load_image_model
from schemas import TextModelRequest, TextModelResponse
from utils import count_tokens
from dependencies import get_rag_content, get_urls_content
from upload import save_file
from rag import pdf_text_extractor, vector_service

models = {} 

@asynccontextmanager 
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    models["text"] = load_text_model() 
    models["text2image"] = load_image_model() 

    yield 

    ... # Run cleanup code here

    models.clear() 

app = FastAPI(lifespan=lifespan) 
# app = FastAPI() 
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY')) 


@app.get("/")
def root_controller():
    return {"status": "healthy"}

@app.get("/generate/text") 
def serve_language_model_controller(prompt: str) -> str: 
    # pipe = load_text_model() 
    output = generate_text(models["text"], prompt) 
    html_output = markdown2.markdown(output, extras=["fenced-code-blocks", "code-colors"])
    return html_output 

@app.post("/generate/text") 
async def serve_text_to_text_controller(
    request: Request, 
    body: TextModelRequest = Body(...),
    urls_content: str = Depends(get_urls_content),
    rag_content: str = Depends(get_rag_content),
) -> TextModelResponse:
    if body.model not in ["tinyLlama", "gemma2b"]: 
        raise HTTPException(
            detail=f"Model {body.model} is not supported",
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    
    # output=generate_text(models['text'], body.prompt, body.temperature)
    prompt = body.prompt + " " + urls_content + rag_content
    output = generate_text(models["text"], prompt, body.temperature)
    html_output = markdown2.markdown(output, extras=["fenced-code-blocks", "code-colors"])
    tokens = count_tokens(body.prompt) + count_tokens(output)
    return TextModelResponse(content=html_output, ip=request.client.host, tokens=tokens)
    # output = generate_text(models["text"], body.prompt, body.temperature)    
    
    # return TextModelResponse(response=output,tokens=tokens)


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
    html_statement = markdown2.markdown(statement, extras=["fenced-code-blocks", "code-colors"])
    return {"statement": html_statement} 

@app.post("/upload")
async def file_upload_controller(
    file: Annotated[UploadFile, File(description="A file read as UploadFile")],
    bg_text_processor: BackgroundTasks,
):
    if file.content_type != "application/pdf":
        raise HTTPException(
            detail=f"Only uploading PDF documents are supported",
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    try:
        filepath = await save_file(file)
        bg_text_processor.add_task(pdf_text_extractor, filepath)
        bg_text_processor.add_task(
            vector_service.store_file_content_in_db,
            filepath.replace("pdf", "txt"), 
            512, 
            "knowledgebase", 
            768
        )
    except Exception as e:
        raise HTTPException(
            detail=f"An error occurred while saving file - Error: {e}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    return {"filename": file.filename, "message": "File uploaded successfully"}