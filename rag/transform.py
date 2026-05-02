# rag/transform.py

import re
from typing import Any, AsyncGenerator

import aiofiles
from sentence_transformers import SentenceTransformer

DEFAULT_CHUNK_SIZE = 1024 * 1024 * 50  # 50 megabytes

embedder = SentenceTransformer("jinaai/jina-embeddings-v2-base-en")

async def load(filepath: str, chunk_size: int) -> AsyncGenerator[str, Any]:
    async with aiofiles.open(filepath, "r", encoding="utf-8") as f: 
        while chunk := await f.read(chunk_size): 
            yield chunk 

def clean(text: str) -> str:
    t = text.replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\. ,", "", t)
    t = t.replace("..", ".")
    t = t.replace(". .", ".")
    cleaned_text = t.replace("\n", " ").strip()
    return cleaned_text 

def embed(text: str) -> list[float]:
    return embedder.encode(text).tolist()