# rag/transform.py

import re
from contextvars import ContextVar
from typing import Any, AsyncGenerator

import aiofiles
from sentence_transformers import SentenceTransformer

DEFAULT_CHUNK_SIZE = 1024 * 1024 * 50  # 50 megabytes

_QUERY_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "shall", "can",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their", "about",
    "how", "when", "where", "why", "there", "here", "than", "then",
})

_embedding_query_context: ContextVar[str | None] = ContextVar(
    "embedding_query_text", default=None
)

_embedder = None

def _get_embedder() -> SentenceTransformer:
    """Lazy initialization of the SentenceTransformer model."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("jinaai/jina-embeddings-v2-base-en")
    return _embedder

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

def get_embedding_query_text() -> str | None:
    return _embedding_query_context.get()


def extract_query_terms(text: str) -> list[str]:
    """Significant terms from a user query for lexical RAG matching."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [
        word
        for word in words
        if len(word) >= 3 and word not in _QUERY_STOPWORDS
    ]


def lexical_overlap_score(terms: list[str], text: str) -> float:
    if not terms:
        return 0.0
    lowered = text.lower()
    return sum(1 for term in terms if term in lowered) / len(terms)


def embed(text: str) -> list[float]:
    _embedding_query_context.set(text)
    return _get_embedder().encode(text).tolist()

def chunk(tokens: list, chunk_size: int) -> list[list]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    return [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
