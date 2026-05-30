# dependencies.py

from rag.constants import KNOWLEDGE_BASE_COLLECTION
from rag.service import vector_service
from rag.transform import clean, embed
from schemas import TextModelRequest, TextModelResponse


from fastapi import Body
from loguru import logger

from scraper import extract_urls, fetch_all

async def get_urls_content(body: TextModelRequest = Body(...)) -> str: 
    urls = extract_urls(body.prompt)
    if urls:
        try:
            urls_content = await fetch_all(urls)
            return urls_content
        except Exception as e:
            logger.warning(f"Failed to fetch one or several URls - Error: {e}")
    return ""

async def get_rag_content(body: TextModelRequest = Body(...)) -> str:
    rag_content = await vector_service.search(
        KNOWLEDGE_BASE_COLLECTION, embed(body.prompt), 3, 0.0
    )
    rag_content_str = "\n".join(
        [clean(c.payload["original_text"]) for c in rag_content]
    )

    if rag_content:
        scores = [round(c.score, 4) for c in rag_content]
        logger.debug(
            f"from RAG ({len(rag_content)} hits, scores={scores}): {rag_content_str}"
        )
    else:
        if await vector_service.db_client.collection_exists(
            KNOWLEDGE_BASE_COLLECTION
        ):
            count = await vector_service.db_client.count(
                collection_name=KNOWLEDGE_BASE_COLLECTION
            )
            logger.debug(
                f"from RAG: empty (collection has {count.count} points)"
            )
        else:
            logger.debug("from RAG: empty (knowledgebase collection does not exist)")

    return rag_content_str


def build_generation_prompt(
    user_prompt: str,
    urls_content: str = "",
    rag_content: str = "",
) -> str:
    context_parts = []

    if rag_content.strip():
        context_parts.append(rag_content.strip())

    if urls_content.strip():
        context_parts.append(urls_content.strip())

    if not context_parts:
        return user_prompt

    document = "\n\n".join(context_parts)
    return (
        f"Document:\n{document}\n\n"
        f"Question: {user_prompt}\n"
        "Answer using only the document above:"
    )