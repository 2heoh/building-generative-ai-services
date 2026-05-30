# rag/service.py

import os

from loguru import logger
from qdrant_client.http.models import ScoredPoint

from .constants import (
    DEFAULT_CHUNK_BYTES,
    KNOWLEDGE_BASE_COLLECTION,
    KNOWLEDGE_BASE_VECTOR_SIZE,
)
from .repository import VectorRepository
from .transform import (
    clean,
    embed,
    extract_query_terms,
    get_embedding_query_text,
    lexical_overlap_score,
    load,
)

_VECTOR_WEIGHT = 0.4
_LEXICAL_WEIGHT = 0.6
_VECTOR_CANDIDATE_LIMIT = 100


class VectorService(VectorRepository):
    def __init__(self):
        super().__init__()

    async def store_file_content_in_db(
        self,
        filepath: str,
        chunk_size: int = DEFAULT_CHUNK_BYTES,
        collection_name: str = KNOWLEDGE_BASE_COLLECTION,
        collection_size: int = KNOWLEDGE_BASE_VECTOR_SIZE,
    ) -> None:
        await self.ensure_collection(collection_name, collection_size)
        logger.debug(f"Inserting {filepath} content into database")
        async for chunk in load(filepath, chunk_size):
            logger.debug(f"Inserting '{chunk[0:20]}...' into database")

            embedding_vector = embed(clean(chunk))
            filename = os.path.basename(filepath)
            await self.create(
                collection_name, embedding_vector, chunk, filename
            )

    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        retrieval_limit: int,
        score_threshold: float,
    ) -> list[ScoredPoint]:
        if not await self.db_client.collection_exists(collection_name):
            logger.debug(
                f"Collection {collection_name} does not exist, returning no results"
            )
            return []

        query_text = get_embedding_query_text()
        terms = extract_query_terms(query_text) if query_text else []

        vector_limit = max(_VECTOR_CANDIDATE_LIMIT, retrieval_limit * 25)
        vector_results = await super().search(
            collection_name,
            query_vector,
            vector_limit,
            score_threshold,
        )

        if not terms:
            return vector_results[:retrieval_limit]

        candidates: dict[int | str, ScoredPoint] = {
            point.id: point for point in vector_results
        }

        offset = None
        while True:
            points, offset = await self.db_client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                text = point.payload.get("original_text", "")
                if lexical_overlap_score(terms, text) <= 0:
                    continue
                if point.id not in candidates:
                    candidates[point.id] = ScoredPoint(
                        id=point.id,
                        version=0,
                        score=0.0,
                        payload=point.payload,
                        vector=None,
                    )

            if offset is None:
                break

        ranked: list[ScoredPoint] = []
        for point in candidates.values():
            text = point.payload.get("original_text", "")
            lexical = lexical_overlap_score(terms, text)
            hybrid = _VECTOR_WEIGHT * point.score + _LEXICAL_WEIGHT * lexical
            ranked.append(point.model_copy(update={"score": hybrid}))

        ranked.sort(key=lambda p: p.score, reverse=True)
        return ranked[:retrieval_limit]


vector_service = VectorService()