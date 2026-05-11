"""
Integration tests for RAG (Retrieval-Augmented Generation) data retrieval.

These tests verify the complete RAG retrieval pipeline:
1. Storing documents in the vector database
2. Retrieving relevant content based on semantic queries
3. Validating retrieval quality and relevance
"""

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from rag.service import vector_service
from rag.transform import embed, clean


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="function")
async def rag_collection():
    """Create a temporary collection for RAG retrieval testing."""
    collection_name = "rag_test_collection"
    
    # Ensure clean state - delete if exists
    client = AsyncQdrantClient(host="localhost", port=6333)
    if await client.collection_exists(collection_name=collection_name):
        await client.delete_collection(collection_name=collection_name)
    
    # Create collection with 768-dim vectors (matches jina-embeddings-v2-base-en)
    await client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    
    yield collection_name
    
    # Cleanup
    if await client.collection_exists(collection_name=collection_name):
        await client.delete_collection(collection_name=collection_name)
    await client.close()


@pytest.fixture(scope="function")
async def populated_rag_collection(rag_collection):
    """Create a collection pre-populated with known test documents."""
    documents = [
        (
            "Python is a high-level, interpreted programming language known for "
            "its readability and versatility. It supports multiple programming "
            "paradigms including object-oriented, imperative, and functional."
        ),
        (
            "Machine learning is a subset of artificial intelligence that focuses "
            "on building systems that learn from data. Common approaches include "
            "supervised learning, unsupervised learning, and reinforcement learning."
        ),
        (
            "FastAPI is a modern, fast web framework for building APIs with Python "
            "3.7+ based on standard Python type hints. It provides automatic "
            "OpenAPI documentation and async support."
        ),
        (
            "Vector databases like Qdrant are designed to store and query high-dimensional "
            "vector data. They enable semantic search by finding vectors similar to a "
            "given query vector using metrics like cosine similarity."
        ),
        (
            "Natural language processing (NLP) is a field of AI that focuses on the "
            "interaction between computers and human language. Tasks include text "
            "classification, named entity recognition, and sentiment analysis."
        ),
    ]
    
    for i, doc_text in enumerate(documents):
        embedding = embed(clean(doc_text))
        await vector_service.create(
            collection_name=rag_collection,
            embedding_vector=embedding,
            original_text=doc_text,
            source="test_integration",
        )
    
    yield rag_collection


# ============================================================================
# Tests - Basic Retrieval
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieve_single_document(populated_rag_collection):
    """Test retrieving a single document with valid structure."""
    query = "What is Python programming language?"
    query_vector = embed(clean(query))
    
    # Retrieve results
    results = await vector_service.search(
        collection_name=populated_rag_collection,
        query_vector=query_vector,
        retrieval_limit=5,
        score_threshold=0.0,
    )
    
    assert len(results) >= 1
    # All returned results should contain programming-related content
    retrieved_texts = [r.payload["original_text"] for r in results]
    assert any(
        "programming" in text.lower() or "Python" in text or "language" in text.lower()
        for text in retrieved_texts
    ), f"Expected programming-related content, got: {retrieved_texts}"
    
    # Test limit=1 returns exactly one result with valid structure
    single_result = await vector_service.search(
        collection_name=populated_rag_collection,
        query_vector=query_vector,
        retrieval_limit=1,
        score_threshold=0.0,
    )
    
    assert len(single_result) == 1
    assert "original_text" in single_result[0].payload
    assert "source" in single_result[0].payload
    assert single_result[0].score > 0.0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieve_multiple_documents(populated_rag_collection):
    """Test retrieving multiple relevant documents."""
    query = "AI and machine learning systems"
    query_vector = embed(clean(query))
    
    results = await vector_service.search(
        collection_name=populated_rag_collection,
        query_vector=query_vector,
        retrieval_limit=3,
        score_threshold=0.0,
    )
    
    assert len(results) >= 2  # Should find ML and NLP documents
    
    retrieved_texts = [r.payload["original_text"] for r in results]
    # At least one result should be about machine learning
    assert any(
        "machine learning" in text.lower() or "artificial intelligence" in text.lower()
        for text in retrieved_texts
    )


# ============================================================================
# Tests - Semantic Relevance
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_semantic_similarity_web_framework(populated_rag_collection):
    """Test semantic retrieval for web framework queries."""
    query = "How to build REST API with fast framework?"
    query_vector = embed(clean(query))
    
    results = await vector_service.search(
        collection_name=populated_rag_collection,
        query_vector=query_vector,
        retrieval_limit=5,
        score_threshold=0.0,
    )
    
    assert len(results) >= 1
    # FastAPI or web framework related document should appear in results
    retrieved_texts = [r.payload["original_text"] for r in results]
    assert any(
        "FastAPI" in text or "web framework" in text or "API" in text
        for text in retrieved_texts
    ), f"Expected API/framework related content, got: {retrieved_texts[:2]}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_semantic_similarity_vector_database(populated_rag_collection):
    """Test semantic retrieval for vector database queries."""
    query = "Store embeddings and search by similarity"
    query_vector = embed(clean(query))
    
    results = await vector_service.search(
        collection_name=populated_rag_collection,
        query_vector=query_vector,
        retrieval_limit=3,
        score_threshold=0.0,
    )
    
    assert len(results) > 0
    # Qdrant/vector database document should be retrieved
    retrieved_texts = [r.payload["original_text"] for r in results]
    assert any(
        "vector" in text.lower() and "similar" in text.lower()
        for text in retrieved_texts
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_semantic_similarity_nlp(populated_rag_collection):
    """Test semantic retrieval for NLP queries."""
    query = "Processing human language with computers"
    query_vector = embed(clean(query))
    
    results = await vector_service.search(
        collection_name=populated_rag_collection,
        query_vector=query_vector,
        retrieval_limit=3,
        score_threshold=0.0,
    )
    
    assert len(results) > 0
    retrieved_texts = [r.payload["original_text"] for r in results]
    assert any(
        "language" in text.lower() and "processing" in text.lower()
        for text in retrieved_texts
    )


# ============================================================================
# Tests - Score Threshold
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_score_threshold_filters_results(populated_rag_collection):
    """Test that score threshold filters out irrelevant results."""
    # Query very different from stored documents
    query = "Recipe for chocolate cake"
    query_vector = embed(clean(query))
    
    # With high threshold, should return few or no results
    results_high_threshold = await vector_service.search(
        collection_name=populated_rag_collection,
        query_vector=query_vector,
        retrieval_limit=5,
        score_threshold=0.8,  # High threshold - strict matching
    )
    
    # With low threshold, should return more results
    results_low_threshold = await vector_service.search(
        collection_name=populated_rag_collection,
        query_vector=query_vector,
        retrieval_limit=5,
        score_threshold=0.0,  # Low threshold - permissive matching
    )
    
    assert len(results_high_threshold) <= len(results_low_threshold)


# ============================================================================
# Tests - Payload Integrity
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieved_payload_contains_metadata(populated_rag_collection):
    """Test that retrieved results contain expected payload fields."""
    query = "Python programming"
    query_vector = embed(clean(query))
    
    results = await vector_service.search(
        collection_name=populated_rag_collection,
        query_vector=query_vector,
        retrieval_limit=1,
        score_threshold=0.0,
    )
    
    assert len(results) == 1
    payload = results[0].payload
    
    assert "original_text" in payload
    assert "source" in payload
    assert payload["source"] == "test_integration"
    assert len(payload["original_text"]) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieved_text_unchanged(populated_rag_collection):
    """Test that retrieved text matches exactly one of the stored documents."""
    query = "What is available in the knowledgebase?"
    query_vector = embed(clean(query))
    
    results = await vector_service.search(
        collection_name=populated_rag_collection,
        query_vector=query_vector,
        retrieval_limit=5,
        score_threshold=0.0,
    )
    
    # All results should match exactly one of the 5 stored documents
    expected_documents = [
        "Python is a high-level, interpreted programming language known for "
        "its readability and versatility. It supports multiple programming "
        "paradigms including object-oriented, imperative, and functional.",
        "Machine learning is a subset of artificial intelligence that focuses "
        "on building systems that learn from data. Common approaches include "
        "supervised learning, unsupervised learning, and reinforcement learning.",
        "FastAPI is a modern, fast web framework for building APIs with Python "
        "3.7+ based on standard Python type hints. It provides automatic "
        "OpenAPI documentation and async support.",
        "Vector databases like Qdrant are designed to store and query high-dimensional "
        "vector data. They enable semantic search by finding vectors similar to a "
        "given query vector using metrics like cosine similarity.",
        "Natural language processing (NLP) is a field of AI that focuses on the "
        "interaction between computers and human language. Tasks include text "
        "classification, named entity recognition, and sentiment analysis.",
    ]
    
    for result in results:
        retrieved_text = result.payload["original_text"]
        assert retrieved_text in expected_documents, (
            f"Retrieved text doesn't match any stored document. "
            f"Got: {retrieved_text[:100]}..."
        )


# ============================================================================
# Tests - Edge Cases
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_empty_collection():
    """Test searching an empty collection returns no results."""
    collection_name = "empty_rag_collection"
    client = AsyncQdrantClient(host="localhost", port=6333)
    
    if await client.collection_exists(collection_name=collection_name):
        await client.delete_collection(collection_name=collection_name)
    
    await client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    
    query_vector = embed("test query")
    results = await vector_service.search(
        collection_name=collection_name,
        query_vector=query_vector,
        retrieval_limit=5,
        score_threshold=0.0,
    )
    
    assert len(results) == 0
    
    # Cleanup
    await client.delete_collection(collection_name=collection_name)
    await client.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieval_limit_respected(populated_rag_collection):
    """Test that retrieval limit is respected."""
    query = "programming AI web database language"
    query_vector = embed(clean(query))
    
    limit = 2
    results = await vector_service.search(
        collection_name=populated_rag_collection,
        query_vector=query_vector,
        retrieval_limit=limit,
        score_threshold=0.0,
    )
    
    assert len(results) <= limit


@pytest.mark.integration
@pytest.mark.asyncio
async def test_semantic_query_different_words_same_meaning(populated_rag_collection):
    """Test semantic search finds relevant content even with different words."""
    # Query uses different terminology - "programming", "software" instead of "Python"
    query = "software development with readable code and multiple paradigms"
    query_vector = embed(clean(query))
    
    results = await vector_service.search(
        collection_name=populated_rag_collection,
        query_vector=query_vector,
        retrieval_limit=5,
        score_threshold=0.0,
    )
    
    assert len(results) > 0
    retrieved_texts = [r.payload["original_text"] for r in results]
    # Should find at least one technical document (Python or generic programming)
    assert any(
        "programming" in text.lower() or "software" in text.lower() or "development" in text.lower()
        for text in retrieved_texts
    ), f"Expected programming-related content, got: {retrieved_texts[:2]}"
