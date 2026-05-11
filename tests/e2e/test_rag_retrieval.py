import io
import asyncio
import pytest
from httpx import ASGITransport, AsyncClient
from reportlab.pdfgen import canvas
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from main import app, models
from models import load_text_model
from rag.service import vector_service
from rag.transform import embed


@pytest.fixture(scope="function")
def text_model():
    """Ensure the text model is loaded for tests in this session."""
    if "text" not in models:
        models["text"] = load_text_model()
    yield models["text"]


@pytest.fixture(scope="function")
async def db_client():
    client = AsyncQdrantClient(host="localhost", port=6333)
    collection_name = "knowledgebase"
    if await client.collection_exists(collection_name=collection_name):
        await client.delete_collection(collection_name=collection_name)
    await client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    yield client
    if await client.collection_exists(collection_name=collection_name):
        await client.delete_collection(collection_name=collection_name)
    await client.close()


def create_minimal_pdf(content: str) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    y = 750
    for line in content.splitlines():
        pdf.drawString(100, y, line)
        y -= 20
    pdf.save()
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_rag_search_returns_uploaded_content(db_client: AsyncQdrantClient):
    """Test that text uploaded via /upload can be retrieved via RAG search."""
    unique_phrase = "Captain Underpants is the ultimate superhero"
    pdf_content = create_minimal_pdf(unique_phrase)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        file_data = {"file": ("test_rag.pdf", pdf_content, "application/pdf")}
        response = await client.post("/upload", files=file_data)
        assert response.status_code == 200

    # Wait for background tasks: PDF extraction + embedding + vector storage
    await asyncio.sleep(10)

    # Query the RAG vector store directly
    query_embedding = embed("Who is the superhero?")
    results = await vector_service.search(
        collection_name="knowledgebase",
        query_vector=query_embedding,
        retrieval_limit=3,
        score_threshold=0.0,
    )

    assert len(results) > 0, "RAG search returned no results"

    retrieved_texts = [r.payload["original_text"] for r in results]
    assert any(
        unique_phrase in text for text in retrieved_texts
    ), f"Expected uploaded phrase in retrieved texts, got: {retrieved_texts}"

    # Cleanup
    for ext in ("pdf", "txt"):
        path = f"uploads/test_rag.{ext}"
        import os
        if os.path.exists(path):
            os.remove(path)


@pytest.mark.asyncio
async def test_rag_search_returns_content_for_multiple_queries(
    db_client: AsyncQdrantClient, text_model
):
    """Test that RAG retrieval works for different semantic queries."""
    content = (
        "Qdrant is a vector database for AI applications. "
        "It supports cosine similarity search. "
        "The company was founded in Berlin."
    )
    pdf_content = create_minimal_pdf(content)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        file_data = {"file": ("test_multi.pdf", pdf_content, "application/pdf")}
        response = await client.post("/upload", files=file_data)
        assert response.status_code == 200

    await asyncio.sleep(10)

    queries = [
        "What type of database is Qdrant?",
        "Where was the Qdrant company founded?",
        "What does Qdrant support?",
    ]

    for query in queries:
        query_embedding = embed(query)
        results = await vector_service.search(
            collection_name="knowledgebase",
            query_vector=query_embedding,
            retrieval_limit=3,
            score_threshold=0.0,
        )
        assert len(results) > 0, f"RAG returned no results for query: {query}"
        retrieved_texts = [r.payload["original_text"] for r in results]
        # Each query should find at least one document related to Qdrant
        assert any(
            "Qdrant" in text for text in retrieved_texts
        ), f"Expected 'Qdrant' in results for '{query}', got: {retrieved_texts}"

    # Cleanup
    import os
    for ext in ("pdf", "txt"):
        path = f"uploads/test_multi.{ext}"
        if os.path.exists(path):
            os.remove(path)