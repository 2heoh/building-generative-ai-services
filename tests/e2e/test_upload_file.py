import io
import os
import asyncio

import pytest
from httpx import ASGITransport, AsyncClient
from reportlab.pdfgen import canvas
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from main import app, models
from models import load_text_model
from rag.constants import KNOWLEDGE_BASE_COLLECTION


@pytest.fixture(scope="function")
def text_model():
    """Ensure the text model is loaded for tests in this session."""
    if "text" not in models:
        models["text"] = load_text_model()
    yield models["text"]


@pytest.fixture(scope="function")
async def db_client():
    """Ensure production knowledgebase exists; never delete it after tests."""
    client = AsyncQdrantClient(host="localhost", port=6333)
    if not await client.collection_exists(collection_name=KNOWLEDGE_BASE_COLLECTION):
        await client.create_collection(
            collection_name=KNOWLEDGE_BASE_COLLECTION,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
    yield client
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
async def test_upload_file(db_client: AsyncQdrantClient):
    pdf_content = create_minimal_pdf("Test file content")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        file_data = {"file": ("test.pdf", pdf_content, "application/pdf")}
        response = await client.post("/upload", files=file_data)

        assert response.status_code == 200
        body = response.json()
        assert body["filename"] == "test.pdf"
        assert body["message"] == "File uploaded successfully"

    # cleanup
    for ext in ["pdf", "txt"]:
        path = os.path.join("uploads", f"test.{ext}")
        if os.path.exists(path):
            os.remove(path)


@pytest.mark.asyncio
async def test_rag_user_workflow(db_client: AsyncQdrantClient, text_model):
    pdf_content = create_minimal_pdf("Ali Parandeh is a software engineer")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        file_data = {"file": ("test.pdf", pdf_content, "application/pdf")}
        upload_response = await client.post("/upload", files=file_data)

        assert upload_response.status_code == 200

        # Wait for background tasks (PDF extraction + vector storage) to complete
        # before querying the RAG system
        await asyncio.sleep(10)

        generate_response = await client.post(
            "/generate/text",
            json={
                "model": "tinyLlama",
                "prompt": "Who is Ali Parandeh?",
                "temperature": 0.7,
            },
        )

        assert generate_response.status_code == 200
        response_json = generate_response.json()
        assert "content" in response_json
        print(response_json["content"].lower())
        assert "software engineer" in response_json["content"].lower()

    # cleanup
    for ext in ["pdf", "txt"]:
        path = os.path.join("uploads", f"test.{ext}")
        if os.path.exists(path):
            os.remove(path)