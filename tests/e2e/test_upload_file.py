import io
import os

import pytest
from httpx import ASGITransport, AsyncClient
from pypdf import PdfWriter
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from main import app


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


def _create_minimal_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_upload_file(db_client):
    pdf_content = _create_minimal_pdf()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        file_data = {"file": ("test.pdf", pdf_content, "application/pdf")}
        response = await client.post("/upload", files=file_data)

        assert response.status_code == 200
        body = response.json()
        assert body["filename"] == "test.pdf"
        assert body["message"] == "File uploaded successfully"

    for ext in ["pdf", "txt"]:
        path = os.path.join("uploads", f"test.{ext}")
        if os.path.exists(path):
            os.remove(path)

            