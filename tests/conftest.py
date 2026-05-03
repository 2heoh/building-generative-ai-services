from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pytest

@pytest.fixture(scope="function") 
async def async_db_client():
    client = AsyncQdrantClient(host="localhost", port=6333) 
    if await client.collection_exists(collection_name="test"):
        await client.delete_collection(collection_name="test")
    await client.create_collection( 
        collection_name="test",
        vectors_config=VectorParams(size=4, distance=Distance.DOT),
    )
    await client.upsert( 
        collection_name="test",
        points=[
            PointStruct(
                id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"doc": "test.pdf"}
            )
        ],
    )
    yield client 
    await client.close() 
