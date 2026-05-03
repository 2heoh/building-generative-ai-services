from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pytest

@pytest.fixture(scope="function") 
def db_client():
    client = QdrantClient(host="localhost", port=6333) 
    if client.collection_exists(collection_name="test"):
        client.delete_collection(collection_name="test")
    client.create_collection( 
        collection_name="test",
        vectors_config=VectorParams(size=4, distance=Distance.DOT),
    )
    client.upsert( 
        collection_name="test",
        points=[
            PointStruct(
                id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"doc": "test.pdf"}
            )
        ],
    )
    yield client 
    client.close() 