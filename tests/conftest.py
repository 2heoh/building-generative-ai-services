from qdrant_client import AsyncQdrantClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from llm_client import LLMClient
from openai import OpenAI

import os
from dotenv import load_dotenv
load_dotenv()

from aiohttp import ClientSession
from main import app

import pytest

# Isolated Qdrant collections for tests — never delete production knowledgebase.
TEST_DORA_COLLECTION = "test_dora_knowledgebase"
TEST_E2E_COLLECTION = "test_e2e_knowledgebase"


def pytest_collection_modifyitems(items):
    """Reorder tests so unit tests run before integration tests."""
    unit_tests = []
    integration_tests = []
    other_tests = []

    for item in items:
        if item.get_closest_marker("integration"):
            integration_tests.append(item)
        elif item.get_closest_marker("unit"):
            unit_tests.append(item)
        else:
            other_tests.append(item)

    items[:] = unit_tests + other_tests + integration_tests


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
            ),
            PointStruct(
                id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"doc": "test2.pdf"}
            ),
            PointStruct(
                id=3, vector=[0.36, 0.55, 0.47, 0.94], payload={"doc": "test3.pdf"}
            ),
        ],
    )
    yield client 
    client.delete_collection(collection_name="test")
    client.close() 


@pytest.fixture(scope="function")
def llm_client():
    client = LLMClient(OpenAI(api_key=os.getenv('OPENAI_API_KEY')) )

    yield client
    


@pytest.fixture
async def test_client():
    async with ClientSession() as client:
        yield client