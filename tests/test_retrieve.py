import pytest

pytestmark = pytest.mark.unit

@pytest.mark.asyncio 
async def test_search_db(async_db_client): 
    result = await async_db_client.query_points(
        collection_name="test", query=[0.18, 0.81, 0.75, 0.12], limit=1
    )
    assert result is not None

