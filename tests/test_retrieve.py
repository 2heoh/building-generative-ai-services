def test_search_db(db_client): 
    result = db_client.search(
        collection_name="test", query_vector=[0.18, 0.81, 0.75, 0.12], limit=1
    )
    assert result is not None