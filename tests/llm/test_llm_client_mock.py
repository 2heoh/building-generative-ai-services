def process_query(query, llm_client):
    response = llm_client.invoke(query)
    return response

def test_process_query_with_mock(mocker):
    llm_client = mocker.Mock() 
    llm_client.invoke.return_value = "mock response"
    query = "some query"

    process_query(query, llm_client)
    process_query(query, llm_client)

    assert llm_client.invoke.call_count == 2
    llm_client.invoke.assert_any_call("some query")