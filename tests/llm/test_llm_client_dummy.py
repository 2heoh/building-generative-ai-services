class DummyLLMClient:
    def invoke(self, query, token): 
        return "some response"

def process_query(query, llm_client, token):
    response = llm_client.invoke(query, token) 
    return response

def test_dummy_llm_client():
    llm_client = DummyLLMClient()
    query = "some query"
    response = process_query(query, llm_client, token="fake_token")
    assert response == "some response"