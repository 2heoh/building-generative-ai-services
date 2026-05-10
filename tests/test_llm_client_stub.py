class StubLLMClient:
    def invoke(self, query):
        if query == "specific query": 
            return "specific response"
        return "default response"

def process_query(query, llm_client):
    response = llm_client.invoke(query)
    return response

def test_stub_llm_client():
    llm_client = StubLLMClient()
    query = "specific query"
    response = process_query(query, llm_client)
    assert response == "specific response"