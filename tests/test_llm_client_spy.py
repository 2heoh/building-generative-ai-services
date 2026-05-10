class SpyLLMClient:
    def __init__(self):
        self.call_count = 0
        self.calls = []

    def invoke(self, query):
        self.call_count += 1 
        self.calls.append((query))
        return "some response"

def process_query(query, llm_client):
    response = llm_client.invoke(query)
    return response

def test_process_query_with_spy():
    llm_client = SpyLLMClient()
    query = "some query"

    process_query(query, llm_client)

    assert llm_client.call_count == 1
    assert llm_client.calls == [("some query")]