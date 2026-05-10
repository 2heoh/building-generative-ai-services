from unittest.mock import MagicMock

import requests


class FakeLLMClient:
    def __init__(self):
        self.cache = dict()

    def invoke(self, query, token=None):
        if query in self.cache:
            return self.cache.get(query)

        response = requests.post("http://localhost:8001", json={"query": query})
        if response.status_code != 200:
            return "Error fetching result"

        result = response.json().get("response")
        self.cache[query] = result
        return result


def process_query(query, llm_client, token):
    response = llm_client.invoke(query, token)
    return response


def test_fake_llm_client():
    llm_client = FakeLLMClient()
    llm_client.invoke = MagicMock(return_value="some response")
    query = "some query"
    response = process_query(query, llm_client, token="fake_token")
    assert response == "some response"
    llm_client.invoke.assert_called_once_with("some query", "fake_token")