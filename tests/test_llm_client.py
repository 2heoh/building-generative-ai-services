import pytest
from openai import OpenAI

from main import openai_client 

class LLMClient:
    def invoke(self, query):
        return openai_client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": query}]
        )
    

@pytest.fixture
def llm_client():
    return LLMClient()

def test_fake(mocker, llm_client):
    class FakeOpenAIClient: 
        @staticmethod
        def create(model, messages):
            return {"choices": [{"message": {"content": "fake response"}}]}

    mocker.patch.object(openai_client.chat, 'completions', new=FakeOpenAIClient)
    result = llm_client.invoke("test query")
    assert result == {"choices": [{"message": {"content": "fake response"}}]}


def test_stub(mocker, llm_client):
    stub = mocker.Mock()
    stub.process.return_value = "stubbed response"
    result = llm_client.invoke(stub)
    assert result == "stubbed response"