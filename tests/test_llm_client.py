import pytest
from openai import OpenAI

from main import openai_client 

class LLMClient:
    def invoke(self, query):

        client = openai_client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": query}]
        )

        self.send_request(query)

        return client
    
    def send_request(self, _query):
        return
    

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
    stub.choices = [mocker.Mock()]
    stub.choices[0].message = mocker.Mock()
    stub.choices[0].message.content = "stubbed response"
    mocker.patch.object(openai_client.chat.completions, 'create', return_value=stub)
    
    result = llm_client.invoke("test query")
    
    assert result.choices[0].message.content == "stubbed response"


def test_spy(mocker, llm_client):
    spy = mocker.spy(LLMClient, 'send_request')
    spy.return_value = "some_value"
    
    llm_client.invoke("some query")
    
    assert spy.call_count == 1    

def test_mock(mocker, llm_client):
    client = mocker.Mock()
    client.return_value = "mocked response"
    mocker.patch.object(openai_client.chat.completions, 'create', client)
    
    llm_client.invoke("some query")
    
    client.assert_called_once_with(
        model="gpt-4o", messages=[{"role": "user", "content": "some query"}]
    )
