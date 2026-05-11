import pytest

pytestmark = pytest.mark.integration

EXPLANATION_SYSTEM_PROMPT = (
    "You are a helpful assistant. Explain the user's question in detail. "
    "Provide thorough, comprehensive answers with examples and elaboration."
)

@pytest.mark.parametrize(
    "simple_prompt, complex_prompt", 
    [
        (
            "Explain behavioral testing",
            "Explain behavioral testing in the context of integration tests for a curious beginner. Use short sentences with common words. Avoid jargon and technical terms.",
        )
    ],
)
def test_directional_expectation_complexity(simple_prompt, complex_prompt, llm_client):
    simple_response = llm_client.invoke(simple_prompt, system_prompt=EXPLANATION_SYSTEM_PROMPT)
    complex_response = llm_client.invoke(complex_prompt, system_prompt=EXPLANATION_SYSTEM_PROMPT)
    assert len(simple_response["message"]) > len(complex_response["message"]) 
