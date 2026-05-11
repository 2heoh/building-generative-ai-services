import textstat
import pytest



pytestmark = pytest.mark.integration



EXPLANATION_SYSTEM_PROMPT = (
    "You are a helpful assistant. Explain the user's question in very simple terms. "
    "Use short sentences and simple words. Write as if explaining to a 10-year-old. "
    "Avoid jargon and complex language."
)

@pytest.mark.parametrize("prompt, expected_score", [ 
    ("Explain behavioral testing", 40),
    ("Explain behavioral testing as simple as you can", 60),
])
def test_minimum_functionality_readability(prompt, expected_score, llm_client):
    
    response = llm_client.invoke(prompt, system_prompt=EXPLANATION_SYSTEM_PROMPT)

    text = response["message"]
    readability_score = textstat.flesch_reading_ease(text) 
    assert expected_score < readability_score < 90
