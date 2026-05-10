import textstat
import pytest

@pytest.mark.parametrize("prompt, expected_score", [ 
    ("Explain behavioral testing", 60),
    ("Explain behavioral testing as simple as you can", 70),
])
def test_minimum_functionality_readability(prompt, expected_score, llm_client):
    response = llm_client.invoke(prompt)

    readability_score = textstat.flesch_reading_ease(response) 

    assert expected_score < readability_score < 90