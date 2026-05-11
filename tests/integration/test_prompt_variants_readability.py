import pytest
import textstat

user_prompt = "Explain behavioral testing"

@pytest.mark.parametrize("prompt, expected_score", [
    (user_prompt, 50),
    (user_prompt.upper(), 50),
    (user_prompt.replace("behavioral", "behavioural"), 50),
    ("  Explain behavioral testing  ", 50),
    ("Explain    behavioral    testing", 50),
    (user_prompt.lower(), 50),
    ("eXpLaIn BeHaViOrAl TeStInG", 50),
    ("Explain behavioral testing.", 50),
    ("Can you explain behavioral testing?", 50),
    ("Describe behavioral testing", 50),
    ("Explain behavioural testing!", 50),
    ("Explain behavorial testing", 50),
    ("Explain behavioral testing in software engineering", 50),
    ("\nExplain behavioral testing", 50),
    ("Explain\tbehavioral\ttesting", 50),    
])
def test_modified_prompt_readability(prompt, expected_score, llm_client):
    explain_prompt = "You are a friendly tutor writing for a popular magazine. Explain the concept in plain, simple language. Use short sentences with common words. Avoid jargon and technical terms. Write as if you are talking to a curious beginner."
    response = llm_client.invoke(prompt, system_prompt=explain_prompt)

    readability_score = textstat.flesch_reading_ease(response["message"])

    
    assert expected_score < readability_score < 90