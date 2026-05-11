import pytest

pytestmark = pytest.mark.integration

@pytest.mark.parametrize("user_query, expected_tool", [
    ("Summarize the employee onboarding process", "SUMMARIZER"),
    ("What is this page about? https://...", "WEBSEARCH"),
    ("Analyze the 2024 annual accounts", "ANALYZER"),
    ("Analyze this German contract and extract key obligations", "ANALYZER"),
    ("Find the latest news about OpenAI", "WEBSEARCH"),
    ("Generate a summary of this PDF report", "SUMMARIZER"),
    ("Extract action items from this meeting transcript", "ANALYZER"),
    ("What are the main risks mentioned in this audit?", "ANALYZER"),
    ("Search for the best Italian restaurants in Berlin", "WEBSEARCH"),
    ("Summarize the attached Slack conversation", "SUMMARIZER"),
    ("Compare these two CSV files and highlight differences", "ANALYZER"),
    ("Create a short executive summary for this proposal", "SUMMARIZER"),
    ("What is the weather in Frankfurt tomorrow?", "WEBSEARCH"),
    ("Analyze customer sentiment in these reviews", "ANALYZER"),
    ("Explain this Kubernetes error log", "ANALYZER"),
    ("Find documentation for FastAPI background tasks", "WEBSEARCH"),
    ("Summarize the architecture decision record", "SUMMARIZER"),
    ("Who won the Formula 1 race this weekend?", "WEBSEARCH"),
    ("Identify trends in quarterly revenue data", "ANALYZER"),
    ("Provide a concise summary of this research paper", "SUMMARIZER"),
    ("Search for recent articles about MLOps best practices", "WEBSEARCH"),
    ("Analyze dependencies in this Python project", "ANALYZER"),
    ("Summarize the onboarding checklist into bullet points", "SUMMARIZER"),    
])
def test_llm_tool_selection_response(user_query, expected_tool, llm_client):
    response = llm_client.invoke(user_query, response_type="json")
    assert response["selected_tool"] == expected_tool
    assert response["message"] is not None