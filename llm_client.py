import json
from openai import OpenAI

TOOL_SELECTION_SYSTEM_PROMPT = """You are a tool selection assistant. Given a user query, you must classify it into one of the following tools:

- SUMMARIZER: For requests to summarize, condense, or provide an overview of content.
- WEBSEARCH: For requests that involve looking up information from a URL or web page.
- ANALYZER: For requests to analyze, examine, or evaluate data, documents, or content.

Respond in JSON format with two fields:
- "selected_tool": one of SUMMARIZER, WEBSEARCH, or ANALYZER
- "message": a brief explanation of why this tool was selected
"""


class LLMClient:
    def __init__(self, client: OpenAI):
        self.client = client

    def invoke(self, query: str, response_type: str = "text") -> dict:
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": TOOL_SELECTION_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            response_format={"type": "json_object"} if response_type == "json" else None,
        )
        content = response.choices[0].message.content
        if response_type == "json":
            return json.loads(content)
        return {"message": content}