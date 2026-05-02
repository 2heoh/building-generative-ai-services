```sh
curl https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $OPEN_ROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen-2.5-coder-32b-instruct",
    "messages": [
      {"role": "user", "content": "hello"}
    ]    
  }'
```

docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant


```sh
curl -X 'POST' \
    'http://localhost:8000/generate/text' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
  "prompt": "What is your name?",
  "model": "tinyLlama",
  "temperature": 0.01
}'
```