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