# langchain-example

## Prerequisites

- docker
- OpenAI API KEY
- GreptimeCloud service

## Ports will be used

| servise        | port      |
|----------------|-----------|
| Flask          | 8000      |


## Development

- install [rye](https://rye-up.com/guide/installation/)
- rye add greptime-llm-langchain-instrument --path ../langchain/ --dev
- rye sync
- export OPENAI_API_KEY=sk-xxx
- export GREPTIME_LLM_HOST=xxx
- export GREPTIME_LLM_DATABASE=xxx
- export GREPTIME_LLM_USERNAME=xxx
- export GREPTIME_LLM_PASSWORD=xxx
- rye run app

then Flask will listen on :8000, and you can use cURL to try

```
curl -X POST -H 'Content-Type: application/json' http://127.0.0.1:8000/langchain/chat -d '{"message":"give me a baby name", "user_id": "123"}'
```
