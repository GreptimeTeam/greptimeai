# langchain-example

NOTE: this will download docker image, and use host network, and this is only OK on Linux so far. This will be resolved soom.

TODO(yuanbohan): make example a docker image, and compose up all services in containers

## Prerequisites

- docker
- OpenAI API KEY

## Ports will be used

| servise        | port      |
|----------------|-----------|
| Flask          | 8000      |
| Prometheus     | 9090      |
| clickhouse     | 8123,9000 |
| otel-collector | 4317,4318 |

## Start service (Optional)

```
docker compose -f docker/docker-compose.yml up prometheus clickhouse -d
docker compose -f docker/docker-compose.yml up otel-collector -d
```

## Development

- install [rye](https://rye-up.com/guide/installation/)
- rye add greptime-llm-langchain-instrument --path ../langchain/ --dev
- rye sync
- export OPENAI_API_KEY=sk-xxx
- rye run app

then Flask will listen on :8000, and you can use cURL to try

```
curl -X POST -H 'Content-Type: application/json' http://127.0.0.1:8000/langchain/chat -d '{"message":"give me a baby name"}'
```
