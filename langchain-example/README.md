# langchain-example

NOTE: this will download docker image, and use host network, and this is only OK on Linux so far

TODO(yuanbohan): make example a docker image, and compose up all services in containers

## prerequisites

- docker
- OpenAI API KEY

## ports will be used

| servise    | port            |
|------------|-----------------|
| Flask      | 8000            |
| Prometheus | 9090            |
| Grafana    | 3000            |
| Jaeger     | 4317/4318/16686 |


## development

- install [rye](https://rye-up.com/guide/installation/)
- rye add greptime-llm-langchain-instrument --path ../langchain/ --dev
- rye sync
- export OPENAI_API_KEY=sk-xxx
- rye run app

then Flask will listen on :8000, and you can use cURL to try

```
curl -X POST -H 'Content-Type: application/json' http://127.0.0.1:8000/langchain -d '{"message":"give me a baby name"}'
```

## explore metrics and traces

#### start service
```
docker compose -f docker/docker-compose.yml up -d
```

#### visit Grafana to visualize mestrics

visit http://localhost:3000, default username/password is `admin`/`admin`, then:

- create a Prometheus connection, use http://localhost:9090, Save and Test
- create a Dashboard, import docs/grafana.json
- and any time you call cURL, metrics and traces will be collected


#### visit Jaeger to visualize traces

http://localhost:16686
