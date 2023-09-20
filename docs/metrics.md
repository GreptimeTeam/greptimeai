metric tables
====

the detail of metrics collected

TODO(yuanbohan): chain, agent, tool needed?

## langchain

| table                         | data model | column       | description   |
|-------------------------------|------------|--------------|---------------|
| `llm_prompt_tokens_total`     | count      | model        | model=gpt-4   |
| `llm_prompt_tokens_cost`      | gauge      | model        |               |
| `llm_completion_tokens_total` | count      | model        |               |
| `llm_completion_tokens_cost`  | gauge      | model        |               |
| `llm_request_duration_ms`     | histogram  | model        |               |
| `llm_errors_total`            | count      | model, error | error=Timeout |
