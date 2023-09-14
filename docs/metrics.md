metric tables
====

the detail of metrics collected

## langchain

| table                         | data model | column             | description                               |
|-------------------------------|------------|--------------------|-------------------------------------------|
| `llm_prompt_tokens_total`     | count      | source, llm, model | source=langchain, llm=openai, model=gpt-4 |
| `llm_prompt_tokens_cost`      | gauge      | source, llm, model |                                           |
| `llm_completion_tokens_total` | count      | source, llm, model |                                           |
| `llm_completion_tokens_cost`  | gauge      | source, llm, model |                                           |
| `llm_request_duration_ms`     | histogram  | source, llm, model |                                           |
| `llm_errors_total`            | count      | source, error      | error=Timeout                             |
