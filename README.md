# greptime-llm-instrument

Greptime instrument sdk for popular LLM (LangChain, OpenAI, etc.)

## Subprojects

- `greptime-llm-langchain-instrument` Instrumentation extension for [LangChain
  python](https://python.langchain.com/docs/get_started/introduction.html).

## Build

For Python projects, we use [`rye`](https://rye-up.com) for dependency
management and build. To build the project, run `rye build` at project
root. Distribution will be generated in `dist` directory.
