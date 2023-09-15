# greptime-llm-instrument

Greptime instrument sdk for popular LLM (LangChain, OpenAI, etc.)

## Subprojects

- `langchain` Instrumentation extension for [LangChain python][langchain].
- `langchain-example` to illustrate how to use greptime-llm-langchain-instrument

## Build

For Python projects, we use [`rye`](https://rye-up.com) for dependency
management and build. To build the project, run `rye build` at project
root. Distribution will be generated in `dist` directory.

[langchain]: https://python.langchain.com/docs/get_started/introduction.html
