trace data
====

the detail of traces collected

Note: this may be changed any time

| name  | event              | attributes to be collected                       | ignored if verbose is False |
|-------|--------------------|--------------------------------------------------|-----------------------------|
| chain | `chain_start`      | metadata, tags, inputs                           | inputs                      |
| chain | `chain_end`        | outputs                                          | outputs                     |
| chain | `chain_error`      | error                                            |                             |
| llm   | `chat_model_start` | metadata, tags, params, messages                 | messages                    |
| llm   | `llm_start`        | metadata, tags, params, prompts                  | prompts                     |
| llm   | `llm_end`          | model, prompt_tokens, completion_tokens, outputs | outputs                     |
| llm   | `llm_error`        | error                                            |                             |
| llm   | `llm_new_token`    |                                                  |                             |
| tool  | `tool_start`       | metadata, tags, name, input                      | input                       |
| tool  | `tool_end`         | output                                           | output                      |
| tool  | `tool_error`       | error                                            |                             |
| agent | `agent_action`     | metadata, tags, type, tool, log, input           | input                       |
| agent | `agent_finish`     | type, log, output                                | output                      |



NOTE:

- `llm.prompts` is List[str]
- `llm.messages` is str
- `llm.outputs` is List[generation]. generation fields: text, finish_reason, log_probability, additional_kwargs, type
