trace data
====

the detail of traces collected

Note: this may be changed any time

| name  | event                 | attributes to be collected                       | ignored if verbose is False |
|-------|-----------------------|--------------------------------------------------|-----------------------------|
| chain | `on_chain_start`      | metadata, tags, inputs                           | inputs                      |
| chain | `on_chain_end`        | outputs                                          | outputs                     |
| chain | `on_chain_error`      | error                                            |                             |
| llm   | `on_chat_model_start` | metadata, tags, params, messages                 | messages                    |
| llm   | `on_llm_start`        | metadata, tags, params, prompts                  | prompts                     |
| llm   | `on_llm_end`          | model, prompt_tokens, completion_tokens, outputs | outputs                     |
| llm   | `on_llm_error`        | error                                            |                             |
| llm   | `on_llm_new_token`    |                                                  |                             |
| tool  | `on_tool_start`       |                                                  |                             |
| tool  | `on_tool_end`         |                                                  |                             |
| tool  | `on_tool_error`       |                                                  |                             |
| agent | `on_agent_action`     |                                                  |                             |
| agent | `on_agent_finish`     |                                                  |                             |
