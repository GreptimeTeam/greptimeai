trace data
====

the detail of traces collected

| event                 | attributes to be collected                               | ignored if verbose is False |
|-----------------------|----------------------------------------------------------|-----------------------------|
| `on_chain_start`      | kwargs, metadata, tags, inputs                           | inputs                      |
| `on_chain_end`        | kwargs, outputs                                          | outputs                     |
| `on_chain_error`      | kwargs, error                                            |                             |
| `on_chat_model_start` | kwargs, metadata, tags, params, messages                 | messages                    |
| `on_llm_start`        | kwargs, metadata, tags, params, prompts                  | prompts                     |
| `on_llm_end`          | kwargs, model, prompt_tokens, completion_tokens, outputs | outputs                     |
| `on_llm_error`        | kwargs, error                                            |                             |
|-----------------------|----------------------------------------------------------|-----------------------------|
| `on_llm_new_token`    |                                                          |                             |
| `on_tool_start`       |                                                          |                             |
| `on_tool_end`         |                                                          |                             |
| `on_tool_error`       |                                                          |                             |
| `on_agent_action`     |                                                          |                             |
| `on_agent_finish`     |                                                          |                             |
