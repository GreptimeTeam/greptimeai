_ERROR_TYPE_LABEL = "type"
_CLASS_TYPE_LABEL = "type"

_SPAN_NAME_LABEL = "span_name"
_USER_ID_LABEL = "user_id"
_MODEL_LABEL = "model"

_PROMPT_TOKENS_LABEl = "prompt_tokens"
_PROMPT_COST_LABEl = "prompt_cost"
_COMPLETION_TOKENS_LABEL = "completion_tokens"
_COMPLETION_COST_LABEL = "completion_cost"

# field in metric table, and field in span_attributes in trace table.
# so far:
#   - openai
#   - langchain
_SOURCE_LABEL = "source"
_SOURCE_VERSION_LABEL = "source_version"

# the following labels are for traces, not for metrics
_INPUT_DISPLAY_LABEL = "inputs"  # for DISPLAY, DO NOT USE IT FOR ANYTHING ELSE
_OUTPUT_DISPLAY_LABEL = "outputs"  # for DISPLAY, DO NOT USE IT FOR ANYTHING ELSE
