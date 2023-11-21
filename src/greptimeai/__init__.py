import logging

from .scope import _NAME, _VERSION

logger = logging.getLogger(f"{_NAME}:{_VERSION}")

_ERROR_TYPE_LABEL = "type"
_CLASS_TYPE_LABEL = "type"


_LLM_SOURCE_LABEL = "llm_source"
_SPAN_NAME_LABEL = "span_name"
_USER_ID_LABEL = "user_id"
_MODEL_LABEL = "model"


_PROMPT_TOKENS_LABEl = "prompt_tokens"
_PROMPT_COST_LABEl = "prompt_cost"
_COMPLETION_TOKENS_LABEL = "completion_tokens"
_COMPLETION_COST_LABEL = "completion_cost"
