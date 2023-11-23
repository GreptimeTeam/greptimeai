from typing import Optional, Callable, List, Tuple, Any, Dict

import openai
from openai import OpenAI

from greptimeai.tracker.openai_tracker.public import (
    pre_extractor,
    post_extractor,
)
from greptimeai import (
    _COMPLETION_COST_LABEL,
    _COMPLETION_TOKENS_LABEL,
    _MODEL_LABEL,
    _USER_ID_LABEL,
    _PROMPT_COST_LABEl,
    _PROMPT_TOKENS_LABEl,
)
from greptimeai.utils.openai.token import (
    get_openai_token_cost_for_model,
)

from openai.types import CreateEmbeddingResponse


class EmbeddingExtractor:
    def __init__(
        self,
        verbose: bool,
        client: Optional[OpenAI] = None,
    ):
        self.verbose = verbose
        self.span_name = "embeddings.create"
        self.obj = client.embeddings if client else openai.embeddings
        self.method_name = "create"
        self.pre_extractor = pre_extractor(self.req_call_func, self.verbose)
        self.post_extractor = post_extractor(self.res_call_func)

    def req_call_func(self) -> Tuple[str, Callable]:
        param_name = ""

        def execute_function(param: Any) -> Any:
            return

        return param_name, execute_function

    def res_call_func(self, resp) -> Dict[str, Any]:
        if isinstance(resp, CreateEmbeddingResponse):
            usage = {}
            if resp.usage:
                usage[_PROMPT_TOKENS_LABEl] = resp.usage.prompt_tokens
            usage[_PROMPT_COST_LABEl] = get_openai_token_cost_for_model(
                resp.model, resp.usage.prompt_tokens, False
            )
            usage[_COMPLETION_TOKENS_LABEL] = 0
            usage[_COMPLETION_COST_LABEL] = 0

            attrs = {
                "usage": usage,
            }
            return attrs
