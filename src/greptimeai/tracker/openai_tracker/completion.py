from typing import Optional, Callable, Tuple, Any, Dict

import openai
from openai import OpenAI
from openai.types import Completion

from greptimeai import (
    _COMPLETION_COST_LABEL,
    _COMPLETION_TOKENS_LABEL,
    _PROMPT_COST_LABEl,
    _PROMPT_TOKENS_LABEl,
)
from greptimeai.tracker.openai_tracker.public import (
    pre_extractor,
    post_extractor,
)
from greptimeai.utils.openai.parser import (
    parse_chat_completion_message_params,
    parse_choices,
)
from greptimeai.utils.openai.token import (
    get_openai_token_cost_for_model,
)


class CompletionExtractor:
    def __init__(
        self,
        verbose: bool,
        client: Optional[OpenAI] = None,
    ):
        self.verbose = verbose
        self.span_name = "completions.create"
        self.obj = client.completions if client else openai.completions
        self.method_name = "create"
        self.pre_extractor = pre_extractor(self.req_call_func, self.verbose)
        self.post_extractor = post_extractor(self.res_call_func)

    def req_call_func(self) -> Tuple[str, Callable]:
        param_name = "prompt"

        def execute_function(param: Any) -> Any:
            return parse_chat_completion_message_params(param)

        return param_name, execute_function

    def res_call_func(self, resp) -> Dict[str, Any]:
        if isinstance(resp, Completion):
            usage = {}
            if resp.usage:
                usage[_PROMPT_TOKENS_LABEl] = resp.usage.prompt_tokens
            usage[_PROMPT_COST_LABEl] = get_openai_token_cost_for_model(
                resp.model, resp.usage.prompt_tokens, False
            )
            usage[_COMPLETION_TOKENS_LABEL] = resp.usage.completion_tokens
            usage[_COMPLETION_COST_LABEL] = get_openai_token_cost_for_model(
                resp.model, resp.usage.completion_tokens, True
            )

            attrs = {
                "usage": usage,
                "choices": parse_choices(resp.choices, self.verbose),
            }
            return attrs
