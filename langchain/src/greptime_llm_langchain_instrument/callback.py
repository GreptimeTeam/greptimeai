from typing import Dict, List, Any, Union
from uuid import UUID
import time
import re

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.schema.messages import BaseMessage
from langchain.schema.output import LLMResult

from . import _TimeTable


class GreptimeCallbackHandler(BaseCallbackHandler):

    def __init__(self, verbose=False) -> None:
        "docstring"
        super().__init__()
        self.__time_tables = _TimeTable()

    def __get_llm_repr(self, output: Any) -> str:
        if re.search("openai", repr(output), re.IGNORECASE):
            return "openai"
        else:
            return "unknown"

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain starts running."""
        print(f"on_chain_start. { inputs = }")
        print(f"on_chain_start. { run_id = }")
        print(f"on_chain_start. { parent_run_id = }")
        print(f"on_chain_start. { kwargs = }")

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain ends running."""
        print(f"on_chain_end. { outputs = }")
        print(f"on_chain_end. { run_id = }")
        print(f"on_chain_end. { parent_run_id = }")
        print(f"on_chain_end. { kwargs = }")

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain errors."""
        print(f"on_chain_error. { error = }")
        print(f"on_chain_error. { run_id = }")
        print(f"on_chain_error. { parent_run_id = }")
        print(f"on_chain_error. { kwargs = }")

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM starts running."""
        self.__time_tables.set(run_id)
        print(f"on_llm_start. { prompts = }")
        print(f"on_llm_start. { run_id = }")
        print(f"on_llm_start. { parent_run_id = }")
        print(f"on_llm_start. { kwargs = }")

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Chat Model starts running."""
        self.__time_tables.set(run_id)
        print(f"on_chat_model_start. { messages = }")
        print(f"on_chat_model_start. { run_id = }")
        print(f"on_chat_model_start. { parent_run_id = }")
        print(f"on_chat_model_start. { kwargs = }")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Collecting token usage, and performance.

        # Table

        langchain_prompt_tokens_count{llm,model}
        langchain_completion_tokens_count{llm,model}
        langchain_tokens_cost{llm,model}
        langchain_callback_duration_seconds{llm,model,error}
        """
        output = (response.llm_output or {})
        model = output.get("model_name")
        llm = self.__get_llm_repr(output)

        token_usage = output.get("token_usage", {})
        completion_tokens = token_usage.get("completion_tokens", 0)
        prompt_tokens = token_usage.get("prompt_tokens", 0)

        latency = self.__time_tables.latency(run_id)
        print(f"on_llm_end. { latency = }s")

    def on_llm_error(self,
                     error: Union[Exception, KeyboardInterrupt],
                     *,
                     run_id: UUID,
                     parent_run_id: Union[UUID, None] = None,
                     **kwargs: Any) -> Any:
        """Run when LLM errors."""
        print(f"on_llm_error. { error = }")
        print(f"on_llm_error. { run_id = }")
        print(f"on_llm_error. { parent_run_id = }")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        ...

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str,
                      **kwargs: Any) -> Any:
        """Run when tool starts running."""
        ...

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        ...

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt],
                      **kwargs: Any) -> Any:
        """Run when tool errors."""
        ...

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""
        ...

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        ...

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        ...
