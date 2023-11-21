from typing import Any, Dict, Iterable, Optional, Sequence, Union

from langchain.schema import ChatGeneration, Generation
from langchain.schema.document import Document
from langchain.schema.messages import BaseMessage

_SPAN_NAME_CHAIN = "chain"
_SPAN_NAME_AGENT = "agent"
_SPAN_NAME_LLM = "llm"
_SPAN_NAME_TOOL = "tool"
_SPAN_NAME_RETRIEVER = "retriever"


def _get_user_id(metadata: Optional[Dict[str, Any]]) -> str:
    """
    get user id from metadata
    """
    return (metadata or {}).get("user_id", "")


def _get_serialized_id(serialized: Dict[str, Any]) -> Optional[str]:
    """
    get id if exist
    """
    ids = serialized.get("id")
    if ids and isinstance(ids, list):
        return ids[len(ids) - 1]
    return None


def _get_serialized_streaming(serialized: Dict[str, Any]) -> bool:
    """
    get streaming if exist
    """
    id = _get_serialized_id(serialized)
    if not id:
        return False

    if id == "OpenAI" or id == "ChatOpenAI":
        return serialized.get("kwargs", {}).get("streaming")
    return False


def _parse(obj: Any) -> Union[Dict[str, Any], Sequence[Any], Any]:
    if hasattr(obj, "to_json"):
        return obj.to_json()

    if isinstance(obj, dict):
        return {key: _parse(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [_parse(item) for item in obj]

    return obj


def _parse_input(raw_input: Any) -> Any:
    if not raw_input:
        return None

    if not isinstance(raw_input, dict):
        return _parse(raw_input)

    return (
        raw_input.get("input")
        or raw_input.get("inputs")
        or raw_input.get("question")
        or raw_input.get("query")
        or _parse(raw_input)
    )


def _parse_output(raw_output: dict) -> Any:
    if not raw_output:
        return None

    if not isinstance(raw_output, dict):
        return _parse(raw_output)

    return (
        raw_output.get("text")
        or raw_output.get("output")
        or raw_output.get("output_text")
        or raw_output.get("answer")
        or raw_output.get("result")
        or _parse(raw_output)
    )


def _parse_generation(gen: Generation) -> Optional[Dict[str, Any]]:
    """
    Generation, or ChatGeneration (which contains message field)
    """
    if not gen:
        return None

    info = gen.generation_info or {}
    attrs = {
        "text": gen.text,
        # the following is OpenAI only?
        "finish_reason": info.get("finish_reason"),
        "log_probability": info.get("logprobs"),
    }

    if isinstance(gen, ChatGeneration):
        message: BaseMessage = gen.message
        attrs["additional_kwargs"] = message.additional_kwargs
        attrs["type"] = message.type

    return attrs


def _parse_generations(
    gens: Sequence[Generation],
) -> Optional[Iterable[Dict[str, Any]]]:
    """
    parse LLMResult.generations[0] to structured fields
    """
    if gens and len(gens) > 0:
        return list(filter(None, [_parse_generation(gen) for gen in gens if gen]))

    return None


def _parse_documents(docs: Sequence[Document]) -> Optional[Sequence[Dict[str, Any]]]:
    """
    parse LLMResult.generations[0] to structured fields
    """

    def _parse_doc(doc: Document) -> Dict[str, Any]:
        return {
            "content": doc.page_content,
            "metadata": doc.metadata,
        }

    if docs and len(docs) > 0:
        return [_parse_doc(doc) for doc in docs]

    return None
