import time
import uuid

import pytest
from langchain.chains import LLMChain
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from greptimeai.langchain.callback import GreptimeCallbackHandler

from ..database.db import get_trace_data, truncate_tables


@pytest.fixture
def _truncate_tables():
    truncate_tables()
    yield


def test_chat(_truncate_tables):
    user_id = str(uuid.uuid4())
    model = "gpt-3.5-turbo"

    callback = GreptimeCallbackHandler()
    chat = ChatOpenAI(model=model)
    prompt = PromptTemplate.from_template("1 + {number} = ")

    chain = LLMChain(llm=chat, prompt=prompt, callbacks=[callback])
    result = chain.run(number=1, callbacks=[callback], metadata={"user_id": user_id})
    assert result == "2"

    callback.collector._force_flush()

    trace = get_trace_data(user_id=user_id, span_name="langchain_llm")
    retry = 0
    while retry < 3 and not trace:
        retry += 1
        time.sleep(2)
        trace = get_trace_data(user_id=user_id, span_name="langchain_llm")

    assert trace is not None

    assert "langchain" == trace.get("resource_attributes", {}).get("service.name")

    assert ["llm_start", "llm_end"] == [
        event.get("name") for event in trace.get("span_events", [])
    ]

    assert trace.get("model", "").startswith(model)
    assert trace.get("prompt_tokens", 0) > 10
    assert trace.get("completion_tokens", 0) == 1
