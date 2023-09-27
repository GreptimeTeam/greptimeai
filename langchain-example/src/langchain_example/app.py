from flask import Flask, request
from prometheus_client import generate_latest

from langchain_example.langchains import (
    stream_llm_chain,
    retry_llm_chain,
    llm_chain,
    callbacks,
    chat_chain,
    agent_executor,
    build_qa,
)
from langchain_example.llmonitors import (
    llmonitor_callbacks,
    llmonitor_chat_chain,
    llmonitor_agent_executor,
)

app = Flask(__name__)
# qa = build_qa()  # this contains a heavy pre-indexing process


@app.route("/langchain/<scenario>", methods=["POST"])
def langchain(scenario: str):
    """
    to chat
    """
    print(f"{ scenario = }")
    metadata = {"user_id": 927}
    message = request.json["message"]
    if scenario == "retry":
        return retry_llm_chain.run(message, callbacks=callbacks, metadata=metadata)
    elif scenario == "streaming":
        return stream_llm_chain.run(message, callbacks=callbacks, metadata=metadata)
    elif scenario == "llm":
        return llm_chain.run(message, callbacks=callbacks, metadata=metadata)
    elif scenario == "agent":
        return agent_executor.run(message, callbacks=callbacks, metadata=metadata)
    elif scenario == "retrieval":
        # return qa.run(message, callbacks=callbacks, metadata=metadata)
        pass
    else:
        return chat_chain.run(message, callbacks=callbacks, metadata=metadata)


@app.route("/llmonitor", methods=["POST"])
def llmonitor():
    """
    llmonitor demo
    """
    message = request.json["message"]

    return llmonitor_chat_chain.run(message, callbacks=llmonitor_callbacks)
    # return llmonitor_agent_executor.run(message, callbacks=llmonitor_callbacks)


@app.route("/metrics")
def metrics():
    """
    for prometheus
    """
    return generate_latest()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
