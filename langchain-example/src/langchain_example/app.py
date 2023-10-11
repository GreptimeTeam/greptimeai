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

app = Flask(__name__)
# qa = build_qa()  # this contains a heavy pre-indexing process


@app.route("/langchain/<scenario>", methods=["POST"])
def langchain(scenario: str):
    """
    to chat
    """
    print(f"{ scenario = }")
    message = request.json["message"]
    user_id = request.json["user_id"]
    metadata = {"user_id": user_id}
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
        return message
    else:
        return chat_chain.run(message, callbacks=callbacks, metadata=metadata)


@app.route("/metrics")
def metrics():
    """
    for prometheus
    """
    return generate_latest()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
