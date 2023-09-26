from flask import Flask, request
from prometheus_client import generate_latest

from langchain_example.langchains import (
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


@app.route("/langchain", methods=["POST"])
def langchain():
    """
    to chat
    """
    message = request.json["message"]

    # return llm_chain.run(message, callbacks=callbacks)
    # return chat_chain.run(message, callbacks=callbacks)
    return agent_executor.run(message, callbacks=callbacks)
    # return qa.run(message, callbacks=callbacks)


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
