from flask import Flask, request
from langchains import (
    agent_executor,
    build_qa,
    callbacks,
    chat_chain,
    llm_chain,
    retry_llm_chain,
    stream_llm_chain,
)

app = Flask(__name__)
# qa = build_qa()  # this contains a heavy pre-indexing process


@app.route("/langchain/<scenario>", methods=["POST"])
def langchain(scenario: str):
    """
    to chat
    """
    print(f"{ scenario = }")
    json = request.json
    message = json.get("message", "")
    metadata = {"user_id": json.get("user_id", "")}
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
