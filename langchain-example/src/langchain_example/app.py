from flask import Flask, request
from prometheus_client import generate_latest

# from langchain_example import chain, callbacks
from langchain_example import qa, callbacks

app = Flask(__name__)


@app.route("/langchain", methods=["POST"])
def langchain():
    """
    to chat
    """
    message = request.json["message"]

    # return chain.run(message, callbacks=callbacks)
    # return qa.run("give me a summary about the second story")
    return qa.run(message, callbacks=callbacks)


@app.route("/metrics")
def metrics():
    """
    for prometheus
    """
    return generate_latest()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
