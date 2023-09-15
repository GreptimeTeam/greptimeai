import os

from flask import Flask, request
from prometheus_client import generate_latest

from langchain_example import chain, callbacks

app = Flask(__name__)


@app.route("/langchain", methods=['POST'])
def list():
    message = request.json["message"]
    return chain.run(message, callbacks=callbacks)


@app.route('/metrics')
def metrics():
    return generate_latest()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
