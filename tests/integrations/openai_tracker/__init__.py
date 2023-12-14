from openai import AsyncOpenAI
from openai import OpenAI

from greptimeai import openai_patcher  # type: ignore
from greptimeai.openai_patcher import _collector  # type: ignore

async_client = AsyncOpenAI()
openai_patcher.setup(client=async_client)

client = OpenAI()
openai_patcher.setup(client=client)


def force_flush():
    _collector._collector._force_flush()
