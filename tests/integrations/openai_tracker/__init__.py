from openai import AsyncOpenAI
from openai import OpenAI

from greptimeai import openai_patcher  # type: ignore

async_client = AsyncOpenAI()
openai_patcher.setup(client=async_client)

client = OpenAI()
openai_patcher.setup(client=client)
