from openai import AsyncOpenAI, OpenAI

from greptimeai import openai_patcher

async_client = AsyncOpenAI()
openai_patcher.setup(client=async_client)

client = OpenAI()
openai_patcher.setup(client=client)
