from openai import AsyncOpenAI, OpenAI

from greptimeai import openai_patcher

async_client = AsyncOpenAI()
openai_patcher.setup(client=async_client)

sync_client = OpenAI()
openai_patcher.setup(client=sync_client)
