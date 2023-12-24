from openai import AsyncOpenAI, OpenAI

from greptimeai import openai_patcher

async_client = AsyncOpenAI()
async_collector = openai_patcher.setup(client=async_client)

sync_client = OpenAI()
sync_collector = openai_patcher.setup(client=sync_client)
