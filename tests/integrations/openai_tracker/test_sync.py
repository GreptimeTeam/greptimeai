import time
import uuid

from model import LlmTrace, db
from openai import OpenAI

from greptimeai import openai_patcher

cursor = db.cursor()
client = OpenAI()
openai_patcher.setup(client=client)


def test_chat_completion():
    user_id = str(uuid.uuid4())
    resp = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "1+1=",
            }
        ],
        model="gpt-3.5-turbo",
        user=user_id,
        seed=1,
    )
    assert resp.choices[0].message.content == "2"

    print(resp)
    time.sleep(5)
    sql = (
        "select model,prompt_tokens,completion_tokens from %s where user_id = '%s'"
        % (LlmTrace.table_name, user_id)
    )
    print(sql)
    cursor.execute(sql)

    result = cursor.fetchone()
    print(result)
    assert resp.model == result[0]
    assert resp.usage.prompt_tokens == result[1]
    assert resp.usage.completion_tokens == result[2]
