import os
from typing import List, Union

import pymysql

from greptimeai import logger

from .model import Tables

connection = pymysql.connect(
    host=os.getenv("GREPTIMEAI_HOST"),
    user=os.getenv("GREPTIMEAI_USERNAME"),
    passwd=os.getenv("GREPTIMEAI_PASSWORD"),
    port=4002,
    db=os.getenv("GREPTIMEAI_DATABASE"),
)

trace_sql = f"SELECT model, prompt_tokens, completion_tokens FROM {Tables.llm_trace} WHERE user_id = '%s'"


def get_trace_data(user_id: str) -> List[Union[str, int]]:
    """
    get trace data for llm trace by user_id
    :param user_id:
    :return: model, prompt_tokens, completion_tokens
    """

    with connection.cursor() as cursor:
        cursor.execute("select * from llm_traces_preview_v01 limit 1")
        trace = cursor.fetchone()
        print(f" {type(trace)=} {trace=}")

    with connection.cursor() as cursor:
        cursor.execute(trace_sql % (user_id))
        trace = cursor.fetchone()
        print(f"{type(trace)=} {trace=}")
        if trace is None:
            raise Exception("trace data is None")
        return list(trace)


def truncate_tables():
    """
    truncate all tables
    :return:
    """
    tables = [
        "llm_completion_tokens",
        "llm_completion_tokens_cost",
        "llm_errors",
        "llm_prompt_tokens",
        "llm_prompt_tokens_cost",
        "llm_request_duration_ms_bucket",
        "llm_request_duration_ms_count",
        "llm_request_duration_ms_sum",
        "llm_traces_preview_v01",
    ]
    try:
        with connection.cursor() as cursor:
            cursor.executemany("TRUNCATE %s", tables)
            connection.commit()
    except Exception as e:
        logger.error(e)
        connection.rollback()
