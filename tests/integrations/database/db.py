import logging
import os
from typing import Union, List

import pymysql

from .model import Tables

db = pymysql.connect(
    host=os.getenv("GREPTIMEAI_HOST"),
    user=os.getenv("GREPTIMEAI_USERNAME"),
    passwd=os.getenv("GREPTIMEAI_PASSWORD"),
    port=4002,
    db=os.getenv("GREPTIMEAI_DATABASE"),
)
cursor = db.cursor()

trace_sql = "SELECT model,prompt_tokens,completion_tokens FROM %s WHERE user_id = '%s'"
truncate_sql = "TRUNCATE %s"


def get_trace_data(user_id: str) -> List[Union[str, int]]:
    """
    get trace data for llm trace by user_id
    :param is_stream:
    :param user_id:
    :return: model, prompt_tokens, completion_tokens
    """

    cursor.execute(trace_sql % (Tables.llm_trace, user_id))
    trace = cursor.fetchone()
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
        cursor.executemany(truncate_sql, tables)
        db.commit()
    except Exception as e:
        logging.error(e)
        db.rollback()
