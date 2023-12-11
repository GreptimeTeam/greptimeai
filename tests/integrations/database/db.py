import os
from typing import Tuple

import pymysql  # type: ignore

from .model import LlmTrace

db = pymysql.connect(
    host=os.getenv("GREPTIMEAI_HOST"),
    user=os.getenv("GREPTIMEAI_USERNAME"),
    passwd=os.getenv("GREPTIMEAI_PASSWORD"),
    port=4002,
    db=os.getenv("GREPTIMEAI_DATABASE"),
)

cursor = db.cursor()

trace_sql = "SELECT model,prompt_tokens,completion_tokens FROM %s WHERE user_id = '%s'"
metric_sql = "SELECT service_name,greptime_value FROM %s WHERE model = '%s'"
truncate_sql = "TRUNCATE %s"


def get_trace_data(user_id: str) -> Tuple:
    """
    get trace data for llm trace by user_id
    :param user_id:
    :return: model, prompt_tokens, completion_tokens
    """
    cursor.execute(trace_sql % (LlmTrace.table_name, user_id))
    trace = cursor.fetchone()
    if trace is None:
        raise Exception("trace data is None")
    return trace


def get_metric_data(table: str, model: str) -> Tuple:
    """
    get metric data by table and model
    :param table:
    :param model:
    :return: service_name, greptime_value
    """
    cursor.execute(metric_sql % (table, model))
    metric = cursor.fetchone()
    if metric is None:
        raise Exception("metric data is None")
    return metric


def truncate_tables():
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
        "numbers",
    ]
    try:
        cursor.executemany(truncate_sql, tables)
        db.commit()
    except:
        db.rollback()
