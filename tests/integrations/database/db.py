import json
import os
from typing import Any, Dict, Optional

import pymysql

from greptimeai import logger

from .model import Tables

connection = pymysql.connect(
    host=os.getenv("MYSQL_HOST"),
    user=os.getenv("MYSQL_USERNAME"),
    passwd=os.getenv("MYSQL_PASSWORD"),
    port=4002,
    db=os.getenv("GREPTIMEAI_DATABASE"),
)


def get_trace_data(user_id: str, span_name: str = "") -> Optional[Dict[str, Any]]:
    sql = f"""
    SELECT
    resource_attributes,
    span_name,
    span_attributes,
    span_events,
    model,
    prompt_tokens,
    completion_tokens
    FROM {Tables.llm_trace}
    WHERE user_id = '{user_id}'
    """
    if span_name:
        sql += f" AND span_name = '{span_name}'"

    with connection.cursor() as cursor:
        cursor.execute(sql)
        trace = cursor.fetchone()
        if not trace:
            return None

        return {
            "resource_attributes": json.loads(trace[0]),
            "span_name": trace[1],
            "span_attributes": json.loads(trace[2]),
            "span_events": json.loads(trace[3]),
            "model": trace[4],
            "prompt_tokens": trace[5],
            "completion_tokens": trace[6],
        }


def truncate_tables():
    """
    truncate all tables
    :return:
    """
    tables = [
        "llm_completion_tokens",
        "llm_completion_tokens_cost",
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
        if "Table not found" in str(e):
            logger.warning("Table not found.")
        else:
            logger.error(e)
            connection.rollback()
