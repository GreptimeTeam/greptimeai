from typing import Any, Dict

from openai._response import APIResponse

from greptimeai import logger


def parse_raw_response(resp: APIResponse) -> Dict[str, Any]:
    dict = {
        "headers": resp.headers,
        "status_code": resp.status_code,
        "url": resp.url,
        "method": resp.method,
        "cookies": resp.http_response.cookies,
    }

    try:
        dict["parsed"] = resp.parse()
    except Exception as e:
        logger.error(f"Failed to parse response, {e}")
        dict["parsed"] = {}

    try:
        dict["text"] = resp.text
    except Exception as e:
        logger.error(f"Failed to get response text, {e}")

    return dict
