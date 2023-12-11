class LlmTrace(object):
    table_name = "llm_traces_preview_v01"

    trace_id: str
    span_id: str
    parent_span_id: str
    resource_attributes: str
    scope_name: str
    scope_version: str
    scope_attributes: str
    trace_state: str
    span_name: str
    span_kind: str
    span_status_code: str
    span_status_message: str
    span_attributes: str
    span_events: str
    span_links: str
    start: float
    end: float
    user_id: str
    model: str
    prompt_tokens: int
    prompt_cost: float
    completion_tokens: int
    completion_cost: float
    greptime_value: str
    greptime_timestamp: float


class LlmPromptToken(object):
    table_name = "llm_prompt_tokens"

    telemetry_sdk_language: str
    telemetry_sdk_name: str
    telemetry_sdk_version: str
    service_name: str
    span_name: str
    model: str
    greptime_value: str
    greptime_timestamp: float


class LlmPromptTokenCost(object):
    table_name = "llm_prompt_tokens_cost"

    telemetry_sdk_language: str
    telemetry_sdk_name: str
    telemetry_sdk_version: str
    service_name: str
    span_name: str
    model: str
    greptime_value: str
    greptime_timestamp: float


class LlmCompletionToken(object):
    table_name = "llm_completion_tokens"

    telemetry_sdk_language: str
    telemetry_sdk_name: str
    telemetry_sdk_version: str
    service_name: str
    span_name: str
    model: str
    greptime_value: str
    greptime_timestamp: float


class LlmCompletionTokenCost(object):
    table_name = "llm_completion_tokens_cost"

    telemetry_sdk_language: str
    telemetry_sdk_name: str
    telemetry_sdk_version: str
    service_name: str
    span_name: str
    model: str
    greptime_value: str
    greptime_timestamp: float


class Number(object):
    table_name = "number"

    number: int


class Tables(object):
    llm_trace = "llm_traces_preview_v01"
    llm_prompt_tokens = "llm_prompt_tokens"
    llm_prompt_tokens_cost = "llm_prompt_tokens_cost"
    llm_completion_tokens = "llm_completion_tokens"
    llm_completion_tokens_cost = "llm_completion_tokens_cost"
