import uuid

from opentelemetry.trace import NoOpTracer

from greptimeai.collection import _TraceContext, _TraceTable


class TestTrace:
    """
    test trace and span
    """

    tracer = NoOpTracer()

    def test_unique_traces(self):
        tables = _TraceTable()
        model_name = "gpt-3.5-turbo"
        span_name, run_id = "chain", uuid.uuid1()
        origin_span = self.tracer.start_span("origin_span")

        origin_context = _TraceContext(span_name, model_name, origin_span)
        tables.put_trace_context(run_id, origin_context)

        # run_id not exist
        assert tables.get_trace_context(uuid.uuid1()) is None

        # the specified run_id
        assert origin_context == tables.get_trace_context(run_id)

        # pop will remove this span
        assert origin_context == tables.pop_trace_context(run_id)

        # None after pop_span
        assert tables.get_trace_context(run_id) is None

    def test_duplicated_id_traces(self):
        tables = _TraceTable()
        model_name = "gpt-3.5-turbo"
        chain_name, agent_name, run_id = "chain", "agent", uuid.uuid1()
        chain_span = self.tracer.start_span("chain_span")
        agent_span = self.tracer.start_span("agent_span")

        chain_context = _TraceContext(chain_name, model_name, chain_span)
        tables.put_trace_context(run_id, chain_context)

        agent_context = _TraceContext(agent_name, model_name, agent_span)
        tables.put_trace_context(run_id, agent_context)

        # check before pop
        assert agent_context == tables.get_trace_context(
            run_id
        )  # if name not specified, the last context will be returned
        assert chain_context == tables.get_trace_context(run_id, chain_name)
        assert agent_context == tables.get_trace_context(run_id, agent_name)

        # pop chain span
        assert chain_context == tables.pop_trace_context(run_id, chain_name)
        assert tables.get_trace_context(run_id, chain_name) is None

        assert agent_context == tables.get_trace_context(run_id)
        assert agent_context == tables.get_trace_context(run_id, agent_name)

        # pop agent span
        assert agent_context == tables.pop_trace_context(run_id, agent_name)

        assert tables.get_trace_context(run_id, chain_name) is None
        assert tables.get_trace_context(run_id, agent_name) is None
        assert tables.get_trace_context(run_id) is None
