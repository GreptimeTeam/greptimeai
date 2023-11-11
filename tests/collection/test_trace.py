import uuid

from opentelemetry.trace import NoOpTracer

from greptimeai.collection import _TraceTable


class TestTrace:
    """
    test trace and span
    """

    tracer = NoOpTracer()

    def test_unique_traces(self):
        tables = _TraceTable()
        model_name = "gpt-3.5-turbo"
        span_name, run_id = "chain", uuid.uuid1()
        origin_span = {
            "trace_id": "random_trace_id"
        }  # this dict is to mock span object
        origin_span = self.tracer.start_span("origin_span")

        tables.put_span(span_name, run_id, origin_span, model_name=model_name)

        # run_id not exist
        assert tables.get_id_span(uuid.uuid1()) is None
        assert tables.get_name_span("not_exist_name", run_id) is None
        assert tables.get_id_model(uuid.uuid1()) is None

        # the specified run_id
        assert origin_span == tables.get_id_span(run_id)
        assert origin_span == tables.get_name_span(span_name, run_id)
        assert model_name == tables.get_id_model(run_id)

        # pop will remove this span
        assert origin_span == tables.pop_span(span_name, run_id)

        # None after pop_span
        assert tables.get_id_span(run_id) is None
        assert tables.get_name_span(span_name, run_id) is None
        assert tables.get_id_model(run_id) is None

    def test_duplicated_id_traces(self):
        tables = _TraceTable()
        model_name = "gpt-3.5-turbo"
        chain_name, agent_name, run_id = "chain", "agent", uuid.uuid1()
        chain_span = self.tracer.start_span("chain_span")
        agent_span = self.tracer.start_span("agent_span")

        tables.put_span(chain_name, run_id, chain_span, model_name)
        # empty model won't override the exist one
        tables.put_span(agent_name, run_id, agent_span, "")

        # check before pop
        assert agent_span == tables.get_id_span(run_id)
        assert chain_span == tables.get_name_span(chain_name, run_id)
        assert agent_span == tables.get_name_span(agent_name, run_id)
        assert model_name == tables.get_id_model(run_id)

        # pop chain span
        assert chain_span == tables.pop_span(chain_name, run_id)
        assert agent_span == tables.get_id_span(run_id)
        assert tables.get_name_span(chain_name, run_id) is None
        assert agent_span == tables.get_name_span(agent_name, run_id)
        assert model_name == tables.get_id_model(run_id)

        # pop agent span
        assert agent_span == tables.pop_span(agent_name, run_id)
        assert tables.get_id_span(run_id) is None
        assert tables.get_name_span(chain_name, run_id) is None
        assert tables.get_name_span(agent_name, run_id) is None
        assert tables.get_id_model(run_id) is None
