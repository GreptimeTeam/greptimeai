import unittest
import uuid

from opentelemetry.trace import NoOpTracer

from greptimeai.langchain import _TraceTable


class TestTrace(unittest.TestCase):
    """
    test trace and span
    """

    tracer = NoOpTracer()

    def test_unique_traces(self):
        tables = _TraceTable()
        span_name, run_id = "chain", uuid.uuid1()
        origin_span = {
            "trace_id": "random_trace_id"
        }  # this dict is to mock span object
        origin_span = self.tracer.start_span("origin_span")

        tables.put_span(span_name, run_id, origin_span)

        self.assertIsNone(tables.get_id_span(uuid.uuid1()))
        self.assertIsNone(tables.get_name_span("not_exist_name", run_id))
        self.assertEqual(origin_span, tables.get_id_span(run_id))
        self.assertEqual(origin_span, tables.get_name_span(span_name, run_id))

        # pop will remove this span
        self.assertEqual(origin_span, tables.pop_span(span_name, run_id))

        # None after pop_span
        self.assertIsNone(tables.get_id_span(run_id))
        self.assertIsNone(tables.get_name_span(span_name, run_id))

    def test_duplicated_id_traces(self):
        tables = _TraceTable()
        chain_name, agent_name, run_id = "chain", "agent", uuid.uuid1()
        chain_span = self.tracer.start_span("chain_span")
        agent_span = self.tracer.start_span("agent_span")

        tables.put_span(chain_name, run_id, chain_span)
        tables.put_span(agent_name, run_id, agent_span)

        # check before pop
        self.assertEqual(agent_span, tables.get_id_span(run_id))
        self.assertEqual(chain_span, tables.get_name_span(chain_name, run_id))
        self.assertEqual(agent_span, tables.get_name_span(agent_name, run_id))

        # pop chain span
        self.assertEqual(chain_span, tables.pop_span(chain_name, run_id))
        self.assertEqual(agent_span, tables.get_id_span(run_id))
        self.assertIsNone(tables.get_name_span(chain_name, run_id))
        self.assertEqual(agent_span, tables.get_name_span(agent_name, run_id))

        # pop agent span
        self.assertEqual(agent_span, tables.pop_span(agent_name, run_id))
        self.assertIsNone(tables.get_id_span(run_id))
        self.assertIsNone(tables.get_name_span(chain_name, run_id))
        self.assertIsNone(tables.get_name_span(agent_name, run_id))
