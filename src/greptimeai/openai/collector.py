import base64

from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import Counter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span

from greptimeai.openai.recorder import Recorder
from greptimeai.scope import (
    _NAME,
    _VERSION,
)


class Collector:
    def __init__(
        self,
        resource_name: str = None,
        host: str = None,
        database: str = None,
        username: str = None,
        password: str = None,
        scheme: str = None,
    ):
        self.resource_name = resource_name
        self.host = host
        self.database = database
        self.username = username
        self.password = password
        self.scheme = scheme

        self.tracer: trace.Tracer = None
        self.openai_error_count = None
        self.requests_duration_histogram = None
        self.completion_tokens_count: Counter = None
        self.prompt_tokens_count: Counter = None

    def setup(self):
        self._init_open_telemetry()
        recorder = Recorder()
        recorder.setup()

    def _init_open_telemetry(self):
        resource = Resource.create({SERVICE_NAME: self.resource_name})

        metrics_endpoint = f"https://{self.host}/v1/otlp/v1/metrics"
        trace_endpoint = f"https://{self.host}/v1/otlp/v1/traces"

        auth = f"{self.username}:{self.password}"
        b64_auth = base64.b64encode(auth.encode()).decode("ascii")
        greptime_headers = {
            "Authorization": f"Basic {b64_auth}",
            "x-greptime-db-name": self.database,
        }

        # metric
        metrics_exporter = OTLPMetricExporter(
            endpoint=metrics_endpoint,
            headers=greptime_headers,
            timeout=5,
        )
        metric_reader = PeriodicExportingMetricReader(metrics_exporter, 5000)
        metre_provider = MeterProvider(
            resource=resource, metric_readers=[metric_reader]
        )
        metrics.set_meter_provider(metre_provider)

        # trace
        span_processor = BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=trace_endpoint, headers=greptime_headers, timeout=5
            )
        )
        trace_provider = TracerProvider(resource=resource)
        trace_provider.add_span_processor(span_processor)
        trace.set_tracer_provider(trace_provider)

        # init class.tracer
        self.tracer = trace.get_tracer(
            instrumenting_module_name=_NAME,
            instrumenting_library_version=_VERSION,
        )

        meter = metrics.get_meter(name=_NAME, version=_VERSION)

        self.prompt_tokens_count = meter.create_counter(
            "openai_prompt_tokens",
            description="counts the amount of openai prompt token",
        )

        self.completion_tokens_count = meter.create_counter(
            "openai_completion_tokens",
            description="counts the amount of openai completion token",
        )

        self.openai_error_count = meter.create_counter(
            "openai_errors",
            description="counts the amount of openai errors",
        )

        self.requests_duration_histogram = meter.create_histogram(
            name="openai_request_duration_ms",
            description="duration of requests of openai in milli seconds",
            unit="ms",
        )

    def get_new_span(self, name: str) -> Span:
        return self.tracer.start_span(name)
