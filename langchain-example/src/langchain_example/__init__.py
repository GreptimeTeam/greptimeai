import base64

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

resource = Resource.create({SERVICE_NAME: "greptime-llm-langchain-example"})

host = "7rxmkk3xkkms.test-ap-southeast-1.aws.greptime.cloud"
db = "ix352urea2qaotlp_traces-public"
username, password = "YUKplETNrYPXz1eferrQjUe8", "Fcqwp8fqGafKKW3F2OAoJLgh"

metrics_endpoint = f"https://{host}/v1/otlp/v1/metrics"
trace_endpoint = f"https://{host}/v1/otlp/v1/traces"

auth = f"{username}:{password}"
b64_auth = base64.b64encode(auth.encode()).decode("ascii")
greptime_headers = {"Authorization": f"Basic {b64_auth}", "x-greptime-db-name": db}


metrics_exporter = OTLPMetricExporter(
    endpoint=metrics_endpoint,
    headers=greptime_headers,
    timeout=5,
)
metric_reader = PeriodicExportingMetricReader(metrics_exporter, 5000)
metre_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(metre_provider)

trace_provider = TracerProvider(resource=resource)
span_processor = BatchSpanProcessor(
    OTLPSpanExporter(
        endpoint=trace_endpoint,
        headers=greptime_headers,
        timeout=5,
    )
)
trace_provider.add_span_processor(span_processor)
trace.set_tracer_provider(trace_provider)
