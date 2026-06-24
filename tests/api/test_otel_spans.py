from fastapi.testclient import TestClient
from opentelemetry import trace as ot_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from pydantic import SecretStr

from aidial_admin_evaluation_metrics import app as app_module
from aidial_admin_evaluation_metrics.app_config import AppSettings


def test_metric_evaluate_creates_span(monkeypatch):
    # Setup in-memory tracer provider and exporter
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    ot_trace.set_tracer_provider(provider)

    # Ensure app module uses the test tracer and restore after test
    monkeypatch.setattr(
        app_module,
        "tracer",
        ot_trace.get_tracer(app_module.__name__),
        raising=False,
    )

    # Create app with minimal settings and call the evaluate endpoint
    app = app_module.create_app(
        AppSettings(dial_url="http://example", dial_api_key=SecretStr("test"))
    )
    with TestClient(app) as client:
        resp = client.post(
            "/evaluate",
            json={
                "metric_name": "exact_match",
                "config": {},
                "input": {
                    "actual": "test answer",
                    "ground_truth": "test answer",
                },
            },
        )
        assert resp.status_code == 200, resp.text

    # Inspect in-memory spans for our metric.evaluate span
    spans = exporter.get_finished_spans()
    eval_spans = [s for s in spans if s.name == "metric.evaluate"]
    assert (
        eval_spans
    ), f"No metric.evaluate spans found. Spans: {[s.name for s in spans]}"

    # Assert one of the metric.evaluate spans has the expected metric.name attribute
    expected_metric = "exact_match"
    assert any(
        (s.attributes or {}).get("metric.name") == expected_metric
        for s in eval_spans
    ), f"metric.evaluate span missing metric.name={expected_metric}; found attrs: {[s.attributes for s in eval_spans]}"
