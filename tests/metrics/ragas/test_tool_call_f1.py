"""Tests for ragas.tool_call_f1 via the /evaluate endpoint.

ToolCallF1 is rule-based — no LLM calls are made, so no cassettes needed.
"""

import pytest

from aidial_admin_evaluation_metrics.metrics.ragas.tool_call_f1 import (
    ToolCallF1Metric,
)


def test_tool_call_f1_full_match(app_client):
    """All tool calls match — F1 should be 1.0."""
    response = app_client.post(
        "/evaluate",
        json={
            "metric_name": "ragas.tool_call_f1",
            "config": {},
            "input": {
                "tools_called": [
                    {"name": "get_weather", "args": {"location": "Paris"}},
                    {"name": "get_weather", "args": {"location": "Berlin"}},
                ],
                "reference_tool_calls": [
                    {"name": "get_weather", "args": {"location": "Paris"}},
                    {"name": "get_weather", "args": {"location": "Berlin"}},
                ],
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["metric_name"] == "ragas.tool_call_f1"
    output = ToolCallF1Metric.Output.model_validate(data["output"])
    assert output.score.type == "value"
    assert output.score.value == pytest.approx(1.0)


def test_tool_call_f1_partial_match(app_client):
    """One extra call (1 TP, 1 FP, 0 FN) — F1 = 2/3 ≈ 0.6667."""
    response = app_client.post(
        "/evaluate",
        json={
            "metric_name": "ragas.tool_call_f1",
            "config": {},
            "input": {
                "tools_called": [
                    {"name": "get_weather", "args": {"location": "Paris"}},
                    {"name": "get_calendar", "args": {}},
                ],
                "reference_tool_calls": [
                    {"name": "get_weather", "args": {"location": "Paris"}},
                ],
            },
        },
    )
    assert response.status_code == 200
    output = ToolCallF1Metric.Output.model_validate(response.json()["output"])
    assert output.score.type == "value"
    assert output.score.value == pytest.approx(0.6667, abs=1e-3)


def test_tool_call_f1_no_match(app_client):
    """Completely wrong tool called — F1 should be 0.0."""
    response = app_client.post(
        "/evaluate",
        json={
            "metric_name": "ragas.tool_call_f1",
            "config": {},
            "input": {
                "tools_called": [
                    {"name": "send_email", "args": {"to": "user@example.com"}},
                ],
                "reference_tool_calls": [
                    {"name": "get_weather", "args": {"location": "Paris"}},
                ],
            },
        },
    )
    assert response.status_code == 200
    output = ToolCallF1Metric.Output.model_validate(response.json()["output"])
    assert output.score.type == "value"
    assert output.score.value == pytest.approx(0.0)


def test_tool_call_f1_no_tool_calls(app_client):
    """Agent made no tool calls but reference expects one — F1 should be 0.0."""
    response = app_client.post(
        "/evaluate",
        json={
            "metric_name": "ragas.tool_call_f1",
            "config": {},
            "input": {
                "tools_called": [],
                "reference_tool_calls": [
                    {"name": "get_weather", "args": {"location": "Paris"}},
                ],
            },
        },
    )
    assert response.status_code == 200
    output = ToolCallF1Metric.Output.model_validate(response.json()["output"])
    assert output.score.type == "value"
    assert output.score.value == pytest.approx(0.0)
