"""Tests for ragas.tool_call_accuracy via the /evaluate endpoint.

ToolCallAccuracy is rule-based — no LLM calls are made, so no cassettes needed.
"""

import pytest

from aidial_admin_evaluation_metrics.metrics.ragas.tool_call_accuracy import (
    ToolCallAccuracyMetric,
)


def test_tool_call_accuracy_full_match(app_client):
    """All tool calls match exactly — score should be 1.0."""
    response = app_client.post(
        "/evaluate",
        json={
            "metric_name": "ragas.tool_call_accuracy",
            "config": {"strict_order": True},
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
    assert data["metric_name"] == "ragas.tool_call_accuracy"
    output = ToolCallAccuracyMetric.Output.model_validate(data["output"])
    assert output.score.type == "value"
    assert output.score.value == pytest.approx(1.0)


def test_tool_call_accuracy_wrong_args(app_client):
    """Tool name matches but argument value is wrong — score should be 0.0."""
    response = app_client.post(
        "/evaluate",
        json={
            "metric_name": "ragas.tool_call_accuracy",
            "config": {"strict_order": True},
            "input": {
                "tools_called": [
                    {"name": "get_weather", "args": {"location": "London"}},
                ],
                "reference_tool_calls": [
                    {"name": "get_weather", "args": {"location": "Paris"}},
                ],
            },
        },
    )
    assert response.status_code == 200
    output = ToolCallAccuracyMetric.Output.model_validate(
        response.json()["output"]
    )
    assert output.score.type == "value"
    assert output.score.value == pytest.approx(0.0)


def test_tool_call_accuracy_strict_order_wrong_sequence(app_client):
    """Reversed call order with strict_order=True — score 0.0.

    ToolCallAccuracy uses positional alignment: each call is matched against
    the reference at the same index. A fully reversed sequence has no
    position-aligned matches, so the score is 0.0.
    """
    response = app_client.post(
        "/evaluate",
        json={
            "metric_name": "ragas.tool_call_accuracy",
            "config": {"strict_order": True},
            "input": {
                "tools_called": [
                    {"name": "get_weather", "args": {"location": "Berlin"}},
                    {"name": "get_weather", "args": {"location": "Paris"}},
                ],
                "reference_tool_calls": [
                    {"name": "get_weather", "args": {"location": "Paris"}},
                    {"name": "get_weather", "args": {"location": "Berlin"}},
                ],
            },
        },
    )
    assert response.status_code == 200
    output = ToolCallAccuracyMetric.Output.model_validate(
        response.json()["output"]
    )
    assert output.score.type == "value"
    assert output.score.value == pytest.approx(0.0)


def test_tool_call_accuracy_non_strict_order(app_client):
    """Reversed call order with strict_order=False — order ignored, score 1.0."""
    response = app_client.post(
        "/evaluate",
        json={
            "metric_name": "ragas.tool_call_accuracy",
            "config": {"strict_order": False},
            "input": {
                "tools_called": [
                    {"name": "get_weather", "args": {"location": "Berlin"}},
                    {"name": "get_weather", "args": {"location": "Paris"}},
                ],
                "reference_tool_calls": [
                    {"name": "get_weather", "args": {"location": "Paris"}},
                    {"name": "get_weather", "args": {"location": "Berlin"}},
                ],
            },
        },
    )
    assert response.status_code == 200
    output = ToolCallAccuracyMetric.Output.model_validate(
        response.json()["output"]
    )
    assert output.score.type == "value"
    assert output.score.value == pytest.approx(1.0)


def test_tool_call_accuracy_wrong_tool_name(app_client):
    """Tool name does not match — score should be 0.0."""
    response = app_client.post(
        "/evaluate",
        json={
            "metric_name": "ragas.tool_call_accuracy",
            "config": {"strict_order": True},
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
    output = ToolCallAccuracyMetric.Output.model_validate(
        response.json()["output"]
    )
    assert output.score.type == "value"
    assert output.score.value == pytest.approx(0.0)


def test_tool_call_accuracy_partial_match(app_client):
    """First call matches, second has wrong args — one of two positions aligns, score 0.5."""
    response = app_client.post(
        "/evaluate",
        json={
            "metric_name": "ragas.tool_call_accuracy",
            "config": {"strict_order": True},
            "input": {
                "tools_called": [
                    {"name": "get_weather", "args": {"location": "Paris"}},
                    {"name": "get_weather", "args": {"location": "London"}},
                ],
                "reference_tool_calls": [
                    {"name": "get_weather", "args": {"location": "Paris"}},
                    {"name": "get_weather", "args": {"location": "Berlin"}},
                ],
            },
        },
    )
    assert response.status_code == 200
    output = ToolCallAccuracyMetric.Output.model_validate(
        response.json()["output"]
    )
    assert output.score.type == "value"
    assert output.score.value == pytest.approx(0.5)


# NOTE: Empty tools_called is intentionally not tested for ToolCallAccuracy.
# ragas.ToolCallAccuracy emits a UserWarning when no tool calls are found,
# which pytest promotes to an exception via filterwarnings = ["error"].
