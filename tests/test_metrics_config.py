"""Tests for /metrics endpoint model config propagation.

Verifies that the config_schema returned by GET /metrics correctly reflects
the supported_models and default_model settings for each metric group.
"""

import os
from unittest.mock import patch

from fastapi.testclient import TestClient

from aidial_admin_evaluation_metrics.app import create_app
from aidial_admin_evaluation_metrics.app_config import (
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    AppSettings,
)

BASE_ENV = {"DIAL_URL": "https://dial.test.example.com"}


def _make_client(env: dict) -> TestClient:
    with patch.dict(os.environ, env, clear=True):
        app = create_app(app_config=AppSettings())
    return TestClient(app)


def _get_metrics_by_name(client: TestClient) -> dict[str, dict]:
    response = client.get("/metrics")
    assert response.status_code == 200
    return {m["name"]: m for m in response.json()["metrics"]}


def _model_prop(metric: dict) -> dict | None:
    """Return the model property from a metric's config_schema, or None if absent."""
    return (metric["config_schema"].get("properties") or {}).get("model")


def test_default_models_in_config_schema():
    """Without overrides, both LLM metrics expose SUPPORTED_MODELS enum and DEFAULT_MODEL default."""
    with _make_client(BASE_ENV) as client:
        metrics = _get_metrics_by_name(client)

    for name in ("deepeval.answer_relevancy", "aidial_rag_eval.nli"):
        prop = _model_prop(metrics[name])
        assert (
            prop is not None
        ), f"{name}: model field missing from config_schema"
        assert prop["enum"] == SUPPORTED_MODELS
        assert prop["default"] == DEFAULT_MODEL


def test_custom_deepeval_models_in_config_schema():
    """EVAL__METRICS__DEEPEVAL__* env vars are reflected in deepeval metric config_schema."""
    env = {
        **BASE_ENV,
        "EVAL__METRICS__DEEPEVAL__SUPPORTED_MODELS": '["model-x", "model-y"]',
        "EVAL__METRICS__DEEPEVAL__DEFAULT_MODEL": "model-y",
    }
    with _make_client(env) as client:
        metrics = _get_metrics_by_name(client)

    prop = _model_prop(metrics["deepeval.answer_relevancy"])
    assert prop is not None
    assert prop["enum"] == ["model-x", "model-y"]
    assert prop["default"] == "model-y"

    # aidial_rag_eval is unaffected — still uses defaults
    prop = _model_prop(metrics["aidial_rag_eval.nli"])
    assert prop is not None
    assert prop["enum"] == SUPPORTED_MODELS
    assert prop["default"] == DEFAULT_MODEL


def test_custom_aidial_rag_eval_models_in_config_schema():
    """EVAL__METRICS__AIDIAL_RAG_EVAL__* env vars are reflected in aidial_rag_eval metric config_schema."""
    env = {
        **BASE_ENV,
        "EVAL__METRICS__AIDIAL_RAG_EVAL__SUPPORTED_MODELS": '["model-a", "model-b"]',
        "EVAL__METRICS__AIDIAL_RAG_EVAL__DEFAULT_MODEL": "model-a",
    }
    with _make_client(env) as client:
        metrics = _get_metrics_by_name(client)

    prop = _model_prop(metrics["aidial_rag_eval.nli"])
    assert prop is not None
    assert prop["enum"] == ["model-a", "model-b"]
    assert prop["default"] == "model-a"

    # deepeval is unaffected — still uses defaults
    prop = _model_prop(metrics["deepeval.answer_relevancy"])
    assert prop is not None
    assert prop["enum"] == SUPPORTED_MODELS
    assert prop["default"] == DEFAULT_MODEL


def test_common_settings_propagate_to_both_groups_in_config_schema():
    """EVAL__METRICS__COMMON__* is reflected in config_schema for both LLM metric groups."""
    env = {
        **BASE_ENV,
        "EVAL__METRICS__COMMON__SUPPORTED_MODELS": '["model-g1", "model-g2"]',
        "EVAL__METRICS__COMMON__DEFAULT_MODEL": "model-g1",
    }
    with _make_client(env) as client:
        metrics = _get_metrics_by_name(client)

    for name in ("deepeval.answer_relevancy", "aidial_rag_eval.nli"):
        prop = _model_prop(metrics[name])
        assert (
            prop is not None
        ), f"{name}: model field missing from config_schema"
        assert prop["enum"] == ["model-g1", "model-g2"]
        assert prop["default"] == "model-g1"


def test_group_override_takes_precedence_over_common_in_config_schema():
    """Group-specific env vars override EVAL__METRICS__COMMON__* in config_schema."""
    env = {
        **BASE_ENV,
        "EVAL__METRICS__COMMON__SUPPORTED_MODELS": '["model-g1", "model-g2"]',
        "EVAL__METRICS__COMMON__DEFAULT_MODEL": "model-g1",
        "EVAL__METRICS__DEEPEVAL__SUPPORTED_MODELS": '["model-d1", "model-d2"]',
        "EVAL__METRICS__DEEPEVAL__DEFAULT_MODEL": "model-d2",
    }
    with _make_client(env) as client:
        metrics = _get_metrics_by_name(client)

    # deepeval uses its own group values
    prop = _model_prop(metrics["deepeval.answer_relevancy"])
    assert prop is not None
    assert prop["enum"] == ["model-d1", "model-d2"]
    assert prop["default"] == "model-d2"

    # aidial_rag_eval falls back to global
    prop = _model_prop(metrics["aidial_rag_eval.nli"])
    assert prop is not None
    assert prop["enum"] == ["model-g1", "model-g2"]
    assert prop["default"] == "model-g1"


def test_non_llm_metrics_have_no_model_field_in_config_schema():
    """Simple metrics and retrieval have no model field in config_schema."""
    with _make_client(BASE_ENV) as client:
        metrics = _get_metrics_by_name(client)

    for name in ("exact_match", "regex_match", "aidial_rag_eval.retrieval"):
        assert (
            _model_prop(metrics[name]) is None
        ), f"{name}: unexpected model field in config_schema"
