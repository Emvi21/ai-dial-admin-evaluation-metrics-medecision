"""Tests for MetricsSettings supported_models and default_model resolution."""

import logging
from unittest.mock import patch

from aidial_admin_evaluation_metrics.app_config import (
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    AppSettings,
)

BASE_ENV = {"DIAL_URL": "https://dial.example.com"}


def test_defaults_applied_to_both_groups():
    """When no model env vars are set, both groups get SUPPORTED_MODELS / DEFAULT_MODEL."""
    with patch.dict("os.environ", BASE_ENV, clear=True):
        settings = AppSettings()

    deepeval = settings.metrics.deepeval
    rag_eval = settings.metrics.aidial_rag_eval

    assert deepeval.supported_models == SUPPORTED_MODELS
    assert deepeval.default_model == DEFAULT_MODEL
    assert rag_eval.supported_models == SUPPORTED_MODELS
    assert rag_eval.default_model == DEFAULT_MODEL


def test_common_settings_propagate_to_both_groups():
    """EVAL__METRICS__COMMON__* is applied to all groups that have no override."""
    with patch.dict(
        "os.environ",
        {
            **BASE_ENV,
            "EVAL__METRICS__COMMON__SUPPORTED_MODELS": '["model-a", "model-b"]',
            "EVAL__METRICS__COMMON__DEFAULT_MODEL": "model-a",
        },
        clear=True,
    ):
        settings = AppSettings()

    assert settings.metrics.deepeval.supported_models == ["model-a", "model-b"]
    assert settings.metrics.deepeval.default_model == "model-a"
    assert settings.metrics.aidial_rag_eval.supported_models == [
        "model-a",
        "model-b",
    ]
    assert settings.metrics.aidial_rag_eval.default_model == "model-a"


def test_group_settings_override_common():
    """Group-specific env vars take precedence over common settings."""
    with patch.dict(
        "os.environ",
        {
            **BASE_ENV,
            "EVAL__METRICS__COMMON__SUPPORTED_MODELS": '["model-a", "model-b"]',
            "EVAL__METRICS__COMMON__DEFAULT_MODEL": "model-a",
            "EVAL__METRICS__DEEPEVAL__SUPPORTED_MODELS": '["model-c"]',
            "EVAL__METRICS__DEEPEVAL__DEFAULT_MODEL": "model-c",
        },
        clear=True,
    ):
        settings = AppSettings()

    # deepeval got its own values
    assert settings.metrics.deepeval.supported_models == ["model-c"]
    assert settings.metrics.deepeval.default_model == "model-c"
    # aidial_rag_eval inherited from global
    assert settings.metrics.aidial_rag_eval.supported_models == [
        "model-a",
        "model-b",
    ]
    assert settings.metrics.aidial_rag_eval.default_model == "model-a"


def test_default_model_not_in_supported_falls_back_to_first(caplog):
    """When default_model is not in supported_models, the first model is used and a warning is logged."""
    with patch.dict(
        "os.environ",
        {
            **BASE_ENV,
            "EVAL__METRICS__DEEPEVAL__SUPPORTED_MODELS": '["model-x", "model-y"]',
            "EVAL__METRICS__DEEPEVAL__DEFAULT_MODEL": "model-z",
        },
        clear=True,
    ):
        with caplog.at_level(logging.WARNING):
            settings = AppSettings()

    assert settings.metrics.deepeval.default_model == "model-x"
    assert "model-z" in caplog.text
    assert "model-x" in caplog.text or "model-x" in str(
        settings.metrics.deepeval.supported_models
    )


def test_global_default_model_not_in_global_supported_falls_back(caplog):
    """Global default_model outside global supported_models affects all inheriting groups."""
    with patch.dict(
        "os.environ",
        {
            **BASE_ENV,
            "EVAL__METRICS__COMMON__SUPPORTED_MODELS": '["model-a"]',
            "EVAL__METRICS__COMMON__DEFAULT_MODEL": "model-z",
        },
        clear=True,
    ):
        with caplog.at_level(logging.WARNING):
            settings = AppSettings()

    assert settings.metrics.deepeval.default_model == "model-a"
    assert settings.metrics.aidial_rag_eval.default_model == "model-a"
    assert "model-z" in caplog.text
