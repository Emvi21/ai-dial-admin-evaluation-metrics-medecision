"""Tests for simple metrics."""

from pydantic import SecretStr

from aidial_admin_evaluation_metrics.app_config import DialConfig
from aidial_admin_evaluation_metrics.dial.llm_client import create_dial_factory
from aidial_admin_evaluation_metrics.metrics import create_metrics_registry

registry = create_metrics_registry(
    create_dial_factory(
        DialConfig(dial_url="http://test", dial_api_key=SecretStr("test"))
    )
)


def test_exact_match_returns_1_for_identical_strings():
    metric = registry.get_metric_instance("exact_match")
    assert metric is not None, "exact_match metric should be registered"
    config = metric.Config()
    input_data = metric.Input(actual="test", ground_truth="test")
    result = metric._evaluate(config, input_data)

    assert result.exact_match.value == 1.0


def test_exact_match_returns_0_for_different_strings():
    metric = registry.get_metric_instance("exact_match")
    assert metric is not None, "exact_match metric should be registered"
    config = metric.Config()
    input_data = metric.Input(actual="wrong", ground_truth="test")
    result = metric._evaluate(config, input_data)

    assert result.exact_match.value == 0.0


def test_exact_match_is_case_sensitive():
    metric = registry.get_metric_instance("exact_match")
    assert metric is not None, "exact_match metric should be registered"
    config = metric.Config()
    input_data = metric.Input(actual="Test", ground_truth="test")
    result = metric._evaluate(config, input_data)

    assert result.exact_match.value == 0.0


def test_regex_match_returns_1_for_matching_pattern():
    metric = registry.get_metric_instance("regex_match")
    assert metric is not None, "regex_match metric should be registered"
    config = metric.Config(pattern=r"\d+")
    input_data = metric.Input(actual="123")
    result = metric._evaluate(config, input_data)

    assert result.regex_match.value == 1.0


def test_regex_match_returns_0_for_non_matching_pattern():
    metric = registry.get_metric_instance("regex_match")
    assert metric is not None, "regex_match metric should be registered"
    config = metric.Config(pattern=r"\d+")
    input_data = metric.Input(actual="abc")
    result = metric._evaluate(config, input_data)

    assert result.regex_match.value == 0.0


def test_regex_match_returns_error_for_invalid_pattern():
    metric = registry.get_metric_instance("regex_match")
    assert metric is not None, "regex_match metric should be registered"
    config = metric.Config(pattern="[invalid")
    input_data = metric.Input(actual="test")
    result = metric._evaluate(config, input_data)

    assert "Invalid regular expression pattern" in result.regex_match.message


def test_regex_match_uses_fullmatch():
    """Test that regex match uses fullmatch (entire string must match)."""
    metric = registry.get_metric_instance("regex_match")
    assert metric is not None, "regex_match metric should be registered"
    config = metric.Config(pattern=r"\d+")
    # Partial match should fail with fullmatch
    input_data = metric.Input(actual="123abc")
    result = metric._evaluate(config, input_data)

    assert result.regex_match.value == 0.0
