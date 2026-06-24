import pytest
from pydantic import SecretStr

from aidial_admin_evaluation_metrics.api.evaluate import (
    EvaluationResponse,
    build_response_examples,
)
from aidial_admin_evaluation_metrics.app_config import DialConfig
from aidial_admin_evaluation_metrics.dial.llm_client import create_dial_factory
from aidial_admin_evaluation_metrics.metrics import create_metrics_registry

registry = create_metrics_registry(
    create_dial_factory(
        DialConfig(dial_url="http://test", dial_api_key=SecretStr("test"))
    )
)


@pytest.mark.parametrize(
    "example",
    build_response_examples(registry),
)
def test_validate_examples_with_response(example):
    EvaluationResponse.model_validate(example)


@pytest.mark.parametrize(
    "example",
    build_response_examples(registry),
)
def test_validate_examples_with_metric_output(example):
    metric_name = example.get("metric_name")
    metric_cls = registry.get_metric(metric_name)
    metric_cls.Output.model_validate(example["output"])
