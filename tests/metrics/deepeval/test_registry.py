"""Tests for deepeval metric registry presence."""

from pydantic import SecretStr

from aidial_admin_evaluation_metrics.app_config import DialConfig
from aidial_admin_evaluation_metrics.dial.llm_client import create_dial_factory
from aidial_admin_evaluation_metrics.metrics import create_metrics_registry

registry = create_metrics_registry(
    create_dial_factory(
        DialConfig(dial_url="http://test", dial_api_key=SecretStr("test"))
    )
)


def test_all_deepeval_metrics_in_registry():
    names = registry.get_metric_names()
    for name in [
        "deepeval.answer_relevancy",
        "deepeval.faithfulness",
        "deepeval.contextual_precision",
        "deepeval.contextual_recall",
        "deepeval.contextual_relevancy",
        "deepeval.g_eval",
    ]:
        assert name in names, f"{name} not found in registry"
