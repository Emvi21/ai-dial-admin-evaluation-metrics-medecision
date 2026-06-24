from typing import Any, Dict

from pydantic import BaseModel, Field

from aidial_admin_evaluation_metrics.metrics.common.registry import (
    MetricsRegistry,
)
from aidial_admin_evaluation_metrics.metrics.common.types import MetricResults


class EvaluationRequest(BaseModel):
    metric_name: str
    config: Dict[str, Any] = Field(default_factory=dict)
    input: Dict[str, Any]


def _serialize(v: Any) -> Any:
    """Serialize a value to a JSON-compatible dict.

    Accepts BaseModel instances or plain dicts.
    Returns a JSON-encoded dict for BaseModel instances,
    an empty dict for None, or the value as-is otherwise.
    """
    if isinstance(v, BaseModel):
        return v.model_dump(mode="json")
    return v if v is not None else {}


def build_request_examples(registry: MetricsRegistry) -> dict:
    """Build OpenAPI examples from all registered metrics.

    Returns:
        Dictionary of examples for the request body in OpenAPI format.
    """
    examples = {}
    for metric_name, metric_examples in registry.get_all_examples().items():
        n = len(metric_examples)
        for i, example in enumerate(metric_examples):
            key = example.name or (
                metric_name if n == 1 else f"{metric_name} #{i + 1}"
            )
            examples[key] = {
                "value": {
                    "metric_name": metric_name,
                    "config": _serialize(example.config),
                    "input": _serialize(example.input),
                }
            }
    return examples


def build_response_examples(registry: MetricsRegistry) -> list:
    """Generate response examples from registered metrics.

    Returns:
        List of example responses for the EvaluationResponse schema.
    """
    examples = []
    for metric_name, metric_examples in registry.get_all_examples().items():
        for example in metric_examples:
            examples.append(
                {
                    "metric_name": metric_name,
                    "output": _serialize(example.expected_output),
                }
            )
    return examples


class EvaluationResponse(BaseModel):
    metric_name: str
    output: MetricResults

    model_config = {
        "json_schema_extra": {
            # The actual examples are set dynamically in the app lifespan function,
            # because they require access to the registry which is only created at app startup.
            "examples": [],
        }
    }
