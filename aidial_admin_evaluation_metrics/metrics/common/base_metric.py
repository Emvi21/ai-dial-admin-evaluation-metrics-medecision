"""Base metric protocol and utilities."""

from typing import Any, Optional

import anyio
from pydantic import BaseModel, ConfigDict

from aidial_admin_evaluation_metrics.metric import MetricsDescription


class MetricExample(BaseModel):
    """Example for a metric with typed inputs and expected outputs.

    Required fields: input, expected_output
    Optional fields: name, description, config
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Optional[str] = (
        None  # If omitted, auto-generated from metric name at serialization time
    )
    description: Optional[str] = None  # What this example demonstrates
    input: Any  # Metric Input instance or plain dict
    expected_output: Any  # Metric Output instance or plain dict
    config: Optional[Any] = (
        None  # Metric Config instance or plain dict; None means use defaults
    )


class BaseMetric:
    """Base class for all evaluation metrics.

    All metrics must inherit from this class. The base class provides a default
    constructor that accepts an llm_factory parameter for dependency injection.

    LLM-based metrics should override __init__ to store and use the factory.
    Non-LLM metrics can use the default implementation (llm_factory is ignored).

    For LLM-based metrics, consider inheriting from BaseLLMMetric which provides
    additional convenience methods for LLM creation.

    Subclasses must define:
        - name: str - The metric's unique identifier
        - description: str - A brief description of what the metric evaluates
        - examples: list[MetricExample] - Example inputs and outputs
        - Config: Pydantic model for configuration
        - Input: Pydantic model for input data
        - Output: Pydantic model for output data
        - _evaluate(config, input) -> Output - The evaluation logic
    """

    name: str
    display_name: str = ""
    description: str = ""
    examples: list[MetricExample] = []

    Config: type[BaseModel]
    Input: type[BaseModel]
    Output: type[BaseModel]

    def __init__(self):
        self.config_cls: type[BaseModel] = type(self).Config

    def _evaluate(self, config, input) -> Any:
        """Evaluate the metric with given configuration and input.

        Args:
            config: Configuration instance (self.Config)
            input: Input data instance (self.Input)

        Returns:
            Output instance (self.Output)
        """
        raise NotImplementedError(
            "Subclasses must implement _evaluate() or evaluate_async()"
        )

    async def evaluate_async(self, config, input) -> Any:
        """Async variant of evaluate().

        Default implementation offloads the synchronous evaluate() to a thread
        pool worker via asyncio.to_thread(), keeping the event loop free.
        Metrics with a native async implementation (e.g. DeepEval-backed metrics)
        should override this method.
        """
        return await anyio.to_thread.run_sync(self._evaluate, config, input)


def get_metric_description(metric_cls: type[BaseMetric]) -> MetricsDescription:
    return MetricsDescription(
        name=metric_cls.name,
        display_name=metric_cls.display_name or metric_cls.name,
        description=metric_cls.description,
        config_schema=metric_cls.Config.model_json_schema(),
        input_schema=metric_cls.Input.model_json_schema(),
        output_schema=metric_cls.Output.model_json_schema(),
    )
