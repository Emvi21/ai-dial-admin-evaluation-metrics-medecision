"""Metric registration system."""

import logging
from typing import Dict, List

from pydantic import BaseModel

from aidial_admin_evaluation_metrics.metric import MetricsDescription
from aidial_admin_evaluation_metrics.metrics.common.base_metric import (
    BaseMetric,
    MetricExample,
)

logger = logging.getLogger(__name__)


class MetricsRegistry:
    """Registry for pre-built evaluation metric instances.

    Each registered instance carries its own ``config_cls`` (the Pydantic model
    used for request validation) which was set at construction time.  This means:
    - The ``/metrics`` JSON schema exposes the correct ``enum`` for UI dropdowns.
    - ``/evaluate`` requests with an unlisted model are rejected with HTTP 422.
    """

    def __init__(self, metrics: List[BaseMetric] | None = None):
        self._instances: Dict[str, BaseMetric] = {}
        self._descriptions: List[MetricsDescription] = []
        self._examples: Dict[str, List[MetricExample]] = {}

        for instance in metrics or []:
            self._register(instance)

    def _register(self, instance: BaseMetric) -> None:
        """Register a pre-built metric instance.

        Example validation uses the unpatched class-level ``Config`` so that
        hardcoded examples are not restricted by the runtime model allowlist.

        Raises:
            ValueError: If a metric with the same name is already registered
            ValidationError: If any example fails schema validation
        """
        metric_name = instance.name
        if metric_name in self._instances:
            raise ValueError(
                f"Metric with name '{metric_name}' is already registered. "
                f"Cannot register {type(instance).__name__}."
            )

        validated_examples = self._validate_examples(instance)

        self._instances[metric_name] = instance
        self._descriptions.append(
            MetricsDescription(
                name=instance.name,
                display_name=instance.display_name or instance.name,
                description=instance.description,
                config_schema=instance.config_cls.model_json_schema(),
                input_schema=instance.Input.model_json_schema(),
                output_schema=instance.Output.model_json_schema(),
            )
        )

        if validated_examples:
            self._examples[metric_name] = validated_examples

    def get_metric_instance(self, name: str) -> BaseMetric:
        """Return the pre-built metric instance for a name.

        Raises:
            KeyError: If metric name is not registered
        """
        return self._instances[name]

    def get_config_cls(self, name: str) -> type[BaseModel]:
        """Return the (potentially patched) Config class for a metric.

        Use this for request validation so the model allowlist is enforced.

        Raises:
            KeyError: If metric name is not registered
        """
        return self._instances[name].config_cls

    def get_all_metrics(self) -> Dict[str, BaseMetric]:
        """Return all registered metric instances keyed by name."""
        return dict(self._instances)

    def get_metric(self, name: str) -> BaseMetric:
        """Return a registered metric instance by name.

        Raises:
            KeyError: If metric name is not registered
        """
        return self._instances[name]

    def get_all_descriptions(self) -> List[MetricsDescription]:
        return list(self._descriptions)

    def get_metric_names(self) -> set[str]:
        return set(self._instances.keys())

    def get_all_examples(self) -> Dict[str, List[MetricExample]]:
        return dict(self._examples)

    def __len__(self) -> int:
        return len(self._instances)

    def _validate_examples(self, instance: BaseMetric) -> List[MetricExample]:
        """Validate that examples conform to the metric's original schemas.

        Uses the unpatched class-level ``Config`` / ``Input`` / ``Output`` so
        that hardcoded examples (which use the original types) are not
        restricted by the runtime model allowlist stored in ``instance.config_cls``.

        Accepts both typed Pydantic model instances and plain dicts for
        input, expected_output, and config.
        """
        validated = []
        for example in instance.examples:
            config = example.config
            inp = example.input
            expected_output = example.expected_output

            if isinstance(config, BaseModel):
                if not isinstance(config, instance.Config):
                    raise ValueError(
                        f"Example config is an instance of {type(config).__name__}, "
                        f"expected {instance.Config.__name__}"
                    )
            else:
                instance.Config.model_validate(config or {})

            if isinstance(inp, BaseModel):
                if not isinstance(inp, instance.Input):
                    raise ValueError(
                        f"Example input is an instance of {type(inp).__name__}, "
                        f"expected {instance.Input.__name__}"
                    )
            else:
                instance.Input.model_validate(inp)

            if isinstance(expected_output, BaseModel):
                if not isinstance(expected_output, instance.Output):
                    raise ValueError(
                        f"Example expected_output is an instance of {type(expected_output).__name__}, "
                        f"expected {instance.Output.__name__}"
                    )
            else:
                instance.Output.model_validate(expected_output)

            validated.append(example)
        return validated
