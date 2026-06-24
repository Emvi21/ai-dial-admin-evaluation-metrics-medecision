"""Base class for LLM-based metrics."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, create_model

from aidial_admin_evaluation_metrics.app_config import (
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    CommonGroupSettings,
)
from aidial_admin_evaluation_metrics.dial.llm_client import DialFactory
from aidial_admin_evaluation_metrics.metrics.common.base_metric import (
    BaseMetric,
)


class BaseLLMMetric(BaseMetric):
    """Base class for metrics that use LLM for evaluation.

    This class provides common functionality for LLM-based metrics,
    including storage of the LLM factory and runtime Config patching
    (restricting the ``model`` field to the configured allowed models).

    Example:
        >>> class MyMetric(BaseLLMMetric):
        ...     def __init__(self, dial_factory, settings):
        ...         super().__init__(dial_factory, settings)
        ...
        ...     def _evaluate(self, config, input):
        ...         llm = self._dial_factory.create_llm(config.model)
        ...         # Use llm for evaluation
    """

    @classmethod
    def create_config_cls(
        cls, settings: CommonGroupSettings
    ) -> type[BaseModel]:
        """Return a Config subclass with the model field restricted to the allowed models.

        Builds ``Literal[*allowed_models]`` at runtime so the JSON schema
        exposes an ``enum`` for UI dropdowns and Pydantic enforces the
        allowed values without a custom validator.
        """
        description = (
            cls.Config.model_fields["model"].description
            or "The LLM deployment name for evaluation."
        )
        allowed_models = settings.supported_models or SUPPORTED_MODELS
        default_model = settings.default_model or DEFAULT_MODEL
        # type: ignore[valid-type] — runtime Literal from a dynamic list is valid
        AllowedModels = Literal[tuple(allowed_models)]  # type: ignore[valid-type]
        field_annotation = Annotated[
            AllowedModels, Field(description=description)
        ]
        return create_model(
            "Config",
            __base__=cls.Config,
            model=(field_annotation, default_model),
        )

    def __init__(
        self, dial_factory: DialFactory, settings: CommonGroupSettings
    ):
        """Initialize the LLM metric with a DIAL factory and group settings.

        Args:
            dial_factory: Factory object that creates DIAL resource instances
                          (LLMs, embeddings models, etc.)
            settings: Group settings providing the allowed models and default
                      model for this metric's config_cls Literal patching
        """
        super().__init__()
        self._dial_factory = dial_factory
        self.config_cls = self.create_config_cls(settings)
