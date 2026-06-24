"""Base class for ragas-backed metrics from ragas.metrics.collections."""

import logging
import math
from typing import cast

from pydantic import BaseModel
from ragas.metrics.collections.base import BaseMetric as RagasBaseMetric

from aidial_admin_evaluation_metrics.app_config import RagasGroupSettings
from aidial_admin_evaluation_metrics.dial.llm_client import DialFactory
from aidial_admin_evaluation_metrics.metrics.common.base_llm_metric import (
    BaseLLMMetric,
)
from aidial_admin_evaluation_metrics.metrics.common.types import MetricError
from aidial_admin_evaluation_metrics.metrics.ragas.config_types import (
    ModelConfig,
)
from aidial_admin_evaluation_metrics.metrics.ragas.output_types import (
    RagasDetails,
    RagasScoreField,
    ScoreOutput,
)
from aidial_admin_evaluation_metrics.metrics.ragas.ragas_llm import DialRagasLLM

logger = logging.getLogger(__name__)


class BaseRagasMetric(BaseLLMMetric):
    """Base class for all ragas-backed metrics from ragas.metrics.collections.

    Subclasses must override:
    - ``_build_ragas_metric(config, dial_llm)``: Instantiate the ragas metric with the
      configured LLM and any metric-specific options from config.
    - ``_build_score_kwargs(input)``: Map our Input Pydantic model to the keyword
      arguments expected by the ragas metric's ``ascore()`` method.

    ``evaluate_async()`` orchestrates the full flow:
    1. Create a LangChain model via ``_create_llm(config.model)``.
    2. Wrap it in ``DialRagasLLM``.
    3. Build the ragas metric instance.
    4. Await ``metric.ascore(**kwargs)`` — native async, no thread-pool.
    5. Return ``Output(score=RagasScoreField(...))`` or ``Output(score=MetricError(...))``.
    """

    class Config(BaseModel):
        model: ModelConfig

    class Output(BaseModel):
        score: ScoreOutput

    def __init__(
        self,
        dial_factory: DialFactory,
        settings: RagasGroupSettings,
    ) -> None:
        super().__init__(dial_factory, settings)

    def _build_ragas_metric(
        self,
        config: "BaseRagasMetric.Config",
        dial_llm: DialRagasLLM,
    ) -> RagasBaseMetric:
        raise NotImplementedError(
            "Subclasses must implement _build_ragas_metric()"
        )

    def _build_score_kwargs(self, input: BaseModel) -> dict:
        raise NotImplementedError(
            "Subclasses must implement _build_score_kwargs()"
        )

    async def _execute_metric(
        self, ragas_metric: RagasBaseMetric, input: BaseModel
    ) -> "BaseRagasMetric.Output":
        """Invoke ``ragas_metric.ascore`` and convert the result to our Output type."""
        kwargs = self._build_score_kwargs(input)
        result = await ragas_metric.ascore(**kwargs)
        score_value = (
            float(result.value) if result.value is not None else math.nan
        )
        return self.Output(
            score=RagasScoreField(
                value=score_value,
                details=RagasDetails(reason=result.reason),
            )
        )

    async def evaluate_async(
        self, config: "BaseRagasMetric.Config", input: BaseModel
    ) -> "BaseRagasMetric.Output":
        assert isinstance(
            config, self.Config
        ), f"Expected config of type {self.Config}, got {type(config)}"
        assert isinstance(
            input, self.Input
        ), f"Expected input of type {self.Input}, got {type(input)}"

        try:
            dial_llm = DialRagasLLM(self._dial_factory, config.model)
            metric = self._build_ragas_metric(config, dial_llm)
            return await self._execute_metric(metric, input)
        except Exception as e:
            logger.exception("Error in %s: %s", self.name, e)
            return self.Output(score=MetricError(message=str(e)))


class BaseDeterministicRagasMetric(BaseRagasMetric):
    """Base class for ragas metrics that require no LLM (rule-based scoring).

    Overrides ``evaluate_async()`` to skip ``DialRagasLLM`` construction,
    since deterministic metrics (e.g. ToolCallAccuracy, ToolCallF1) never
    call an LLM during scoring.

    Subclasses should define an empty ``Config`` (no ``model`` field) and
    omit any input fields that are not consumed by the ragas metric.
    """

    class Config(BaseModel):
        pass

    @classmethod
    def create_config_cls(cls, settings):
        # No model field to patch — return Config as-is.
        return cls.Config

    def _build_deterministic_ragas_metric(
        self, config: "BaseDeterministicRagasMetric.Config"
    ) -> RagasBaseMetric:
        raise NotImplementedError(
            "Subclasses must implement _build_deterministic_ragas_metric()"
        )

    async def evaluate_async(
        self, config: BaseModel, input: BaseModel
    ) -> "BaseRagasMetric.Output":
        assert isinstance(
            config, self.Config
        ), f"Expected config of type {self.Config}, got {type(config)}"
        assert isinstance(
            input, self.Input
        ), f"Expected input of type {self.Input}, got {type(input)}"

        try:
            ragas_metric = self._build_deterministic_ragas_metric(
                cast("BaseDeterministicRagasMetric.Config", config)
            )
            return await self._execute_metric(ragas_metric, input)
        except Exception as e:
            logger.exception("Error in %s: %s", self.name, e)
            return self.Output(score=MetricError(message=str(e)))
