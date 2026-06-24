"""Base class for deepeval-backed metrics."""
import logging
import math
import os
from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel

from aidial_admin_evaluation_metrics.app_config import DeepEvalGroupSettings
from aidial_admin_evaluation_metrics.dial.llm_client import DialFactory
from aidial_admin_evaluation_metrics.metrics.common.base_llm_metric import (
    BaseLLMMetric,
)
from aidial_admin_evaluation_metrics.metrics.common.types import MetricError
from aidial_admin_evaluation_metrics.metrics.deepeval.config_types import (
    IncludeReasonConfig,
    ModelConfig,
    StrictModeConfig,
    ThresholdConfig,
)
from aidial_admin_evaluation_metrics.metrics.deepeval.deepeval_llm import (
    DialDeepEvalLLM,
)
from aidial_admin_evaluation_metrics.metrics.deepeval.output_types import (
    DeepEvalDetails,
    DeepEvalScoreField,
    ScoreOutput,
)

logger = logging.getLogger(__name__)


def check_telemetry_opt_out():
    deepeval_telemetry_opt_out = os.getenv("DEEPEVAL_TELEMETRY_OPT_OUT")
    if deepeval_telemetry_opt_out is None:
        raise RuntimeError(
            "Environment variable DEEPEVAL_TELEMETRY_OPT_OUT is not set. "
            "Please set it to 1 or 0 to explicitly opt in or out of telemetry."
        )
    logger.info(
        f"DEEPEVAL_TELEMETRY_OPT_OUT is set to {deepeval_telemetry_opt_out}"
    )


# Make sure that DEEPEVAL_TELEMETRY_OPT_OUT is explicitly set to 0 or 1
# We do not want unintentional data sharing without explicit consent
check_telemetry_opt_out()


class BaseDeepEvalMetric(BaseLLMMetric):
    """Base class for all deepeval-backed metrics.

    Provides a shared `Config`, `Output`, and helper logic for instantiating
    DeepEval metrics and running `measure()`.

    Exposed config fields intentionally exclude `async_mode` and
    `evaluation_template` for service-level control and security. The
    `model` field is wrapped into a `DialDeepEvalLLM` before being passed to
    DeepEval.
    """

    class Config(BaseModel):
        model: ModelConfig
        threshold: ThresholdConfig = 0.5
        strict_mode: StrictModeConfig = False

    class Output(BaseModel):
        score: ScoreOutput

    deepeval_metric_cls: type[BaseMetric] = BaseMetric

    def __init__(
        self,
        dial_factory: DialFactory,
        settings: DeepEvalGroupSettings,
    ):
        super().__init__(dial_factory, settings)
        self._verbose_mode = settings.verbose_mode

    def build_test_case(self, input: BaseModel) -> LLMTestCase:
        if not isinstance(input, self.Input):
            raise ValueError(
                f"Expected input of type {self.Input}, got {type(input)}"
            )

        return LLMTestCase(**input.model_dump())

    def build_metric(
        self, config: Config, deepeval_llm: DialDeepEvalLLM
    ) -> BaseMetric:
        """Instantiate the underlying deepeval metric with the provided config and LLM."""

        metric_kwargs: dict[str, Any] = config.model_dump()

        # Override model to convert model name to a DialDeepEvalLLM wrapper
        metric_kwargs["model"] = deepeval_llm

        # Override async_mode to False, since it is dictated by the service implementation
        # and not exposed to the used in Config
        metric_kwargs["async_mode"] = False

        # Do not expose verbose_mode via the public API, since deepeval verbose more prints
        # the details to stdout and does not affect reason or verbose_logs in the output
        metric_kwargs["verbose_mode"] = self._verbose_mode

        try:
            metric = self.deepeval_metric_cls(**metric_kwargs)
            return metric
        except TypeError as e:
            raise TypeError(
                f"Failed to construct deepeval metric {self.deepeval_metric_cls.__name__} with kwargs {list(metric_kwargs.keys())}: {e}"
            ) from e

    async def evaluate_async(
        self, config: BaseDeepEvalMetric.Config, input: BaseModel
    ) -> BaseDeepEvalMetric.Output:
        """Run a deepeval metric asynchronously via a_measure().

        Calls a_measure() directly so DialDeepEvalLLM.a_generate() is used,
        keeping all LLM I/O non-blocking on the event loop.
        """
        assert isinstance(
            config, self.Config
        ), f"Expected config of type {self.Config}, got {type(config)}"
        assert isinstance(
            input, self.Input
        ), f"Expected input of type {self.Input}, got {type(input)}"

        try:
            deepeval_llm = DialDeepEvalLLM(self._dial_factory, config.model)

            test_case = self.build_test_case(input)

            metric = self.build_metric(config, deepeval_llm)
            await metric.a_measure(
                test_case,
                _show_indicator=False,
                _log_metric_to_confident=False,
            )

            score = metric.score
            return self.Output(
                score=DeepEvalScoreField(
                    value=math.nan if score is None else float(score),
                    details=DeepEvalDetails(
                        reason=metric.reason or "",
                        verbose_logs=metric.verbose_logs or "",
                    ),
                )
            )
        except Exception as e:
            logger.exception("Error in %s: %s", self.name, e)
            return self.Output(score=MetricError(message=str(e)))


class BaseDeepEvalRAGMetric(BaseDeepEvalMetric):
    """Extension of BaseDeepEvalMetric for deepeval's built-in RAG metrics.

    Adds ``include_reason`` to the shared Config so callers can control whether
    the metric returns a human-readable explanation alongside its score.
    """

    class Config(BaseDeepEvalMetric.Config):
        include_reason: IncludeReasonConfig = True
