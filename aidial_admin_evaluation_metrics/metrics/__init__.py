from typing import Optional

import aidial_admin_evaluation_metrics.metrics.aidial_rag_eval as aidial_rag_eval_metrics
import aidial_admin_evaluation_metrics.metrics.deepdiff as deepdiff_metrics
import aidial_admin_evaluation_metrics.metrics.deepeval as deepeval_metrics
import aidial_admin_evaluation_metrics.metrics.ragas as ragas_metrics
import aidial_admin_evaluation_metrics.metrics.simple as simple_metrics
from aidial_admin_evaluation_metrics.app_config import MetricsSettings
from aidial_admin_evaluation_metrics.dial.llm_client import DialFactory
from aidial_admin_evaluation_metrics.metrics.common.base_metric import (
    BaseMetric,
)
from aidial_admin_evaluation_metrics.metrics.common.registry import (
    MetricsRegistry,
)


def _create_instances(
    settings: MetricsSettings,
    dial_factory: DialFactory,
) -> list[BaseMetric]:
    """Explicitly construct every metric instance with its resolved settings."""
    de = settings.deepeval
    re = settings.aidial_rag_eval
    ra = settings.ragas
    return [
        simple_metrics.ExactMatchMetric(),
        simple_metrics.RegexMatchMetric(),
        simple_metrics.SetComparisonMetric(),
        simple_metrics.MultipleSetsComparisonMetric(),
        aidial_rag_eval_metrics.RetrievalMetric(),
        aidial_rag_eval_metrics.NLIMetric(dial_factory, re),
        aidial_rag_eval_metrics.GenerationMetric(dial_factory, re),
        aidial_rag_eval_metrics.RagEvalMetric(dial_factory, re),
        deepeval_metrics.AnswerRelevancyMetric(dial_factory, de),
        deepeval_metrics.FaithfulnessMetric(dial_factory, de),
        deepeval_metrics.ContextualPrecisionMetric(dial_factory, de),
        deepeval_metrics.ContextualRecallMetric(dial_factory, de),
        deepeval_metrics.ContextualRelevancyMetric(dial_factory, de),
        deepeval_metrics.GEvalMetric(dial_factory, de),
        ragas_metrics.FaithfulnessMetric(dial_factory, ra),
        ragas_metrics.ContextPrecisionMetric(dial_factory, ra),
        ragas_metrics.ContextRecallMetric(dial_factory, ra),
        ragas_metrics.AnswerRelevancyMetric(dial_factory, ra),
        ragas_metrics.NoiseSensitivityMetric(dial_factory, ra),
        ragas_metrics.ToolCallAccuracyMetric(dial_factory, ra),
        ragas_metrics.ToolCallF1Metric(dial_factory, ra),
        deepdiff_metrics.DeepDiffMetric(),
    ]


def create_metrics_registry(
    dial_factory: DialFactory,
    settings: Optional[MetricsSettings] = None,
) -> MetricsRegistry:
    """Create a new MetricsRegistry with the given settings and factory."""
    resolved = settings or MetricsSettings()
    return MetricsRegistry(metrics=_create_instances(resolved, dial_factory))


__all__ = ["create_metrics_registry"]
