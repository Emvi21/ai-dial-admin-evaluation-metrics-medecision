"""Contextual precision metric backed by deepeval."""

from deepeval.metrics import (
    ContextualPrecisionMetric as DeepEvalContextualPrecisionMetric,
)
from pydantic import BaseModel

from aidial_admin_evaluation_metrics.app_config import DEFAULT_MODEL
from aidial_admin_evaluation_metrics.metrics.common.base_metric import (
    MetricExample,
)
from aidial_admin_evaluation_metrics.metrics.deepeval.base import (
    BaseDeepEvalRAGMetric,
)
from aidial_admin_evaluation_metrics.metrics.deepeval.input_types import (
    ActualOutput,
    DeepevalInput,
    ExpectedOutput,
    RetrievalContext,
)
from aidial_admin_evaluation_metrics.metrics.deepeval.output_types import (
    DeepEvalDetails,
    DeepEvalScoreField,
)


class ContextualPrecisionMetric(BaseDeepEvalRAGMetric):
    name: str = "deepeval.contextual_precision"
    display_name: str = "DeepEval: Contextual Precision"
    description: str = (
        "The contextual precision metric uses LLM-as-a-judge to measure your RAG "
        "pipeline's retriever by evaluating whether nodes in your retrieval_context "
        "that are relevant to the given input are ranked higher than irrelevant ones. "
        "Requires LLM access via DIAL. For more details: https://deepeval.com/docs/metrics-contextual-precision"
    )

    class Input(BaseModel):
        input: DeepevalInput
        actual_output: ActualOutput
        expected_output: ExpectedOutput
        retrieval_context: RetrievalContext

    deepeval_metric_cls = DeepEvalContextualPrecisionMetric

    examples = [
        MetricExample(
            config=BaseDeepEvalRAGMetric.Config(
                model=DEFAULT_MODEL, threshold=0.5
            ),
            input=Input(
                input="What is the diameter of the Earth?",
                actual_output="The diameter of the Earth is approximately 12,742 kilometers.",
                expected_output="The diameter of the Earth is approximately 12,742 kilometers.",
                retrieval_context=[
                    "The Earth has a diameter of approximately 12,742 kilometers.",
                    "The Earth is the third planet from the Sun.",
                ],
            ),
            expected_output=BaseDeepEvalRAGMetric.Output(
                score=DeepEvalScoreField(
                    value=1.0,
                    details=DeepEvalDetails(
                        reason="The most relevant context node is ranked first.",
                        verbose_logs="Verdicts:\n[ContextualPrecisionVerdict(verdict=True, reason='Directly states the diameter.')]",
                    ),
                )
            ),
        )
    ]
