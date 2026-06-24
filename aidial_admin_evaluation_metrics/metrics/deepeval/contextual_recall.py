"""Contextual recall metric backed by deepeval."""

from deepeval.metrics import (
    ContextualRecallMetric as DeepEvalContextualRecallMetric,
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
    DeepevalInput,
    ExpectedOutput,
    RetrievalContext,
)
from aidial_admin_evaluation_metrics.metrics.deepeval.output_types import (
    DeepEvalDetails,
    DeepEvalScoreField,
)


class ContextualRecallMetric(BaseDeepEvalRAGMetric):
    name: str = "deepeval.contextual_recall"
    display_name: str = "DeepEval: Contextual Recall"
    description: str = (
        "The contextual recall metric uses LLM-as-a-judge to measure the quality of "
        "your RAG pipeline's retriever by evaluating the extent to which the "
        "retrieval_context aligns with the expected_output. Requires LLM access via DIAL. "
        "For more details: https://deepeval.com/docs/metrics-contextual-recall"
    )

    class Input(BaseModel):
        input: DeepevalInput
        expected_output: ExpectedOutput
        retrieval_context: RetrievalContext

    deepeval_metric_cls = DeepEvalContextualRecallMetric

    examples = [
        MetricExample(
            config=BaseDeepEvalRAGMetric.Config(
                model=DEFAULT_MODEL, threshold=0.5
            ),
            input=Input(
                input="What is the diameter of the Earth?",
                expected_output="The diameter of the Earth is approximately 12,742 kilometers.",
                retrieval_context=[
                    "The Earth has a diameter of approximately 12,742 kilometers.",
                    "The Earth orbits the Sun at a distance of about 150 million kilometers.",
                ],
            ),
            expected_output=BaseDeepEvalRAGMetric.Output(
                score=DeepEvalScoreField(
                    value=1.0,
                    details=DeepEvalDetails(
                        reason="All statements in the expected output are supported by the retrieved context.",
                        verbose_logs="Verdicts:\n[VerdictWithExpectedOutput(verdict=True, reason='Directly stated in context.')]",
                    ),
                )
            ),
        )
    ]
