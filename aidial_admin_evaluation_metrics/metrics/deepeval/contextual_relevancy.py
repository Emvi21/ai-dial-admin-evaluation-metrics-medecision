"""Contextual relevancy metric backed by deepeval."""

from deepeval.metrics import (
    ContextualRelevancyMetric as DeepEvalContextualRelevancyMetric,
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
    RetrievalContext,
)
from aidial_admin_evaluation_metrics.metrics.deepeval.output_types import (
    DeepEvalDetails,
    DeepEvalScoreField,
)


class ContextualRelevancyMetric(BaseDeepEvalRAGMetric):
    name: str = "deepeval.contextual_relevancy"
    display_name: str = "DeepEval: Contextual Relevancy"
    description: str = (
        "The contextual relevancy metric uses LLM-as-a-judge to measure the quality of "
        "your RAG pipeline's retriever by evaluating the overall relevance of the "
        "information presented in your retrieval_context for a given input. Requires LLM "
        "access via DIAL. For more details: https://deepeval.com/docs/metrics-contextual-relevancy"
    )

    class Input(BaseModel):
        input: DeepevalInput
        retrieval_context: RetrievalContext

    deepeval_metric_cls = DeepEvalContextualRelevancyMetric

    examples = [
        MetricExample(
            config=BaseDeepEvalRAGMetric.Config(
                model=DEFAULT_MODEL, threshold=0.5
            ),
            input=Input(
                input="What is the diameter of the Earth?",
                retrieval_context=[
                    "The Earth has a diameter of approximately 12,742 kilometers.",
                    "The Earth is the third planet from the Sun.",
                ],
            ),
            expected_output=BaseDeepEvalRAGMetric.Output(
                score=DeepEvalScoreField(
                    value=0.5,
                    details=DeepEvalDetails(
                        reason="Only one of the two retrieved context nodes is relevant to the question.",
                        verbose_logs="Verdicts:\n[ContextualRelevancyVerdicts(verdicts=[...])]",
                    ),
                )
            ),
        )
    ]
