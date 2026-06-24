"""Answer relevancy metric backed by deepeval."""

from deepeval.metrics import (
    AnswerRelevancyMetric as DeepEvalAnswerRelevancyMetric,
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
)
from aidial_admin_evaluation_metrics.metrics.deepeval.output_types import (
    DeepEvalDetails,
    DeepEvalScoreField,
)


class AnswerRelevancyMetric(BaseDeepEvalRAGMetric):
    name: str = "deepeval.answer_relevancy"
    display_name: str = "DeepEval: Answer Relevancy"
    description: str = (
        "The answer relevancy metric uses LLM-as-a-judge to measure the quality of "
        "your RAG pipeline's generator by evaluating how relevant the actual_output "
        "of your LLM application is compared to the provided input. Requires LLM "
        "access via DIAL. For more details: https://deepeval.com/docs/metrics-answer-relevancy"
    )

    class Input(BaseModel):
        input: DeepevalInput
        actual_output: ActualOutput

    deepeval_metric_cls = DeepEvalAnswerRelevancyMetric

    examples = [
        MetricExample(
            config=BaseDeepEvalRAGMetric.Config(
                model=DEFAULT_MODEL, threshold=0.5
            ),
            input=Input(
                input="What is the diameter of the Earth?",
                actual_output="The diameter of the Earth is approximately 12,742 kilometers.",
            ),
            expected_output=BaseDeepEvalRAGMetric.Output(
                score=DeepEvalScoreField(
                    value=1.0,
                    details=DeepEvalDetails(
                        reason="The output directly answers the question about the diameter of the Earth.",
                        verbose_logs="Statements:\n['The diameter of the Earth is approximately 12,742 kilometers.']\n \nVerdicts:\n[AnswerRelevancyVerdict(verdict='yes', reason=None)]",
                    ),
                )
            ),
        )
    ]
