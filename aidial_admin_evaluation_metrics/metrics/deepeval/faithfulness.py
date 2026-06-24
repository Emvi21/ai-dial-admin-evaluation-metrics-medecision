"""Faithfulness metric backed by deepeval."""

from typing import Annotated, Optional

from deepeval.metrics import FaithfulnessMetric as DeepEvalFaithfulnessMetric
from pydantic import BaseModel, Field

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
    RetrievalContext,
)
from aidial_admin_evaluation_metrics.metrics.deepeval.output_types import (
    DeepEvalDetails,
    DeepEvalScoreField,
)


class FaithfulnessMetric(BaseDeepEvalRAGMetric):
    name: str = "deepeval.faithfulness"
    display_name: str = "DeepEval: Faithfulness"
    description: str = (
        "The faithfulness metric uses LLM-as-a-judge to evaluate whether the actual_output "
        "factually aligns with the contents of the retrieval_context (i.e., whether the "
        "answer is supported by the retrieved evidence). Requires LLM access via DIAL. "
        "For more details: https://deepeval.com/docs/metrics-faithfulness"
    )

    class Config(BaseDeepEvalRAGMetric.Config):
        truths_extraction_limit: Annotated[
            Optional[int],
            Field(
                default=None,
                ge=1,
                description=(
                    "Maximum number of factual truths to extract from the "
                    "retrieval context; if None, all extracted truths are considered."
                ),
            ),
        ] = None

        penalize_ambiguous_claims: Annotated[
            bool,
            Field(
                default=False,
                description=(
                    "If True, ambiguous claims are not counted as faithful."
                ),
            ),
        ] = False

    class Input(BaseModel):
        input: DeepevalInput
        actual_output: ActualOutput
        retrieval_context: RetrievalContext

    deepeval_metric_cls = DeepEvalFaithfulnessMetric

    examples = [
        MetricExample(
            config=Config(model=DEFAULT_MODEL, threshold=0.5),
            input=Input(
                input="What is the diameter of the Earth?",
                actual_output="The diameter of the Earth is approximately 12,742 kilometers.",
                retrieval_context=[
                    "The Earth has a diameter of approximately 12,742 kilometers."
                ],
            ),
            expected_output=BaseDeepEvalRAGMetric.Output(
                score=DeepEvalScoreField(
                    value=1.0,
                    details=DeepEvalDetails(
                        reason="The answer is fully supported by the retrieved context.",
                        verbose_logs="Truths:\n['The Earth has a diameter of approximately 12,742 kilometers.']\n \nClaims:\n['The diameter of the Earth is approximately 12,742 kilometers.']",
                    ),
                )
            ),
        )
    ]
