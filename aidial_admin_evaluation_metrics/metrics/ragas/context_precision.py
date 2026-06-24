"""Context Precision metric backed by ragas."""

from pydantic import BaseModel
from ragas.metrics.collections import (
    ContextPrecisionWithReference as RagasContextPrecision,
)

from aidial_admin_evaluation_metrics.app_config import DEFAULT_MODEL
from aidial_admin_evaluation_metrics.metrics.common.base_metric import (
    MetricExample,
)
from aidial_admin_evaluation_metrics.metrics.ragas.base import BaseRagasMetric
from aidial_admin_evaluation_metrics.metrics.ragas.input_types import (
    Reference,
    RetrievedContexts,
    UserInput,
)
from aidial_admin_evaluation_metrics.metrics.ragas.output_types import (
    RagasDetails,
    RagasScoreField,
)


class ContextPrecisionMetric(BaseRagasMetric):
    name: str = "ragas.context_precision"
    display_name: str = "Ragas: Context Precision"
    description: str = (
        "Evaluates the retriever's ability to rank relevant chunks higher than irrelevant ones for a given query in the retrieved context. "
        "Specifically, it assesses the degree to which relevant chunks in the retrieved context are placed at the top of the ranking. "
        "Requires LLM access via DIAL. "
        "For more details: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/"
    )

    class Input(BaseModel):
        user_input: UserInput
        reference: Reference
        retrieved_contexts: RetrievedContexts

    examples = [
        MetricExample(
            config=BaseRagasMetric.Config(model=DEFAULT_MODEL),
            input=Input(
                user_input="What year was the Eiffel Tower completed?",
                reference="The Eiffel Tower was completed in 1889.",
                retrieved_contexts=[
                    "The Eiffel Tower was completed in 1889 for the World's Fair.",
                    "Paris is the capital of France.",
                ],
            ),
            expected_output=BaseRagasMetric.Output(
                score=RagasScoreField(value=1.0, details=RagasDetails())
            ),
        )
    ]

    def _build_ragas_metric(self, config, dial_llm):
        return RagasContextPrecision(llm=dial_llm)

    def _build_score_kwargs(self, input: Input) -> dict:  # type: ignore[override]
        return {
            "user_input": input.user_input,
            "reference": input.reference,
            "retrieved_contexts": input.retrieved_contexts,
        }
