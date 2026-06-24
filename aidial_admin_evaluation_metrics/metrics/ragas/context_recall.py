"""Context Recall metric backed by ragas."""

from pydantic import BaseModel
from ragas.metrics.collections import ContextRecall as RagasContextRecall

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


class ContextRecallMetric(BaseRagasMetric):
    name: str = "ragas.context_recall"
    display_name: str = "Ragas: Context Recall"
    description: str = (
        "Measures how many of the relevant documents (or pieces of information) were successfully retrieved. "
        "It focuses on not missing important results. "
        "Higher recall means fewer relevant documents were left out. "
        "In short, recall is about not missing anything important. "
        "Requires LLM access via DIAL. "
        "For more details: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/"
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
                ],
            ),
            expected_output=BaseRagasMetric.Output(
                score=RagasScoreField(value=1.0, details=RagasDetails())
            ),
        )
    ]

    def _build_ragas_metric(self, config, dial_llm):
        return RagasContextRecall(llm=dial_llm)

    def _build_score_kwargs(self, input: Input) -> dict:  # type: ignore[override]
        return {
            "user_input": input.user_input,
            "retrieved_contexts": input.retrieved_contexts,
            "reference": input.reference,
        }
