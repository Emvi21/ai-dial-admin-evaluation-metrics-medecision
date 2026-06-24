"""Faithfulness metric backed by ragas."""

from pydantic import BaseModel
from ragas.metrics.collections import Faithfulness as RagasFaithfulness

from aidial_admin_evaluation_metrics.app_config import DEFAULT_MODEL
from aidial_admin_evaluation_metrics.metrics.common.base_metric import (
    MetricExample,
)
from aidial_admin_evaluation_metrics.metrics.ragas.base import BaseRagasMetric
from aidial_admin_evaluation_metrics.metrics.ragas.input_types import (
    Response,
    RetrievedContexts,
    UserInput,
)
from aidial_admin_evaluation_metrics.metrics.ragas.output_types import (
    RagasDetails,
    RagasScoreField,
)


class FaithfulnessMetric(BaseRagasMetric):
    name: str = "ragas.faithfulness"
    display_name: str = "Ragas: Faithfulness"
    description: str = (
        "Measures how factually consistent a response is with the retrieved context. "
        "A response is considered faithful if all its claims can be supported by the retrieved context. "
        "Requires LLM access via DIAL. "
        "For more details: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/"
    )

    class Input(BaseModel):
        user_input: UserInput
        response: Response
        retrieved_contexts: RetrievedContexts

    examples = [
        MetricExample(
            config=BaseRagasMetric.Config(model=DEFAULT_MODEL),
            input=Input(
                user_input="When was the Eiffel Tower built?",
                response="The Eiffel Tower was built in 1889.",
                retrieved_contexts=[
                    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris. "
                    "It was constructed from 1887 to 1889."
                ],
            ),
            expected_output=BaseRagasMetric.Output(
                score=RagasScoreField(value=1.0, details=RagasDetails())
            ),
        )
    ]

    def _build_ragas_metric(self, config, dial_llm):
        return RagasFaithfulness(llm=dial_llm)

    def _build_score_kwargs(self, input: Input) -> dict:  # type: ignore[override]
        return {
            "user_input": input.user_input,
            "response": input.response,
            "retrieved_contexts": input.retrieved_contexts,
        }
