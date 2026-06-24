"""Noise Sensitivity metric backed by ragas."""

from typing import Literal

from pydantic import BaseModel, Field
from ragas.metrics.collections import NoiseSensitivity as RagasNoiseSensitivity

from aidial_admin_evaluation_metrics.app_config import DEFAULT_MODEL
from aidial_admin_evaluation_metrics.metrics.common.base_metric import (
    MetricExample,
)
from aidial_admin_evaluation_metrics.metrics.ragas.base import BaseRagasMetric
from aidial_admin_evaluation_metrics.metrics.ragas.config_types import (
    ModelConfig,
)
from aidial_admin_evaluation_metrics.metrics.ragas.input_types import (
    Reference,
    Response,
    RetrievedContexts,
    UserInput,
)
from aidial_admin_evaluation_metrics.metrics.ragas.output_types import (
    RagasDetails,
    RagasScoreField,
)


class NoiseSensitivityMetric(BaseRagasMetric):
    name: str = "ragas.noise_sensitivity"
    display_name: str = "Ragas: Noise Sensitivity"
    description: str = (
        "Measures how often a system makes errors by providing incorrect responses when utilizing retrieved documents. "
        "The score ranges from 0 to 1, with lower values indicating better performance. "
        "In 'relevant' mode (default), measures errors when utilizing relevant retrieved documents. "
        "In 'irrelevant' mode, measures errors when utilizing irrelevant retrieved documents. "
        "Requires LLM access via DIAL. "
        "For more details: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/noise_sensitivity/"
    )

    class Config(BaseRagasMetric.Config):
        model: ModelConfig
        mode: Literal["relevant", "irrelevant"] = Field(
            default="relevant",
            description=(
                "Context sensitivity mode: 'relevant' measures errors from relevant-context noise, "
                "'irrelevant' measures errors from irrelevant context."
            ),
        )

    class Input(BaseModel):
        user_input: UserInput
        response: Response
        reference: Reference
        retrieved_contexts: RetrievedContexts

    examples = [
        MetricExample(
            config=Config(model=DEFAULT_MODEL, mode="relevant"),
            input=Input(
                user_input="When was the Eiffel Tower built?",
                response="The Eiffel Tower was built in 1889.",
                reference="The Eiffel Tower was constructed from 1887 to 1889.",
                retrieved_contexts=[
                    "The Eiffel Tower was constructed from 1887 to 1889 as the entrance arch "
                    "for the 1889 World's Fair."
                ],
            ),
            expected_output=BaseRagasMetric.Output(
                score=RagasScoreField(value=0.0, details=RagasDetails())
            ),
        )
    ]

    def _build_ragas_metric(self, config: Config, dial_llm):  # type: ignore[override]
        return RagasNoiseSensitivity(llm=dial_llm, mode=config.mode)

    def _build_score_kwargs(self, input: Input) -> dict:  # type: ignore[override]
        return {
            "user_input": input.user_input,
            "response": input.response,
            "reference": input.reference,
            "retrieved_contexts": input.retrieved_contexts,
        }
