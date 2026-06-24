"""Answer Relevancy metric backed by ragas (requires embeddings)."""

from pydantic import BaseModel
from ragas.metrics.collections import AnswerRelevancy as RagasAnswerRelevancy
from ragas.metrics.collections.base import BaseMetric as RagasBaseMetric

from aidial_admin_evaluation_metrics.app_config import (
    DEFAULT_MODEL,
    RagasGroupSettings,
)
from aidial_admin_evaluation_metrics.dial.llm_client import DialFactory
from aidial_admin_evaluation_metrics.metrics.common.base_metric import (
    MetricExample,
)
from aidial_admin_evaluation_metrics.metrics.ragas.base import BaseRagasMetric
from aidial_admin_evaluation_metrics.metrics.ragas.config_types import (
    EmbeddingsModelConfig,
    ModelConfig,
)
from aidial_admin_evaluation_metrics.metrics.ragas.input_types import (
    Response,
    UserInput,
)
from aidial_admin_evaluation_metrics.metrics.ragas.output_types import (
    RagasDetails,
    RagasScoreField,
)
from aidial_admin_evaluation_metrics.metrics.ragas.ragas_embeddings import (
    DialRagasEmbeddings,
)
from aidial_admin_evaluation_metrics.metrics.ragas.ragas_llm import DialRagasLLM


class AnswerRelevancyMetric(BaseRagasMetric):
    name: str = "ragas.answer_relevancy"
    display_name: str = "Ragas: Answer Relevancy"
    description: str = (
        "Measures how relevant a response is to the user input. "
        "An answer is considered relevant if it directly and appropriately addresses the original question. "
        "This metric focuses on how well the answer matches the intent of the question, without evaluating factual accuracy. "
        "It penalizes answers that are incomplete or include unnecessary details. "
        "Requires LLM and embeddings access via DIAL. "
        "For more details: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/"
    )

    class Config(BaseRagasMetric.Config):
        model: ModelConfig
        embeddings_model: EmbeddingsModelConfig = "text-embedding-ada-002"

    class Input(BaseModel):
        user_input: UserInput
        response: Response

    examples = [
        MetricExample(
            config=Config(model=DEFAULT_MODEL),
            input=Input(
                user_input="What year was the Eiffel Tower completed?",
                response="The Eiffel Tower was completed in 1889.",
            ),
            expected_output=BaseRagasMetric.Output(
                score=RagasScoreField(value=1.0, details=RagasDetails())
            ),
        )
    ]

    def __init__(
        self,
        dial_factory: DialFactory,
        settings: RagasGroupSettings,
    ) -> None:
        super().__init__(dial_factory, settings)

    def _build_ragas_metric(
        self, config: "BaseRagasMetric.Config", dial_llm: DialRagasLLM
    ) -> RagasBaseMetric:
        assert isinstance(config, AnswerRelevancyMetric.Config)
        dial_embeddings = DialRagasEmbeddings(
            self._dial_factory.create_embeddings_model(config.embeddings_model)
        )
        return RagasAnswerRelevancy(llm=dial_llm, embeddings=dial_embeddings)

    def _build_score_kwargs(self, input: "AnswerRelevancyMetric.Input") -> dict:  # type: ignore[override]
        return {
            "user_input": input.user_input,
            "response": input.response,
        }
