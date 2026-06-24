"""Natural Language Inference (NLI) metric using ai-dial-rag-eval library."""

from typing import Annotated

from aidial_rag_eval.generation.inference import calculate_inference
from pydantic import BaseModel, Field

from aidial_admin_evaluation_metrics.app_config import (
    DEFAULT_MODEL,
    AidialRagEvalGroupSettings,
)
from aidial_admin_evaluation_metrics.dial.llm_client import DialFactory
from aidial_admin_evaluation_metrics.metrics.aidial_rag_eval.config_types import (
    ModelConfig,
)
from aidial_admin_evaluation_metrics.metrics.aidial_rag_eval.generation_output import (
    InferenceDetails,
    InferenceField,
)
from aidial_admin_evaluation_metrics.metrics.common.base_llm_metric import (
    BaseLLMMetric,
)
from aidial_admin_evaluation_metrics.metrics.common.base_metric import (
    MetricExample,
)
from aidial_admin_evaluation_metrics.metrics.common.types import MetricError


class NLIMetric(BaseLLMMetric):
    """Evaluates natural language inference from premise to hypothesis.

    This metric uses Natural Language Inference (NLI) to check if a hypothesis
    is entailed by a premise. The metric uses an LLM to:
    - Segment the hypothesis into atomic statements
    - Decontextualize statements for standalone evaluation
    - Classify each statement as ENT (entailment), CONT (contradiction), or NEUT (neutral)
    - Calculate inference score as the proportion of entailed statements

    An inference score of 1.0 means all statements in the hypothesis are entailed
    by the premise. A score of 0.0 means no statements are entailed.
    """

    name: str = "aidial_rag_eval.nli"
    display_name: str = "DIAL RAG Eval: NLI (Natural Language Inference)"
    description: str = (
        "Evaluates natural language inference using ai-dial-rag-eval library. "
        "Computes the proportion of statements in the hypothesis that are entailed "
        "by the premise. Requires LLM."
    )

    class Config(BaseModel):
        """Configuration for NLI metric."""

        model: ModelConfig

    def __init__(
        self,
        dial_factory: DialFactory,
        settings: AidialRagEvalGroupSettings,
    ):
        super().__init__(dial_factory, settings)
        self._max_concurrency = settings.max_concurrency

    class Input(BaseModel):
        """Input for NLI evaluation."""

        premise: Annotated[
            str,
            Field(
                description="The premise text from which the hypothesis should be inferred",
            ),
        ]
        hypothesis: Annotated[
            str,
            Field(
                description="The hypothesis text to check for entailment",
            ),
        ]

    class Output(BaseModel):
        """Output from NLI evaluation."""

        inference: Annotated[
            InferenceField | MetricError,
            Field(
                discriminator="type",
                description="Inference score (0-1) indicating the proportion of entailed statements",
            ),
        ]

    examples = [
        MetricExample(
            config=Config(model=DEFAULT_MODEL),
            input=Input(
                premise="The Earth is the third planet from the Sun. The diameter of the Earth is approximately 12,742 kilometers.",
                hypothesis="The Earth has a diameter of about 12,742 km.",
            ),
            expected_output=Output(
                inference=InferenceField(
                    value=1.0,
                    details=InferenceDetails(
                        json_explanation='[{"inference": 1.0, "hypothesis": "The Earth has a diameter of about 12,742 km.", "premise": ["The Earth is the third planet from the Sun. The diameter of the Earth is approximately 12,742 kilometers."], "explanation": "[{\\"explanation\\": \\"The statement directly repeats information given in the premise about the Earth\'s diameter.\\", \\"tag\\": \\"ENT\\", \\"statement\\": \\"The Earth has a diameter of about 12,742 km.\\"}]"}]',
                        highlight='{"corpus": [{"text": "The Earth has a diameter of about 12,742 km.", "score": 0.0, "title": 1.0}, {"text": "", "score": 0.0}]}',
                    ),
                )
            ),
        )
    ]

    def _evaluate(self, config: Config, input: Input) -> Output:
        """Evaluate natural language inference from premise to hypothesis.

        Args:
            config: Configuration containing LLM model selection
            input: Input containing premise and hypothesis

        Returns:
            Output containing inference score and detailed explanation
        """
        try:
            llm = self._dial_factory.create_llm(config.model)
            result = calculate_inference(
                premise=input.premise,
                hypothesis=input.hypothesis,
                llm=llm,
                max_concurrency=self._max_concurrency,
                show_progress_bar=False,
            )
            return self.Output(
                inference=InferenceField(
                    value=float(result.inference),
                    details=InferenceDetails(
                        json_explanation=result.json,
                        highlight=result.highlight,
                    ),
                )
            )
        except Exception as e:
            return self.Output(inference=MetricError(message=str(e)))
