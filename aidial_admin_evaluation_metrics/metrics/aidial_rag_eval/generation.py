"""Generation metrics using ai-dial-rag-eval library's DataFrame interface."""

import logging

from aidial_rag_eval.dataframe.metrics import create_generation_metrics_report
from pydantic import BaseModel

from aidial_admin_evaluation_metrics.app_config import (
    DEFAULT_MODEL,
    AidialRagEvalGroupSettings,
)
from aidial_admin_evaluation_metrics.dial.llm_client import DialFactory
from aidial_admin_evaluation_metrics.metrics.aidial_rag_eval.config_types import (
    ModelConfig,
)
from aidial_admin_evaluation_metrics.metrics.aidial_rag_eval.generation_output import (
    AnswerToGroundTruthField,
    ContextToAnswerField,
    GroundTruthToAnswerField,
    InferenceDetails,
    InferenceField,
    extract_generation_metric,
    extract_row,
    output_error,
)
from aidial_admin_evaluation_metrics.metrics.aidial_rag_eval.input_types import (
    AnswerInput,
    ContextInput,
    DocumentsInput,
    GroundTruthAnswerInput,
    QuestionInput,
    get_answers_dataframe,
    get_ground_truth_dataframe,
)
from aidial_admin_evaluation_metrics.metrics.common.base_llm_metric import (
    BaseLLMMetric,
)
from aidial_admin_evaluation_metrics.metrics.common.base_metric import (
    MetricExample,
)

logger = logging.getLogger(__name__)

METRICS_BINDS = [
    "ground_truth_to_answer_inference",
    "answer_to_ground_truth_inference",
    "context_to_answer_inference",
]


class GenerationMetric(BaseLLMMetric):
    """Evaluates answer quality using three NLI-based generation metrics.

    The metric uses the aidial-rag-eval library's DataFrame interface, wrapping single-row
    inputs into DataFrames for batch processing. Each inference score is computed using LLM
    to segment text into atomic statements and classify entailment.
    """

    name: str = "aidial_rag_eval.generation"
    display_name: str = "DIAL RAG Eval: Generation"
    description: str = (
        "Evaluates answer quality using three NLI-based generation metrics: "
        "ground_truth_to_answer, answer_to_ground_truth and context_to_answer. "
        "Requires LLM access via DIAL."
    )

    class Config(BaseModel):
        """Configuration for generation metrics."""

        model: ModelConfig

    def __init__(
        self,
        dial_factory: DialFactory,
        settings: AidialRagEvalGroupSettings,
    ):
        super().__init__(dial_factory, settings)
        self._max_concurrency = settings.max_concurrency

    class Input(BaseModel):
        """Input for aidial_rag_eval metrics."""

        question: QuestionInput
        ground_truth_answer: GroundTruthAnswerInput
        context: ContextInput
        answer: AnswerInput
        documents: DocumentsInput = None

    class Output(BaseModel):
        """Output from generation metrics evaluation."""

        ground_truth_to_answer: GroundTruthToAnswerField
        answer_to_ground_truth: AnswerToGroundTruthField
        context_to_answer: ContextToAnswerField

    examples = [
        MetricExample(
            config=Config(model=DEFAULT_MODEL),
            input=Input(
                question="What is the diameter of Earth?",
                ground_truth_answer="The diameter of the Earth is approximately 12,742 kilometers.",
                context=[
                    "The Earth is the third planet from the Sun.",
                    "The diameter of the Earth is approximately 12,742 kilometers, making it the fifth largest planet.",
                ],
                answer="Earth has a diameter of about 12,742 km.",
                documents=["earth_facts.txt"],
            ),
            expected_output=Output(
                ground_truth_to_answer=InferenceField(
                    value=1.0,
                    details=InferenceDetails(
                        json_explanation='[{"inference": 1.0, "hypothesis": "Earth is 12,742 km wide.", "premise": ["What is the diameter of Earth?\nThe diameter of Earth is about 12,742 km."], "explanation": "[{\\"explanation\\": \\"The statement directly restates the diameter of Earth as given in the premise.\\", \\"tag\\": \\"ENT\\", \\"statement\\": \\"Earth is 12,742 km wide.\\"}]"}]',
                        highlight='{"corpus": [{"text": "Earth is 12,742 km wide.", "score": 0.0, "title": 1.0}, {"text": "", "score": 0.0}]}',
                    ),
                ),
                answer_to_ground_truth=InferenceField(
                    value=1.0,
                    details=InferenceDetails(
                        json_explanation='[{"inference": 1.0, "hypothesis": "The diameter of Earth is about 12,742 km.", "premise": ["What is the diameter of Earth?\nEarth is 12,742 km wide."], "explanation": "[{\\"explanation\\": \\"The statement directly confirms the width of the Earth as given in the premise.\\", \\"tag\\": \\"ENT\\", \\"statement\\": \\"The diameter of Earth is about 12,742 km.\\"}]"}]',
                        highlight='{"corpus": [{"text": "The diameter of Earth is about 12,742 km.", "score": 0.0, "title": 1.0}, {"text": "", "score": 0.0}]}',
                    ),
                ),
                context_to_answer=InferenceField(
                    value=0.0,
                    details=InferenceDetails(
                        json_explanation='[{"inference": 0.0, "hypothesis": "Earth is 12,742 km wide.", "premise": ["Earth facts"], "explanation": "[{\\"explanation\\": \\"The premise is too general and does not provide the specific width of the Earth, making this statement neutral.\\", \\"tag\\": \\"NEUT\\", \\"statement\\": \\"Earth is 12,742 km wide.\\"}]"}]',
                        highlight='{"corpus": [{"text": "Earth is 12,742 km wide.", "score": -1.0, "title": 0.0}, {"text": "", "score": 0.0}]}',
                    ),
                ),
            ),
        )
    ]

    def _evaluate(self, config: Config, input: Input) -> Output:
        """Evaluate answer quality using three NLI-based generation metrics.

        Args:
            config: Configuration containing LLM model selection
            input: Input containing question, ground truth, context, and answer

        Returns:
            Output containing three inference scores with detailed explanations
        """
        try:
            llm = self._dial_factory.create_llm(config.model)

            ground_truth_df = get_ground_truth_dataframe(input)
            answers_df = get_answers_dataframe(input)

            result_df = create_generation_metrics_report(
                ground_truth=ground_truth_df,
                answers=answers_df,
                llm=llm,
                metric_binds=METRICS_BINDS,
                max_concurrency=self._max_concurrency,
                show_progress_bar=False,
            )
            row = extract_row(result_df)

            return self.Output(
                ground_truth_to_answer=extract_generation_metric(row, "gt_ans"),
                answer_to_ground_truth=extract_generation_metric(row, "ans_gt"),
                context_to_answer=extract_generation_metric(row, "ctx_ans"),
            )
        except Exception as e:
            logger.exception("Error during generation evaluation: %s", e)
            return output_error(e, self.Output)
