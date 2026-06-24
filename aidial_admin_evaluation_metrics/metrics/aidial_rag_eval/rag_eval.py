"""Full rag eval metrics using ai-dial-rag-eval library's DataFrame interface."""

import logging

from aidial_rag_eval.dataframe.metrics import create_rag_eval_metrics_report
from pydantic import BaseModel

from aidial_admin_evaluation_metrics.app_config import (
    DEFAULT_MODEL,
    AidialRagEvalGroupSettings,
)
from aidial_admin_evaluation_metrics.dial.llm_client import DialFactory
from aidial_admin_evaluation_metrics.metrics.aidial_rag_eval.config_types import (
    MatcherConfig,
    ModelConfig,
)
from aidial_admin_evaluation_metrics.metrics.aidial_rag_eval.generation import (
    METRICS_BINDS,
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
    GroundTruthFactsInput,
    QuestionInput,
    get_answers_dataframe,
    get_ground_truth_dataframe,
)
from aidial_admin_evaluation_metrics.metrics.aidial_rag_eval.retrieval import (
    get_matcher_type,
)
from aidial_admin_evaluation_metrics.metrics.aidial_rag_eval.retrieval_output import (
    F1ScoreOutput,
    MrrOutput,
    PrecisionOutput,
    RecallDetails,
    RecallField,
    RecallOutput,
    extract_recall,
    extract_retrieval_metric,
)
from aidial_admin_evaluation_metrics.metrics.common.base_llm_metric import (
    BaseLLMMetric,
)
from aidial_admin_evaluation_metrics.metrics.common.base_metric import (
    MetricExample,
)
from aidial_admin_evaluation_metrics.metrics.common.types import (
    MetricOutputField,
)

logger = logging.getLogger(__name__)


class RagEvalMetric(BaseLLMMetric):
    name: str = "aidial_rag_eval.rag_eval"
    display_name: str = "DIAL RAG Eval: Full Evaluation"
    description: str = (
        "Evaluates RAG quality using ai-dial-rag-eval metrics. "
        "Includes retrieval metrics (recall, precision, f1, mrr) and "
        "generation metrics (ground truth to answer, answer to ground truth, context to answer). "
        "Requires LLM access via DIAL for generation metrics."
    )

    class Config(BaseModel):
        """Configuration for RAG evaluation metric."""

        matcher: MatcherConfig = "citation"
        model: ModelConfig

    def __init__(
        self,
        dial_factory: DialFactory,
        settings: AidialRagEvalGroupSettings,
    ):
        super().__init__(dial_factory, settings)
        self._max_concurrency = settings.max_concurrency

    class Input(BaseModel):
        """Input for RAG evaluation."""

        question: QuestionInput
        ground_truth_answer: GroundTruthAnswerInput
        ground_truth_facts: GroundTruthFactsInput
        context: ContextInput
        answer: AnswerInput
        documents: DocumentsInput = None

    class Output(BaseModel):
        """Output from RAG evaluation."""

        recall: RecallOutput
        precision: PrecisionOutput
        f1: F1ScoreOutput
        mrr: MrrOutput
        ground_truth_to_answer: GroundTruthToAnswerField
        answer_to_ground_truth: AnswerToGroundTruthField
        context_to_answer: ContextToAnswerField

    examples = [
        MetricExample(
            config=Config(matcher="citation", model=DEFAULT_MODEL),
            input=Input(
                question="What is the diameter of Earth?",
                ground_truth_answer="The diameter of Earth is about 12,742 km.",
                ground_truth_facts=["Earth fact"],
                context=["Earth facts"],
                answer="Earth is 12,742 km wide.",
            ),
            expected_output=Output(
                recall=RecallField(
                    value=1.0,
                    details=RecallDetails(
                        facts_ranks=[0],
                        context_relevance=[1],
                        context_highlight=[
                            '{"match": [{"text": "Earth fact", "facts": [0]}, {"text": "s", "facts": []}]}'
                        ],
                    ),
                ),
                precision=MetricOutputField(value=1.0),
                f1=MetricOutputField(value=1.0),
                mrr=MetricOutputField(value=1.0),
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
        """Evaluate RAG quality using ai-dial-rag-eval metrics.

        Args:
            config: Configuration containing LLM model selection
            input: Input containing question, ground truth, context, and answer

        Returns:
            Output containing RAG evaluation metrics including retrieval and generation scores
        """
        try:
            matcher = get_matcher_type(config.matcher)
            llm = self._dial_factory.create_llm(config.model)

            ground_truth_df = get_ground_truth_dataframe(input)
            answers_df = get_answers_dataframe(input)

            result_df = create_rag_eval_metrics_report(
                ground_truth=ground_truth_df,
                answers=answers_df,
                matcher=matcher,
                llm=llm,
                metric_binds=METRICS_BINDS,
                max_concurrency=self._max_concurrency,
                show_progress_bar=False,
            )
            row = extract_row(result_df)

            return self.Output(
                recall=extract_recall(row),
                precision=extract_retrieval_metric(row, "precision"),
                f1=extract_retrieval_metric(row, "f1"),
                mrr=extract_retrieval_metric(row, "mrr"),
                ground_truth_to_answer=extract_generation_metric(row, "gt_ans"),
                answer_to_ground_truth=extract_generation_metric(row, "ans_gt"),
                context_to_answer=extract_generation_metric(row, "ctx_ans"),
            )
        except Exception as e:
            logger.exception("Error during RAG evaluation: %s", e)
            return output_error(e, self.Output)
