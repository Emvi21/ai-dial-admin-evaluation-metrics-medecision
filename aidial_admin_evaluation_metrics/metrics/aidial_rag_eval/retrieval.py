"""Retrieval metric using ai-dial-rag-eval library."""


from aidial_rag_eval.facts.citation import CitationMatcher
from aidial_rag_eval.facts.exact import ExactStringMatcher
from aidial_rag_eval.retrieval.metrics import calculate_metrics
from pydantic import BaseModel

from aidial_admin_evaluation_metrics.metrics.aidial_rag_eval.config_types import (
    MatcherConfig,
    MatcherType,
)
from aidial_admin_evaluation_metrics.metrics.aidial_rag_eval.input_types import (
    ContextInput,
    GroundTruthFactsInput,
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
from aidial_admin_evaluation_metrics.metrics.common.base_metric import (
    BaseMetric,
    MetricExample,
)
from aidial_admin_evaluation_metrics.metrics.common.types import (
    MetricOutputField,
)

MATCHER_MAP = {
    "citation": CitationMatcher,
    "exact": ExactStringMatcher,
}


def get_matcher_type(matcher_name: MatcherType):
    matcher_type = MATCHER_MAP.get(matcher_name)
    if not matcher_type:
        raise ValueError(f"Unsupported matcher type: {matcher_name}")
    return matcher_type


class RetrievalMetric(BaseMetric):
    """Evaluates RAG retrieval quality based on ground truth facts.

    This metric uses the ai-dial-rag-eval library to calculate retrieval metrics
    by matching ground truth facts against retrieved context chunks.

    Metrics calculated:
    - Recall: Ratio of facts found in the context
    - Precision: Ratio of context chunks that contain facts
    - F1: Harmonic mean of precision and recall
    - MRR: Mean Reciprocal Rank of the first occurrence of each fact
    """

    name: str = "aidial_rag_eval.retrieval"
    display_name: str = "DIAL RAG Eval: Retrieval"
    description: str = "Evaluates RAG retrieval based on the ground truth facts using ai-dial-rag-eval library. Supports configurable matching strategies: fuzzy citation matching (default) or exact equality matching."

    class Config(BaseModel):
        """Configuration for retrieval metric."""

        matcher: MatcherConfig = "citation"

    class Input(BaseModel):
        """Input for retrieval evaluation."""

        ground_truth_facts: GroundTruthFactsInput
        retrieved_context: ContextInput

    class Output(BaseModel):
        """Output from retrieval evaluation."""

        recall: RecallOutput
        precision: PrecisionOutput
        f1: F1ScoreOutput
        mrr: MrrOutput

    examples = [
        MetricExample(
            config=Config(matcher="citation"),
            input=Input(
                ground_truth_facts=[
                    "The diameter of the Earth is approximately 12,742 kilometers.",
                    "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.",
                ],
                retrieved_context=[
                    "The Earth, our home planet, is the third planet from the sun.",
                    "It's the only planet known to have an atmosphere containing free oxygen and oceans of liquid water on its surface. The diameter of the Earth is approximately 12,742 kilometers.",
                    "The Pacific Ocean is extending from the Arctic Ocean in the north to the Southern Ocean in the south.",
                ],
            ),
            expected_output=Output(
                precision=MetricOutputField(value=0.67),
                recall=RecallField(
                    value=0.5,
                    details=RecallDetails(
                        facts_ranks=[1, -1],
                        context_relevance=[0, 1, 0],
                        context_highlight=["", "", ""],
                    ),
                ),
                f1=MetricOutputField(value=0.57),
                mrr=MetricOutputField(value=0.625),
            ),
        )
    ]

    def _evaluate(self, config: Config, input: Input) -> Output:
        """Evaluate retrieval quality using ai-dial-rag-eval library.

        Args:
            config: Configuration containing matcher type selection
            input: Input containing ground_truth_facts and retrieved_context

        Returns:
            Output containing recall, precision, f1, and mrr metrics
        """

        facts = input.ground_truth_facts
        context = input.retrieved_context

        matcher = get_matcher_type(config.matcher)
        match_result = matcher.match_facts(facts, context)
        metrics = calculate_metrics(match_result)

        row = metrics | match_result._asdict()

        return self.Output(
            recall=extract_recall(row),
            precision=extract_retrieval_metric(row, "precision"),
            f1=extract_retrieval_metric(row, "f1"),
            mrr=extract_retrieval_metric(row, "mrr"),
        )
