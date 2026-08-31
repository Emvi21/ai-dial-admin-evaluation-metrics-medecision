"""MAF Full Evaluation Pipeline: chains Node 1 → Node 2 → Node 3 in a single metric call."""

from typing import Annotated, Any

from pydantic import BaseModel, Field

from aidial_admin_evaluation_metrics.app_config import CommonGroupSettings
from aidial_admin_evaluation_metrics.dial.llm_client import DialFactory
from aidial_admin_evaluation_metrics.metrics.common.base_llm_metric import (
    BaseLLMMetric,
)
from aidial_admin_evaluation_metrics.metrics.common.base_metric import (
    MetricExample,
)
from aidial_admin_evaluation_metrics.metrics.common.types import (
    MetricError,
    MetricOutputField,
)
from aidial_admin_evaluation_metrics.metrics.maf.node1_questionnaire import (
    Node1QuestionnaireMetric,
)
from aidial_admin_evaluation_metrics.metrics.maf.node2_qa_extraction import (
    Node2QAExtractionMetric,
)
from aidial_admin_evaluation_metrics.metrics.maf.node3_decision import (
    Node3DecisionMetric,
)


class MAFPipelineMetric(BaseLLMMetric):
    """Full MAF evaluation pipeline: Node 1 → Node 2 → Node 3.

    Runs all three nodes sequentially, passing intermediate results internally.
    Surfaces all sub-metrics from every node in a single platform call.
    """

    name: str = "maf.pipeline"
    display_name: str = "MAF Full Evaluation Pipeline"
    description: str = (
        "Runs the full MAF evaluation pipeline (Node 1 → Node 2 → Node 3) in a single call. "
        "Returns all sub-metrics: coverage_recall, efficiency_precision, f_beta (Node 1), "
        "n2_accuracy, n2_accuracy_all (Node 2), decision_match, justification_match, "
        "overall_passed (Node 3). Requires LLM."
    )

    class Config(BaseModel):
        model: str = Field(description="The LLM deployment name for evaluation.")

    class Input(BaseModel):
        gt_criteria: Annotated[
            list[dict],
            Field(
                description=(
                    "GT criteria list with 'id', 'description', 'expected_value'. "
                    "C*/EX* criteria are evaluated by Nodes 1 and 2."
                )
            ),
        ]
        questionnaire: Annotated[
            list[dict],
            Field(
                description=(
                    "List of generated question dicts with 'value', 'options', "
                    "optional 'response_type'. Evaluated by Node 1."
                )
            ),
        ]
        actual_qa: Annotated[
            list[dict],
            Field(
                description=(
                    "Q&A pairs from the extractor. Each dict may have "
                    "'question', 'ai_answer'/'value'/'result', 'rationale', 'citations'. "
                    "Evaluated by Node 2."
                )
            ),
        ]
        gt_pathways: Annotated[
            list[dict],
            Field(
                description=(
                    "GT pathway list with 'pathway_id', 'decision', and either "
                    "'logic' (e.g. 'C1 AND C2') or 'criteria_ids' list. Used by Node 3."
                )
            ),
        ]
        actual_decision: Annotated[
            Any,
            Field(
                description=(
                    "Agent's decision output. Dict with 'decision', 'rationale', 'criteria' "
                    "(or 'full_notes'). JSON string also accepted. Evaluated by Node 3."
                )
            ),
        ]
        service_context: Annotated[
            dict | None,
            Field(
                default=None,
                description=(
                    "Optional dict with 'service_code' and/or 'service_description' "
                    "for Node 1 service-context redundancy checks."
                ),
            ),
        ] = None
        policy_text: Annotated[
            str | None,
            Field(
                default=None,
                description=(
                    "Optional policy text (up to 30k chars used by Node 1). "
                    "Enables Rule 8 policy cross-check."
                ),
            ),
        ] = None
        case_inputs: Annotated[
            dict | None,
            Field(
                default=None,
                description=(
                    "Optional case input fields (e.g. 'treatment_setting_desc') "
                    "for Node 3 non-standard criterion resolution."
                ),
            ),
        ] = None

    class Output(BaseModel):
        # ── Node 1 ───────────────────────────────────────────────────────────
        coverage_recall: Annotated[
            MetricOutputField | MetricError,
            Field(
                discriminator="type",
                description="[Node 1] Fraction of GT criteria with FULL coverage (0–1).",
            ),
        ]
        efficiency_precision: Annotated[
            MetricOutputField | MetricError,
            Field(
                discriminator="type",
                description="[Node 1] Fraction of clinical questions that are essential (0–1).",
            ),
        ]
        f_beta: Annotated[
            MetricOutputField | MetricError,
            Field(
                discriminator="type",
                description="[Node 1] F-beta (beta=2, recall-weighted) combining recall and precision (0–1).",
            ),
        ]
        # ── Node 2 ───────────────────────────────────────────────────────────
        n2_accuracy: Annotated[
            MetricOutputField | MetricError,
            Field(
                discriminator="type",
                description="[Node 2] Extraction accuracy over N1-covered criteria (0–1).",
            ),
        ]
        n2_accuracy_all: Annotated[
            MetricOutputField | MetricError,
            Field(
                discriminator="type",
                description="[Node 2] Extraction accuracy over all criteria, penalising N1 misses (0–1).",
            ),
        ]
        # ── Node 3 ───────────────────────────────────────────────────────────
        decision_match: Annotated[
            MetricOutputField | MetricError,
            Field(
                discriminator="type",
                description="[Node 3] 1.0 if actual decision matches contextual GT, else 0.0.",
            ),
        ]
        justification_match: Annotated[
            MetricOutputField | MetricError,
            Field(
                discriminator="type",
                description="[Node 3] 1.0 if rationale satisfies at least one acceptable justification path.",
            ),
        ]
        overall_passed: Annotated[
            MetricOutputField | MetricError,
            Field(
                discriminator="type",
                description="[Node 3] 1.0 if both decision and justification match (or override Rule 6/7 fires).",
            ),
        ]

    examples = [
        MetricExample(
            name="perfect pipeline",
            description="Fully covered criterion, correct extraction, correct decision",
            config=Config(model="gemini-3.5-flash"),
            input=Input(
                gt_criteria=[{"id": "C1", "description": "Patient age >= 18", "expected_value": "TRUE"}],
                questionnaire=[{"value": "Is the patient 18 or older?", "options": ["Yes", "No"]}],
                actual_qa=[{"question": "Is the patient 18 or older?", "value": "Yes"}],
                gt_pathways=[{"pathway_id": "P1", "logic": "C1", "decision": "criteria_met"}],
                actual_decision={
                    "decision": "criteria_met",
                    "rationale": "Patient is 45 years old, meeting the age criterion.",
                    "criteria": [],
                },
            ),
            expected_output=Output(
                coverage_recall=MetricOutputField(value=1.0),
                efficiency_precision=MetricOutputField(value=1.0),
                f_beta=MetricOutputField(value=1.0),
                n2_accuracy=MetricOutputField(value=1.0),
                n2_accuracy_all=MetricOutputField(value=1.0),
                decision_match=MetricOutputField(value=1.0),
                justification_match=MetricOutputField(value=1.0),
                overall_passed=MetricOutputField(value=1.0),
            ),
        ),
    ]

    def __init__(self, dial_factory: DialFactory, settings: CommonGroupSettings):
        super().__init__(dial_factory, settings)
        self._n1 = Node1QuestionnaireMetric(dial_factory, settings)
        self._n2 = Node2QAExtractionMetric(dial_factory, settings)
        self._n3 = Node3DecisionMetric(dial_factory, settings)

    async def evaluate_async(self, config: Config, input: Input) -> Output:
        # ── Node 1: questionnaire coverage ───────────────────────────────────
        n1_out = await self._n1.evaluate_async(
            Node1QuestionnaireMetric.Config(model=config.model),
            Node1QuestionnaireMetric.Input(
                gt_criteria=input.gt_criteria,
                questionnaire=input.questionnaire,
                service_context=input.service_context,
                policy_text=input.policy_text,
            ),
        )

        if isinstance(n1_out.coverage_recall, MetricError):
            err = n1_out.coverage_recall
            return self.Output(
                coverage_recall=err, efficiency_precision=err, f_beta=err,
                n2_accuracy=err, n2_accuracy_all=err,
                decision_match=err, justification_match=err, overall_passed=err,
            )

        node1_result = n1_out.coverage_recall.details.get("node1_result", [])

        # ── Node 2: Q&A extraction accuracy ──────────────────────────────────
        n2_out = await self._n2.evaluate_async(
            Node2QAExtractionMetric.Config(model=config.model),
            Node2QAExtractionMetric.Input(
                gt_criteria=input.gt_criteria,
                actual_qa=input.actual_qa,
                node1_result=node1_result,
            ),
        )

        if isinstance(n2_out.n2_accuracy, MetricError):
            err = n2_out.n2_accuracy
            return self.Output(
                coverage_recall=n1_out.coverage_recall,
                efficiency_precision=n1_out.efficiency_precision,
                f_beta=n1_out.f_beta,
                n2_accuracy=err, n2_accuracy_all=err,
                decision_match=err, justification_match=err, overall_passed=err,
            )

        node2_result = n2_out.n2_accuracy.details.get("node2_result", {})

        # ── Node 3: decision reasoning audit ─────────────────────────────────
        n3_out = await self._n3.evaluate_async(
            Node3DecisionMetric.Config(model=config.model),
            Node3DecisionMetric.Input(
                gt_criteria=input.gt_criteria,
                gt_pathways=input.gt_pathways,
                actual_decision=input.actual_decision,
                node2_result=node2_result,
                case_inputs=input.case_inputs,
            ),
        )

        return self.Output(
            coverage_recall=n1_out.coverage_recall,
            efficiency_precision=n1_out.efficiency_precision,
            f_beta=n1_out.f_beta,
            n2_accuracy=n2_out.n2_accuracy,
            n2_accuracy_all=n2_out.n2_accuracy_all,
            decision_match=n3_out.decision_match,
            justification_match=n3_out.justification_match,
            overall_passed=n3_out.overall_passed,
        )
