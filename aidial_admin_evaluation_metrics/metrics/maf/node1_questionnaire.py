"""MAF Node 1: Questionnaire coverage evaluation metric.

Evaluates whether an AI-generated medical intake questionnaire covers every
GT criterion.

Outputs:
  coverage_recall   — fraction of GT criteria with FULL coverage
  efficiency_precision — fraction of clinical questions that are essential
  f_beta            — F-beta (beta=2, recall-weighted) combining both

Also emits a `node1_result` detail blob consumed by maf.node2_qa_extraction.
"""

import json
import re
from typing import Annotated, Literal

from langchain_core.messages import HumanMessage, SystemMessage
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

_NODE1_SYSTEM = """You are an expert clinical auditor evaluating an AI-generated medical intake questionnaire.
Your task is to determine if the generated questions successfully capture the clinical information required by the provided Ground Truth (GT) Criteria.

Core Standard:
- The Ground Truth (GT) Criteria define the MANDATORY clinical scope, patient populations, demographic bounds, numeric thresholds, and clinical conditions that must be assessed.
- Every clinical parameter specified in a GT criterion (e.g., age range, BMI cut-off, duration, clinical severity) must be fully and faithfully captured by the questionnaire.

Each question in the questionnaire has:
- `value`: the question text.
- `options`: the answer choices (verbatim clinical definitions for Multiple select, or empty [] for Free text).
- `response_type`: "Multiple select" or "Free text". All structured questions use Multiple select.

Rules for Evaluation:

1. Coverage Evaluation (Strictly Binary):
   - Broad or complex GT criteria do NOT need to be satisfied by a single question, but instead MUST be deconstructed into granular sub-questions (see Rule 5). Evaluate the collective information of all mapped questions as a union.
   - If the combined questions fully satisfy every condition, threshold, and clinical requirement of the GT criterion without scope gaps, the coverage is FULL.
   - If even a single condition, sub-component, or numeric/demographic range is missing, truncated, shifted, or incomplete, the coverage MUST be evaluated as NOT_COVERED.

2. Answer Options Validation:
   - All structured questions use **Multiple select**: each option is evaluated independently as a "clinical search target".
   - You MUST extract the verbatim option text into `extracted_options` before judging.
   - For **Free text** questions: open-ended capture is acceptable for capturing unstructured details (e.g., device name, free-form history).

3. Essential vs. Redundant:
   - `essential_question_indices`: 1-based indices of questions adding unique, necessary information to satisfy the criterion. If coverage is NOT_COVERED, this list MUST be empty [].
   - `redundant_question_indices`: 1-based indices of questions whose information space is entirely subsumed by an essential question. If `essential_question_indices` is empty, `redundant_question_indices` MUST also be empty [].

4. Modularity and Independence:
   - Multiple select allows a single question to cover multiple criteria simultaneously via independent options. Do NOT penalize a question solely for spanning multiple criteria.
   - Compound AND-criterion questions that bundle multiple independent mandatory requirements into a single checkbox without modularity violate Rule 5.

5. Strict Deconstruction of Complex/Nested Criteria:
   - If a clinical criterion contains nested, multi-part definitions (e.g., failure of conservative therapy requiring specific modalities and duration), the questions MUST deconstruct and specifically target each mandatory sub-component individually.
   - If the questionnaire only generates a high-level summary question (e.g., "Has conservative treatment failed?") with no granular child questions, downgrade coverage to NOT_COVERED.
   - **Mixed Parent-Child Fallacy:** penalize a summary/parent question combined with detailed child questions ONLY if, even after disregarding the parent question, the child questions taken together still fail to independently and fully cover every mandatory sub-component. If the child questions alone already provide full granular coverage, a redundant parent/summary question is NOT a coverage defect — index it under `redundant_question_indices` (Rule 3) instead of downgrading coverage to NOT_COVERED.

6. The Conditional/Procedural Question Coverage Fallacy:
   - If a criterion contains a conditional obligation (e.g., "If PT ongoing, progress notes required, else N/A"), the question options must provide distinct paths for both the active state and the N/A state. If the question fails to provide an explicit bypass/N/A option, mark NOT_COVERED.

7. Service-Context Redundancy:
   - Questions that re-ask for the requested procedure type already specified in `<Service_Context>` are redundant and must NOT provide coverage for any clinical criterion.

8. Option Scope & Parameter Fidelity (Strict Boundary Enforcement):
   - A question option must faithfully match the clinical scope and parameters of the GT criterion.
   - **Range Truncation / Boundary Shift:** If a GT criterion specifies a specific demographic or clinical range (e.g., "Age 10 to 18 years", "BMI 30 to 34.9", "Duration ≥ 6 weeks"), the question options MUST cover that entire specified window.
   - If an option narrows or shifts the boundary (e.g., defining adolescent as "aged 13 to 17 years" when the GT requires "Age 10 to 18 years"), any patient in the excluded window (e.g., ages 10, 11, and 12) is omitted or misclassified into an incorrect/investigational category. This represents a fatal clinical coverage gap. You MUST evaluate this as NOT_COVERED (Failure Rule: Rule 8 or Rule 10).
   - **Policy-Text Cross-Check (Before Flagging a Boundary Shift):** When `<Policy_Text>` is provided and the option's narrowing qualifier text (or a close paraphrase) appears verbatim in the policy source text, do NOT treat this as an unauthorized restriction under this rule — the questionnaire is faithfully reflecting the policy's own definition, and the discrepancy is in the GT criterion's own (possibly abbreviated) text. In that case set `failure_rule` = "Rule 8-GT-Gap" instead of a plain boundary-shift violation, and note the GT-text-vs-policy-text discrepancy in `clinical_scope_analysis`.
   - **Policy Expansion:** If an option introduces unauthorized diagnoses or conditions not part of the criterion, causing false-positive eligibility, downgrade to NOT_COVERED.

9. Q&A Agent Independence:
   - The downstream Q&A agent evaluates each question in complete isolation without access to other questions or questionnaire state.
   - Questions with cross-question references (e.g., "If 'Yes' to the previous question...") are invalid and result in NOT_COVERED.
   - Options must be self-contained and embed necessary qualifying and disqualifying clinical parameters.

10. Categorical & Demographic Partitions:
   - Questions that partition patients into mutually exclusive cohorts (e.g., Adult vs. Adolescent vs. Preadolescent) must align accurately with the GT criterion definitions.
   - A partition is flawed and fails coverage if its cutoffs create a mismatch against the GT criterion.
   - Example of Scope Mismatch Failure:
     - GT Criterion: "Age 10 to 18 years (adolescent)"
     - Question Options: "Adult (18+)", "Adolescent (13 to 17)", "Preadolescent child (under 13)"
     - Evaluation: NOT_COVERED. The question creates a 10-12 age blind spot where a 10, 11, or 12-year-old patient who clinically qualifies as an adolescent under the GT criterion is incorrectly forced into the preadolescent category.

Output Generation Protocol:
1. Populate `criterion_id` and `criterion_description`.
2. Extract verbatim options into `extracted_options`.
3. In `clinical_scope_analysis`, conduct a strict, explicit comparison of the GT criterion's required scope/thresholds against the extracted options.
4. If ALL parameters and clinical conditions match completely, set `coverage` = "FULL", `failure_rule` = None, and list `essential_question_indices`.
5. If ANY threshold, condition, or sub-component is mismatched, truncated, or missing, set `coverage` = "NOT_COVERED", specify the `failure_rule` (e.g. "Rule 8" or "Rule 10"), and set `essential_question_indices` = [].
6. In `considered_question_indices`, list the 1-based indices of every question you examined while judging this criterion — including any rejected, redundant, or insufficient ones. Populate this regardless of the coverage verdict; it is independent of `essential_question_indices`, which stays empty on NOT_COVERED.
"""


def _node1_user_prompt(
    gt_criteria: list[dict],
    questionnaire: list[dict],
    service_context: dict | None = None,
    policy_text: str | None = None,
) -> str:
    annotated_q = []
    for i, q in enumerate(questionnaire):
        entry: dict = {"index": i + 1, "value": q["value"], "options": q.get("options", [])}
        rt = q.get("response_type", "")
        if rt:
            entry["response_type"] = rt
        annotated_q.append(entry)

    service_section = ""
    if service_context:
        service_section = f"""
<Service_Context>
{json.dumps(service_context, indent=2)}
</Service_Context>
"""

    _MAX_POLICY_CHARS = 30_000
    policy_section = ""
    if policy_text:
        truncated = policy_text[:_MAX_POLICY_CHARS]
        if len(policy_text) > _MAX_POLICY_CHARS:
            truncated += f"\n\n[Policy text truncated at {_MAX_POLICY_CHARS} characters to fit context window]"
        policy_section = f"""
<Policy_Text>
{truncated}
</Policy_Text>
"""

    return f"""{service_section}{policy_section}
<GT_Criteria>
{json.dumps(gt_criteria, indent=2)}
</GT_Criteria>

<Generated_Questionnaire>
The following questions are 1-indexed. Refer to them by their 1-based index number (starting from 1).
{json.dumps(annotated_q, indent=2)}
</Generated_Questionnaire>
"""


class _CriterionAssessment(BaseModel):
    criterion_id: str
    criterion_description: str
    extracted_options: list[str] = Field(
        default_factory=list,
        description="Verbatim text of the specific question options mapped to this criterion. Empty if none.",
    )
    clinical_scope_analysis: str = Field(
        description="Detailed step-by-step verification comparing the GT criterion's required scope, age/numeric thresholds, and conditions against the extracted options.",
    )
    essential_question_indices: list[int] = Field(
        default_factory=list,
        description="1-based indices of questions necessary to satisfy the criterion. Empty if NOT_COVERED.",
    )
    considered_question_indices: list[int] = Field(
        default_factory=list,
        description="1-based indices of every question examined while judging this criterion, including rejected/insufficient ones. Populated regardless of coverage verdict.",
    )
    redundant_question_indices: list[int] = Field(
        default_factory=list,
        description="1-based indices of redundant questions. Must be empty if essential_question_indices is empty.",
    )
    reasoning: str = Field(
        description="Concise final summary explaining why the criterion is FULL or NOT_COVERED.",
    )
    failure_rule: str | None = Field(
        default=None,
        description="The primary rule justifying NOT_COVERED (e.g. 'Rule 1', 'Rule 5', 'Rule 8', 'Rule 10'). Must be None if coverage is FULL.",
    )
    coverage: Literal["FULL", "NOT_COVERED"] = Field(
        description="Strictly binary coverage verdict. Evaluated LAST after complete clinical analysis.",
    )


class _Node1JudgeResult(BaseModel):
    criteria_assessment: list[_CriterionAssessment] = Field(default_factory=list)


_WHITELIST_PATTERNS = [
    re.compile(r"\bnpi\b", re.IGNORECASE),
    re.compile(r"\bdob\b", re.IGNORECASE),
    re.compile(r"\bdate\s+of\s+birth\b", re.IGNORECASE),
    re.compile(r"\bpatient\s+name\b", re.IGNORECASE),
    re.compile(r"\bdate\s+of\s+service\b", re.IGNORECASE),
    re.compile(r"\binsurance\s+(id|number|member)\b", re.IGNORECASE),
    re.compile(r"\bphysician\s+name\b", re.IGNORECASE),
    re.compile(r"\bprovider\s+name\b", re.IGNORECASE),
    re.compile(r"\bfax\s+number\b", re.IGNORECASE),
    re.compile(r"\bphone\s+number\b", re.IGNORECASE),
]

_BETA = 2


def _compute_node1_metrics(
    assessments: list[_CriterionAssessment],
    questionnaire: list[dict],
) -> dict:
    whitelisted = {i for i, q in enumerate(questionnaire) if any(p.search(q["value"]) for p in _WHITELIST_PATTERNS)}
    total_criteria = len(assessments)
    total_clinical_q = len(questionnaire) - len(whitelisted)

    full_count = sum(1 for c in assessments if c.coverage == "FULL")
    nc_count = sum(1 for c in assessments if c.coverage == "NOT_COVERED")
    coverage_recall = full_count / total_criteria if total_criteria else 0.0

    essential_set: set[int] = set()
    for c in assessments:
        essential_set.update(idx - 1 for idx in c.essential_question_indices)
    essential_set -= whitelisted
    efficiency_precision = len(essential_set) / total_clinical_q if total_clinical_q else 0.0

    beta_sq = _BETA ** 2
    denom = beta_sq * efficiency_precision + coverage_recall
    f_beta = (1 + beta_sq) * (efficiency_precision * coverage_recall) / denom if denom > 0 else 0.0

    full_ids = tuple(sorted(c.criterion_id for c in assessments if c.coverage == "FULL"))
    fingerprint = str(("FULL", full_ids))

    return {
        "total_criteria": total_criteria,
        "full": full_count,
        "not_covered": nc_count,
        "total_clinical_q": total_clinical_q,
        "whitelisted": len(whitelisted),
        "essential_q": len(essential_set),
        "coverage_recall": round(coverage_recall, 4),
        "efficiency_precision": round(efficiency_precision, 4),
        "f_beta": round(f_beta, 4),
        "fingerprint": fingerprint,
    }


# ── Metric ────────────────────────────────────────────────────────────────────

class Node1QuestionnaireMetric(BaseLLMMetric):
    """Node 1: evaluates questionnaire coverage of GT clinical criteria.

    Emits coverage_recall, efficiency_precision, and f_beta (beta=2).
    The ``node1_result`` detail blob must be passed as input to
    ``maf.node2_qa_extraction`` to continue the evaluation pipeline.
    """

    name: str = "maf.node1_questionnaire"
    display_name: str = "MAF Node 1: Questionnaire Coverage"
    description: str = (
        "Evaluates whether an AI-generated medical intake questionnaire covers every GT clinical criterion. "
        "Returns coverage_recall, efficiency_precision, and f_beta. "
        "The node1_result detail blob must be forwarded to maf.node2_qa_extraction. Requires LLM."
    )

    class Config(BaseModel):
        model: str = Field(description="The LLM deployment name for evaluation.")

    class Input(BaseModel):
        gt_criteria: Annotated[
            list[dict],
            Field(description="List of GT criteria dicts with 'id', 'description', 'expected_value' fields. Only C*/EX* criteria are evaluated."),
        ]
        questionnaire: Annotated[
            list[dict],
            Field(description="List of generated question dicts with 'value', 'options', optional 'response_type'."),
        ]
        service_context: Annotated[
            dict | None,
            Field(default=None, description="Optional dict with 'service_code' and/or 'service_description' for context."),
        ] = None
        policy_text: Annotated[
            str | None,
            Field(default=None, description="Optional policy text (up to 30k chars used). Enables Rule 8 policy cross-check."),
        ] = None

    class Output(BaseModel):
        coverage_recall: Annotated[
            MetricOutputField | MetricError,
            Field(discriminator="type", description="Fraction of GT criteria with FULL coverage (0–1)."),
        ]
        efficiency_precision: Annotated[
            MetricOutputField | MetricError,
            Field(discriminator="type", description="Fraction of clinical questions that are essential (0–1)."),
        ]
        f_beta: Annotated[
            MetricOutputField | MetricError,
            Field(discriminator="type", description="F-beta (beta=2, recall-weighted) combining recall and precision (0–1)."),
        ]

    examples = [
        MetricExample(
            name="perfect coverage",
            description="All criteria fully covered, all questions essential",
            config=Config(model="gemini-3.5-flash"),
            input=Input(
                gt_criteria=[{"id": "C1", "description": "Patient age >= 18", "expected_value": "TRUE"}],
                questionnaire=[{"value": "Is the patient 18 or older?", "options": ["Yes", "No"]}],
            ),
            expected_output=Output(
                coverage_recall=MetricOutputField(value=1.0),
                efficiency_precision=MetricOutputField(value=1.0),
                f_beta=MetricOutputField(value=1.0),
            ),
        ),
    ]

    def __init__(self, dial_factory: DialFactory, settings: CommonGroupSettings):
        super().__init__(dial_factory, settings)

    async def evaluate_async(self, config: Config, input: Input) -> Output:
        try:
            # Only standard C*/EX* criteria are evaluated by Node 1 (matches case_runner.py)
            node1_criteria = [c for c in input.gt_criteria if c.get("id") and c["id"][0] in ("C", "E")]

            if not node1_criteria:
                return self.Output(
                    coverage_recall=MetricOutputField(value=0.0, details={"reason": "no C*/EX* criteria in gt_criteria"}),
                    efficiency_precision=MetricOutputField(value=0.0, details={"reason": "no C*/EX* criteria in gt_criteria"}),
                    f_beta=MetricOutputField(value=0.0, details={"reason": "no C*/EX* criteria in gt_criteria"}),
                )

            if not input.questionnaire:
                return self.Output(
                    coverage_recall=MetricOutputField(value=0.0, details={"reason": "empty questionnaire"}),
                    efficiency_precision=MetricOutputField(value=0.0, details={"reason": "empty questionnaire"}),
                    f_beta=MetricOutputField(value=0.0, details={"reason": "empty questionnaire"}),
                )

            chain = self._dial_factory.create_llm_with_schema(config.model, _Node1JudgeResult)
            messages = [
                SystemMessage(content=_NODE1_SYSTEM),
                HumanMessage(content=_node1_user_prompt(
                    node1_criteria,
                    input.questionnaire,
                    service_context=input.service_context,
                    policy_text=input.policy_text,
                )),
            ]
            llm_result: _Node1JudgeResult = await chain.ainvoke(messages)

            # Filter to only standard criteria (anti-hallucination)
            llm_result.criteria_assessment = [
                a for a in llm_result.criteria_assessment
                if a.criterion_id and a.criterion_id[0] in ("C", "E")
            ]

            metrics = _compute_node1_metrics(llm_result.criteria_assessment, input.questionnaire)

            # Serialize node1_result for downstream Node 2 metric
            node1_result_blob = [a.model_dump() for a in llm_result.criteria_assessment]

            details = {
                **metrics,
                "node1_result": node1_result_blob,
            }

            return self.Output(
                coverage_recall=MetricOutputField(value=metrics["coverage_recall"], details=details),
                efficiency_precision=MetricOutputField(value=metrics["efficiency_precision"], details=details),
                f_beta=MetricOutputField(value=metrics["f_beta"], details=details),
            )
        except Exception as e:
            err = MetricError(message=str(e))
            return self.Output(coverage_recall=err, efficiency_precision=err, f_beta=err)
