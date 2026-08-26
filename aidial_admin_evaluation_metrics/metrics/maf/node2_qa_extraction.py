"""MAF Node 2 + 2b: Q&A extraction accuracy metric.

Evaluates whether the AI Q&A extractor correctly resolves each GT criterion's
clinical state from clinical documents.

Node 2b (partial coverage resolver) is invoked automatically for every
NOT_COVERED criterion (those N1 failed to cover) before returning, so the
output ``node2_result`` blob already contains resolved_state for all criteria
and is ready for maf.node3_decision.

Outputs:
  n2_accuracy       — accuracy over N1-covered criteria only
  n2_accuracy_all   — accuracy over all criteria (penalises N1 misses)

The ``node2_result`` detail blob must be forwarded to maf.node3_decision.
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

_DOC_GAP_PATTERN = re.compile(r"no documentation .*(?:in|within) the supplied excerpts", re.IGNORECASE)
_ALL_ABSENT_PATTERN = re.compile(r"\bnone of the above\b", re.IGNORECASE)
_NA_BYPASS_PATTERN = re.compile(r"^n/a\s*-", re.IGNORECASE)


def _classify_answer_category(answer: str) -> str:
    try:
        parsed = json.loads(answer)
        options = parsed if isinstance(parsed, list) else [answer]
    except (TypeError, ValueError):
        options = [answer]

    texts = [str(o).strip() for o in options if str(o).strip()]
    if not texts:
        return "OTHER"
    if any(_NA_BYPASS_PATTERN.match(t) for t in texts):
        return "NA_BYPASS"
    if any(_ALL_ABSENT_PATTERN.search(t) for t in texts):
        return "ALL_ABSENT"
    if any(_DOC_GAP_PATTERN.search(t) for t in texts):
        return "DOC_GAP"
    return "OTHER"


def _normalize_actual_qa(items: list[dict]) -> list[dict]:
    out = []
    for item in items:
        answer = item.get("ai_answer") or item.get("value") or item.get("result", "")
        cit_raw = item.get("citation", {})
        citations = []
        if isinstance(cit_raw, dict):
            for cit_list in cit_raw.values():
                for c in cit_list or []:
                    if isinstance(c, dict):
                        citations.append(c.get("citation", ""))
        elif isinstance(cit_raw, list):
            citations = [str(c) for c in cit_raw]
        out.append({
            "question": item.get("question", ""),
            "answer": str(answer),
            "rationale": item.get("rationale", ""),
            "citations": citations or item.get("citations", []),
        })
    return out


_NODE2_SYSTEM = """
You are a strict clinical compliance extraction evaluator assessing document-based Q&A extraction in a prior authorization context.
Your task is to check whether the extracted Q&A output correctly resolves the value of each Ground Truth (GT) criterion, adhering strictly to the No-Inference Rule.

For each GT criterion you are given:
- A description of what the criterion assesses.
- An expected_value — the specific value the clinical documents should yield (e.g. 'TRUE', 'FALSE', 'UNKNOWN').
- A mapped_qa list — the specific Q&A entries pre-selected as covering this criterion. Evaluate them as a group.
  Each Q&A entry contains: the question text, the extracted answer, and a rationale field with the AI extractor's
  reasoning for that answer. You may use the rationale as additional context to understand the evidence behind the
  answer, but the answer itself remains the primary signal for your evaluation.

**CRITICAL MANDATE: THE NO-INFERENCE RULE (Strict Grounding)**
You must act as a strict compiler rather than a decision-making doctor. Do not make any clinical leaps, assumptions, or clinical inferences to fill in gaps.

**CRITICAL MANDATE ON QUESTION INDEXING (1-BASED):**
All question indices provided in the input objects under `<Criteria_With_Mapped_QA>` (under the key `"index"`) are 1-based (i.e., Question 1 has `"index": 1`, Question 8 has `"index": 8`, etc.). In your `reasoning` explanation and in your output fields (specifically `relevant_qa_indices`), you MUST strictly reference these EXACT 1-based indices. Do NOT convert them to 0-based indices or offset them (e.g., if the question index in mapped_qa is 8, always refer to it as "Question 8", NOT "Question 7").
1. If the Q&A represents that a condition is "unclear", "partially treated", "not ruled out", or "distally unresolved", or if the files/answers do not formally verify of a condition's active status, its actual clinical state MUST be mapped to **UNKNOWN**.
2. Do NOT assume that a failure to do/complete a rule-out check (such as Q&A answer "No" to "Have non-physiologic modifiers of pain been ruled out?") is equivalent to a definitive "FALSE" or "FAIL". If a clinical modifier or condition (e.g. severe diabetic neuropathy) is present, and has not been ruled out (Q&A "No" to whether it was ruled out), the clinical status of this mimicking or modifying effect is **UNKNOWN**.
3. Direct Q&A mapping only applies for standard, unconditional, and explicit clinical states. The mapping rules below apply to all answer types (binary Yes/No AND multi-option verbatim strings).

**PRECEDENCE AND DETERMINISTIC DISPATCH:**
- Rule 5's Semantic Compatibility exception takes precedence over the general No-Inference framing above whenever `expected_value` is **UNKNOWN** and the answer indicates absence, silence, or N/A — apply Rule 5 before concluding INCORRECT under the preamble mandates.
- Each Q&A entry's `category` field (`NA_BYPASS`, `ALL_ABSENT`, `DOC_GAP`, or `OTHER`) is computed deterministically upstream from the raw answer text. Treat it as authoritative for initial dispatch into the branches of Rule 3 below — do not re-classify the raw answer text yourself. Only fall back to your own reading of the answer when `category` is `OTHER`.

Rules for Semantic Confirmation and logical matching:

1. **Strict Reliance on Answers (No Rationale):**
   - You must evaluate coverage and correctness using ONLY the provided question text and the corresponding answer. Do NOT assume, infer, or hallucinate any external rationale, background context, or unprovided information.
   - Focus entirely on whether the *actual answers* directly and unambiguously resolve the criterion to the `expected_value`.

2. **The Conditional/Procedural Question Fallacy (Critical Safety Rule):**
   - For a criterion checking the presence of a condition (e.g., "Clinically significant depression" where expected_value is `FALSE`), if the mapped question is conditional/procedural (e.g., "If the patient has depression, has a case-by-case review been performed?"):
   - An answer of "No" (meaning the review was NOT performed) does **NOT** semantically confirm the expected_value of `FALSE` (that the patient is free of depression). A "No" simply means no review occurred—which could mean either the patient is healthy, OR the patient is sick but neglected.
   - Therefore, a "No" answer to such conditional questions is highly ambiguous and must be evaluated as **INCORRECT** (cannot confirm `FALSE`), unless another unconditional question in the group explicitly confirms that the patient does not have the condition.

3. **Resolve actual clinical state (resolved_state):**
   - In addition to evaluating the match against expected_value, you must determine the actual clinical state of this criterion as resolved from the Q&A answers.
   - **Multiple Select Answer Mapping (all structured questions):** All structured questions use Multiple select. The `ai_answer` field for Multiple select responses is a **JSON-array string** (e.g., `'["Option A", "Option C"]'`) — parse it as an array to obtain the list of selected and/or agent-generated options. Each criterion maps to one or more specific option texts. Resolve as follows:
     - If the parsed array includes a **provided option** whose text precisely represents this criterion's required clinical state, resolve to **TRUE** (the condition is present/documented).
     - If the parsed array includes an **agent-generated custom option** (text not from the question's provided options list) that directly documents the criterion's required clinical state (a concise clinical fact extracted from the record), resolve to **TRUE** using semantic evaluation.
     - If the parsed array includes `"Unknown"` or a documentation-gap option (e.g., `"No documentation in the supplied excerpts of [element]"`), the criterion's status cannot be confirmed — resolve to **UNKNOWN**.
     - If the parsed array does NOT include any option (provided or custom) representing this criterion's clinical state, resolve to **FALSE** (the condition is absent/not documented).
     - If the parsed array includes "None of the above" or an equivalent all-absent selection, resolve ALL associated criteria to **FALSE**.
     - If a selected option is "N/A - [specific bypass reason]", the criterion's pathway is inactive for this patient — resolve to **UNKNOWN**.
     - If the answer is an empty array or non-informative, resolve to **UNKNOWN**.
     - Note: each option is independent — the resolution of one criterion is entirely unaffected by which other options are selected or not selected.
   - **Free text / legacy binary answers:** For Free text questions (where no options list is present), `ai_answer` is a plain string (never a bare "Yes" or "No"). Resolve using direct semantic mapping: a concise documented clinical fact → **TRUE** or **FALSE** based on whether it affirms or negates the criterion; `"N/A"` or a documentation-gap statement → **UNKNOWN**; ambiguous or empty → **UNKNOWN**, with the exceptions below.
     - **CRITICAL EXCEPTION (Semantic Compatibility of FALSE and UNKNOWN):** If the expected_value is **UNKNOWN** because of a silent record/lack of documentation, but the Q&A answer is **"No"** (legacy binary) or a documentation-gap custom option (e.g., `"No documentation in the supplied excerpts of [element]"`), you MUST resolve the state to **UNKNOWN** instead of FALSE to maintain correct alignment, and mark this match as **CORRECT**!
     - **CRITICAL NEGATION / RULE-OUT EXCEPTION (Negated Clinical States):** If the mapped Q&A question is inherently a negation, exclusion, or rule-out question (e.g., asking whether a mimicking condition has been "ruled out", "excluded", "absent", or "free of"), answering **"Yes"** (meaning the mimic has indeed been ruled out/excluded/absent) resolves the clinical presence check of that condition to **"FALSE"** rather than "TRUE" (since the mimic is clinically absent/false). Answering **"No"** (not ruled out / not excluded / present) resolves to **"UNKNOWN"** (under the No-Inference exception, as it remains completely unverified).
   - **Ambiguity, Rule-out & Fallacy Exception:** Output "UNKNOWN" if the files are completely silent on the topic, the status is unrecorded, the Q&A layout falls under the **Conditional/Procedural Question Fallacy** (Section 2), or a rule-out check indicates that a condition is not ruled out.

4. **Structured Reasoning Chain (CoT):**
   - In your internal processing, always execute these distinct steps:
     Step 1 (Fact Extraction): Extract the exact facts or Q&A answers.
     Step 2 (Apply No-Inference Rule): Assess whether the status of this clinical state is confirmed, denied, or unresolved.
     Step 3 (Schema Mapping): Map directly to "TRUE", "FALSE", or "UNKNOWN" based on Step 2 and the mapping rules in Rule 3.
     Step 4 (Verify Match): Compare your mapped resolved_state to the expected_value. Do not assume or invent clinical outcomes.

5. **Semantic Compatibility of FALSE and UNKNOWN (Silence in the Chart):**
   - In clinical claims and prior authorization datasets, a lack of documentation or silence in the chart for a specific criterion/disease is often expected to be **UNKNOWN** in the Ground Truth (since the text is silent).
   - However, the QA extractor often answers **"No"** (or selects a non-compliant option) to questions about the presence of these same silent conditions/criteria (indicating no clinical evidence of the condition exists in the documentation).
   - **RULE:** You MUST treat an expected_value of **UNKNOWN** and an actual Q&A answer of **"No"** (legacy binary), a documentation-gap custom option (e.g., `"No documentation in the supplied excerpts of [element]"`), `"Unknown"`, or an "N/A - [bypass reason]" option as **SEMANTICALLY COMPATIBLE** (evaluate as **CORRECT** / match=true) whenever the answer is driven by a lack of clinical evidence or the pathway is inactive. In such cases, map the `resolved_state` to **UNKNOWN** to match the expected_value.
     - **STRICT ANTI-PASSTHROUGH FOR SELF-CONTRADICTING AFFIRMATIVE ANSWERS:** This semantic compatibility ONLY applies when the answer indicates absence, silence, or inapplicability. It NEVER applies when the Q&A selects a clearly affirmative/active clinical option. If a Q&A selects an affirmative option (e.g., "Yes" or a verbatim option describing an active clinical state), but the expected_value is **UNKNOWN**, this is always **INCORRECT** (mismatch).
     - **GROUP-CONTRADICTION PENALTY (Critical Parent-Child Logic Rule):** If a mapped Q&A group contains a broad summary question ("parent") answered affirmatively (e.g., "Yes" or a compliant verbatim option), but is accompanied by sub-criteria/documentation questions ("child" elements) answered negatively (e.g., "No" meaning mandatory documentation is missing), **the entire group's resolved state must be mapped to TRUE** (because of the affirmative parent summary answer). Since the parent question's affirmative answer was invalid and contradicted the child checks, this is a faulty extraction and MUST be marked as **INCORRECT**!

**Edge-Case Few-Shot Examples:**
- Example A (Ambiguous Rule-out):
  - GT Criterion: EX2 "Non-physiologic modifiers of pain must be ruled out" (expected_value: UNKNOWN).
  - Q&A: "Have non-physiologic modifiers of pain been ruled out prior to surgery?" -> "No"
  - Evaluation CoT: Step 1: Q&A answer is 'No' (not ruled out). Step 2/3: Applying no-inference rule, the neuropathy exists but has not been ruled out — its clinical outcome is unresolved. resolved_state = 'UNKNOWN'. Step 4: 'UNKNOWN' matches expected_value 'UNKNOWN'.
  - Tag: CORRECT. resolved_state: UNKNOWN.

- Example B (Presence check with silent record):
  - GT Criterion: C3 "Patient has failed conservative treatment" (expected_value: TRUE).
  - Q&A: "Has conservative therapy failed?" -> "No"
  - Evaluation CoT: Step 1: Q&A answered 'No'. Step 2/3: No clinical leap — 'No' maps to FALSE. Step 4: 'FALSE' vs expected 'TRUE' = mismatch.
  - Tag: INCORRECT. resolved_state: FALSE.

- Example C (Silent Record / Lack of Documentation mapped to UNKNOWN):
  - GT Criterion: "History of a myocardial infarction or stroke within the preceding year" (expected_value: UNKNOWN).
  - Q&A: "Is there a history of MI/stroke within the preceding year?" -> "No"
  - Evaluation CoT: Step 1: Q&A answered 'No' due to no documentation. Step 2/3: Under Semantic Compatibility, 'No' from a silent record is UNKNOWN. Step 4: 'UNKNOWN' matches expected 'UNKNOWN'.
  - Tag: CORRECT. resolved_state: UNKNOWN.

- Example D (Multi-option answer: N/A inactive pathway):
  - GT Criterion: C5 "Ab interno stent is in conjunction with cataract surgery" (expected_value: UNKNOWN).
  - Q&A: "Is the ab interno stent being requested in conjunction with cataract surgery?" -> "N/A - Request is for an Ab Externo Aqueous Shunt"
  - Evaluation CoT: Step 1: Answer is an explicit N/A option indicating this pathway is inactive. Step 2/3: The criterion's pathway does not apply to this patient — resolved_state = UNKNOWN. Step 4: 'UNKNOWN' matches expected 'UNKNOWN'.
  - Tag: CORRECT. resolved_state: UNKNOWN.

- Example E (Multi-option answer: verbatim compliant option):
  - GT Criterion: C3 "Patient has Moderate open-angle glaucoma" (expected_value: TRUE).
  - Q&A: "What is the severity of the patient's open-angle glaucoma?" -> "Moderate: definite optic disc or RNFL... consistent with glaucoma and visual field abnormalities in one hemifield..."
  - Evaluation CoT: Step 1: Answer is the verbatim Moderate option. Step 2/3: Selecting the Moderate option unambiguously confirms this severity level is present. resolved_state = TRUE. Step 4: 'TRUE' matches expected 'TRUE'.
  - Tag: CORRECT. resolved_state: TRUE.

- Example F (Self-Contradicting affirmative answer for a silent record):
  - GT Criterion: "Prescription of continuous home oxygen therapy for at least 15 hours per day" (expected_value: UNKNOWN).
  - Q&A: "Is there a documented prescription of home oxygen therapy for at least 15 hours per day?" -> "Yes"
  - Evaluation CoT: Step 1: Q&A answered 'Yes'. Step 2/3: 'Yes' resolves to TRUE. Step 4: 'TRUE' vs expected 'UNKNOWN' = mismatch.
  - Tag: INCORRECT. resolved_state: TRUE.

- Example G (Group Contradiction / Summary Yes with Missing sub-components):
  - GT Criterion: "Failure of conservative treatment for a minimum of six weeks ... HEP requires documentation" (expected_value: UNKNOWN).
  - Q&A Group: Q8 "Has the patient failed conservative treatment for six weeks?" -> "Yes"; Q13 "Is there documentation of HEP?" -> "No"; Q14 "Is there follow-up documentation?" -> "No"
  - Evaluation CoT: Step 1: Q8='Yes', Q13='No', Q14='No'. Step 2/3: Group-Contradiction Penalty applies — parent 'Yes' drives resolved_state to TRUE despite missing child documentation. Step 4: 'TRUE' vs expected 'UNKNOWN' = mismatch.
  - Tag: INCORRECT. resolved_state: TRUE.

Assign a tag:
- CORRECT: combined answers confirm expected_value.
- INCORRECT: combined answers contradict expected_value.

Note: You are only sent criteria that have relevant mapped Q&A pairs, so the "NOT_FOUND" tag is not applicable and must not be used.

Return one entry per GT criterion.
"""


def _node2_user_prompt(criterion_groups: list[dict]) -> str:
    cleaned_groups = []
    for g in criterion_groups:
        cleaned_qa = []
        for qa in g.get("mapped_qa", []):
            entry: dict = {
                "index": qa["index"],
                "question": qa["question"],
                "answer": qa["answer"],
                "category": qa.get("category", "OTHER"),
            }
            if qa.get("rationale"):
                entry["rationale"] = qa["rationale"]
            cleaned_qa.append(entry)
        cleaned_groups.append({
            "id": g["id"],
            "description": g["description"],
            "expected_value": g["expected_value"],
            "mapped_qa": cleaned_qa,
        })
    return f"""
<Criteria_With_Mapped_QA>
{json.dumps(cleaned_groups, indent=2)}
</Criteria_With_Mapped_QA>
"""


_NODE2B_SYSTEM = """
You are a clinical evidence recovery evaluator. Your task is to infer the clinical state of criteria
that were NOT fully covered by the AI-generated questionnaire, using any Q&A evidence that is even
partially relevant — including compound or structurally flawed questions.

You are given:
1. NOT_COVERED_CRITERIA: criteria the questionnaire did not fully cover, with the Node 1 auditor's
   explanation of WHY coverage failed (e.g., compound question structure, ambiguous "No" resolution).
   Each criterion also lists the Q&A indices Node 1 considered partially relevant.
2. ALL_QA_PAIRS: the full set of Q&A pairs (indexed 1-based) with question text, extracted answer,
   and — when available — the extractor's rationale explaining the evidence behind the answer.

EVALUATION RULES:

**DETERMINISTIC DISPATCH:** Each Q&A entry's `category` field (`NA_BYPASS`, `ALL_ABSENT`, `DOC_GAP`, or `OTHER`) is computed deterministically upstream from the raw answer text — treat it as authoritative when applying Rules 1 and 2 below (e.g. `category == "NA_BYPASS"` maps directly to the N/A-bypass branch of Rule 2). Only fall back to your own reading of the answer text when `category` is `OTHER`.

1. Affirmative Answer Resolution (Checked Multiple Select or explicit affirmative):
   - All structured questions use Multiple select. For each criterion, check whether the criterion's specific option appears among the checked answers.
   - If the criterion's option IS checked, resolve to TRUE.
   - If the criterion's option is NOT checked (absent from the checked list), resolve to FALSE. Each checkbox is independent — do not let other checked or unchecked options affect this criterion's resolution.
   - For legacy or free-text answers that are clearly affirmative (e.g., "Yes"), resolve to TRUE.

2. Negative or Ambiguous Answer (Cannot Reliably Decompose):
   - If the answer is "No", ambiguous, or conditional (binary questions), you MUST resolve each
     affected criterion to UNKNOWN. You cannot determine which sub-condition failed.
   - If the checked option is "N/A - [bypass reason]" or similar, the
     criterion's pathway is inactive for this patient — resolve to UNKNOWN.
   - If the selected option represents a non-compliant state (e.g., selecting a severity that does
     not meet the criterion's threshold), resolve to FALSE only when this is clear and unambiguous.
   - Do NOT infer a specific sub-condition's failure from a compound "No" or N/A.

3. Rationale-as-Evidence (Lenient Grounding):
   - Unlike Node 2 (which evaluates only the answer), you MAY use explicit clinical facts stated in
     the rationale field as direct evidence (e.g., "patient is 90 years old", "MRI confirmed diagnosis
     of X", "no prior treatment documented"). This includes rationale-based resolution of the selected
     option's meaning in context.
   - Apply No-Inference only when the rationale is genuinely ambiguous or contains no relevant facts.

4. No-Inference Floor:
   - Do not make clinical leaps beyond what the Q&A text and rationale explicitly state.
   - If no Q&A pair (or its rationale) contains relevant clinical facts for a criterion, set
     resolved_state to UNKNOWN.

5. Index references:
   - All indices are 1-based exactly as shown in ALL_QA_PAIRS. Reference only indices that exist.
   - relevant_qa_indices must only contain indices you actually used as evidence.

6. Default: UNKNOWN.
   - If no relevant Q&A or rationale exists for a criterion, resolved_state = UNKNOWN,
     relevant_qa_indices = [], and reasoning explains why no evidence was found.

FEW-SHOT EXAMPLES:

Example A — Compound "Yes" resolves multiple criteria:
  Criterion C1: "Patient has a confirmed diagnosis of osteoporosis" (expected: TRUE)
  Criterion C2: "Patient is postmenopausal" (expected: TRUE)
  Q3: "Is the patient a postmenopausal individual with a confirmed diagnosis of osteoporosis?" → "Yes"
  → C1: resolved_state=TRUE, relevant_qa_indices=[3], reasoning="Compound question Q3 answered Yes, unambiguously confirming osteoporosis diagnosis."
  → C2: resolved_state=TRUE, relevant_qa_indices=[3], reasoning="Compound question Q3 answered Yes, unambiguously confirming postmenopausal status."

Example B — Compound "No" cannot be decomposed:
  Criterion C1: "Patient has failed physical therapy" (expected: TRUE)
  Criterion C2: "Patient has failed chiropractic treatment" (expected: TRUE)
  Q5: "Has the patient failed both physical therapy and chiropractic treatment?" → "No"
  → C1: resolved_state=UNKNOWN, relevant_qa_indices=[5], reasoning="Compound question Q5 answered No — cannot determine which treatment was not failed."
  → C2: resolved_state=UNKNOWN, relevant_qa_indices=[5], reasoning="Compound question Q5 answered No — cannot determine which treatment was not failed."

Example C — Rationale contains explicit clinical fact:
  Criterion C4: "Patient age ≥ 18 years" (expected: TRUE)
  Q7: "Is the patient a pediatric individual?" → "No"
  Rationale: "The patient is a 72-year-old adult; pediatric status does not apply."
  → C4: resolved_state=TRUE, relevant_qa_indices=[7], reasoning="Q7 rationale explicitly states patient is 72 years old, confirming age ≥ 18."

Example D — Multi-option verbatim answer resolves severity criterion:
  Criterion C3: "Patient has Moderate open-angle glaucoma" (expected: TRUE)
  Q4: "What is the severity of the patient's open-angle glaucoma?" → "Moderate: definite optic disc or RNFL... visual field abnormalities in one hemifield..."
  → C3: resolved_state=TRUE, relevant_qa_indices=[4], reasoning="Q4 selected the verbatim Moderate option, directly confirming the criterion's severity requirement."

Example E — Multi-option N/A renders pathway inactive:
  Criterion C5: "Ab interno stent is in conjunction with cataract surgery" (expected: UNKNOWN)
  Q5: "Is the ab interno stent being requested in conjunction with cataract surgery?" → "N/A - Request is for an Ab Externo Aqueous Shunt"
  → C5: resolved_state=UNKNOWN, relevant_qa_indices=[5], reasoning="Q5 selected N/A — criterion is inactive for this patient's request type."

Return one entry per criterion listed in NOT_COVERED_CRITERIA. Every criterion must appear in the output.
"""


def _node2b_user_prompt(
    not_covered_groups: list[dict],
    all_qa: list[dict],
    n1_assessments_by_id: dict[str, dict],
) -> str:
    criteria_list = []
    for g in not_covered_groups:
        cid = g["id"]
        assessment = n1_assessments_by_id.get(cid)
        entry: dict = {
            "id": cid,
            "description": g["description"],
            "expected_value": g["expected_value"],
        }
        if assessment:
            entry["node1_coverage_reasoning"] = assessment.get("reasoning", "")
            considered = assessment.get("considered_question_indices", [])
            if considered:
                entry["node1_considered_indices"] = considered
        criteria_list.append(entry)

    qa_list = []
    for i, qa in enumerate(all_qa, start=1):
        entry = {
            "index": i,
            "question": qa["question"],
            "answer": qa["answer"],
            "category": _classify_answer_category(qa["answer"]),
        }
        if qa.get("rationale"):
            entry["rationale"] = qa["rationale"]
        qa_list.append(entry)

    return f"""
<NOT_COVERED_CRITERIA>
{json.dumps(criteria_list, indent=2)}
</NOT_COVERED_CRITERIA>

<ALL_QA_PAIRS>
{json.dumps(qa_list, indent=2)}
</ALL_QA_PAIRS>
"""


# ── LLM output schemas ────────────────────────────────────────────────────────

class _QACriterionEval(BaseModel):
    criterion_id: str
    criterion_description: str
    expected_value: str
    relevant_qa_indices: list[int] = Field(
        default_factory=list,
        description="1-based indices of Q&A pairs identified as relevant to this criterion.",
    )
    actual_answers: list[str] = Field(
        default_factory=list,
        description="The answers from the relevant Q&A pairs.",
    )
    value_match: bool = Field(
        description="True if the relevant answers, taken together, semantically confirm the expected_value."
    )
    tag: Literal["CORRECT", "INCORRECT", "NOT_FOUND"] = Field(
        description=(
            "CORRECT: combined answers confirm expected_value. "
            "INCORRECT: combined answers contradict expected_value. "
            "NOT_FOUND: no Q&A pairs are relevant to this criterion."
        )
    )
    resolved_state: str = Field(
        description="The actual clinical state of this criterion as resolved from the Q&A answers and clinical context. Must be exactly 'TRUE', 'FALSE', or 'UNKNOWN'."
    )
    reasoning: str
    resolved_by: Literal["node2", "node2b"] | None = Field(
        default=None,
        description="Provenance of this evaluation: 'node2' if resolved by the main Q&A judge, 'node2b' if rescued by the partial resolver. None if not yet resolved by either.",
    )


class _Node2JudgeResult(BaseModel):
    criteria_evaluations: list[_QACriterionEval] = Field(default_factory=list)


class _PartialCriterionEval(BaseModel):
    criterion_id: str
    resolved_state: Literal["TRUE", "FALSE", "UNKNOWN"]
    relevant_qa_indices: list[int] = Field(
        default_factory=list,
        description="1-based indices of Q&A pairs that provided evidence for this resolution. Empty if none found.",
    )
    reasoning: str = Field(
        description="Explanation of why the resolved_state was determined, citing specific Q&A content."
    )


class _Node2bJudgeResult(BaseModel):
    partial_evaluations: list[_PartialCriterionEval] = Field(default_factory=list)


def _build_criterion_qa_groups(
    n1_assessments: list[dict],
    actual_qa: list[dict],
    gt_by_id: dict[str, dict],
) -> list[dict]:
    groups = []
    for a in n1_assessments:
        cid = a["criterion_id"]
        gt = gt_by_id.get(cid, {})
        mapped_qa = [
            {"index": i, "category": _classify_answer_category(actual_qa[i - 1]["answer"]), **actual_qa[i - 1]}
            for i in a.get("essential_question_indices", [])
            if 1 <= i <= len(actual_qa)
        ]
        groups.append({
            "id": cid,
            "description": gt.get("description", a.get("criterion_description", "")),
            "expected_value": gt.get("expected_value", ""),
            "mapped_qa": mapped_qa,
        })
    return groups


def _apply_partial_resolver_results(
    n2_evals: list[_QACriterionEval],
    criterion_groups: list[dict],
    partial_evals: list[_PartialCriterionEval],
    all_qa: list[dict],
    n1_covered_ids: set[str],
) -> tuple[list[_QACriterionEval], list[dict]]:
    partial_by_id = {e.criterion_id: e for e in partial_evals}

    for ev in n2_evals:
        if ev.criterion_id in n1_covered_ids:
            continue
        partial = partial_by_id.get(ev.criterion_id)
        if partial is None:
            continue
        ev.resolved_state = partial.resolved_state
        ev.reasoning = f"[Partial Resolver] {partial.reasoning}"
        ev.resolved_by = "node2b"

    groups_by_id = {g["id"]: g for g in criterion_groups}
    for cid, partial in partial_by_id.items():
        if cid in n1_covered_ids:
            continue
        group = groups_by_id.get(cid)
        if group is None:
            continue
        existing_indices = {qa["index"] for qa in group["mapped_qa"]}
        for i in partial.relevant_qa_indices:
            if i < 1 or i > len(all_qa):
                continue
            if i not in existing_indices:
                group["mapped_qa"].append({"index": i, **all_qa[i - 1]})
                existing_indices.add(i)
        group["mapped_qa"].sort(key=lambda qa: qa["index"])

    return n2_evals, criterion_groups


# ── Metric ────────────────────────────────────────────────────────────────────

class Node2QAExtractionMetric(BaseLLMMetric):
    """Node 2 + 2b: evaluates Q&A extraction accuracy across all GT criteria.

    Node 2 evaluates criteria whose questions were covered (FULL) by Node 1.
    Node 2b (partial coverage resolver) is run automatically for NOT_COVERED
    criteria, recovering resolved_state from any partially-relevant Q&A evidence.

    Returns n2_accuracy (over N1-covered criteria) and n2_accuracy_all (over
    all criteria). The ``node2_result`` detail blob must be forwarded to
    ``maf.node3_decision``.
    """

    name: str = "maf.node2_qa_extraction"
    display_name: str = "MAF Node 2: Q&A Extraction Accuracy"
    description: str = (
        "Evaluates whether the Q&A extractor correctly resolves each GT criterion's clinical state. "
        "Node 2b (partial resolver) automatically handles NOT_COVERED criteria. "
        "Returns n2_accuracy (N1-covered criteria only) and n2_accuracy_all (all criteria). "
        "Requires LLM. Forward node2_result to maf.node3_decision."
    )

    class Config(BaseModel):
        model: str = Field(description="The LLM deployment name for evaluation.")

    class Input(BaseModel):
        gt_criteria: Annotated[
            list[dict],
            Field(description="Same gt_criteria list passed to Node 1 (all criteria, including non-C*/EX*)."),
        ]
        actual_qa: Annotated[
            list[dict],
            Field(description="Q&A pairs from the extractor. Each dict may have 'question', 'ai_answer'/'value'/'result', 'rationale', 'citations'."),
        ]
        node1_result: Annotated[
            list[dict],
            Field(description="The node1_result blob from maf.node1_questionnaire output details (list of criterion assessment dicts)."),
        ]

    class Output(BaseModel):
        n2_accuracy: Annotated[
            MetricOutputField | MetricError,
            Field(discriminator="type", description="Extraction accuracy over N1-covered criteria (0–1)."),
        ]
        n2_accuracy_all: Annotated[
            MetricOutputField | MetricError,
            Field(discriminator="type", description="Extraction accuracy over all criteria, penalising N1 misses (0–1)."),
        ]

    examples = [
        MetricExample(
            name="perfect extraction",
            description="Single covered criterion extracted correctly",
            config=Config(model="gemini-3.5-flash"),
            input=Input(
                gt_criteria=[{"id": "C1", "description": "Patient age >= 18", "expected_value": "TRUE"}],
                actual_qa=[{"question": "Is the patient 18 or older?", "value": "Yes"}],
                node1_result=[{
                    "criterion_id": "C1",
                    "criterion_description": "Patient age >= 18",
                    "coverage": "FULL",
                    "essential_question_indices": [1],
                    "considered_question_indices": [1],
                    "redundant_question_indices": [],
                    "extracted_options": [],
                    "clinical_scope_analysis": "",
                    "reasoning": "",
                    "failure_rule": None,
                }],
            ),
            expected_output=Output(
                n2_accuracy=MetricOutputField(value=1.0),
                n2_accuracy_all=MetricOutputField(value=1.0),
            ),
        ),
    ]

    def __init__(self, dial_factory: DialFactory, settings: CommonGroupSettings):
        super().__init__(dial_factory, settings)

    async def evaluate_async(self, config: Config, input: Input) -> Output:
        try:
            n1_assessments = input.node1_result
            gt_by_id = {c["id"]: c for c in input.gt_criteria if c.get("id")}
            # node1_2_criteria = [c for c in input.gt_criteria if c.get("id") and c["id"][0] in ("C", "E")]

            qa_normalized = _normalize_actual_qa(input.actual_qa)

            n1_covered_ids = {a["criterion_id"] for a in n1_assessments if a.get("coverage") == "FULL"}
            criterion_groups = _build_criterion_qa_groups(n1_assessments, qa_normalized, gt_by_id)

            # ── Node 2: LLM evaluation of covered criteria with mapped QA ──────
            covered_groups_with_qa = [
                g for g in criterion_groups
                if g["id"] in n1_covered_ids and g["mapped_qa"]
            ]

            if covered_groups_with_qa:
                chain = self._dial_factory.create_llm_with_schema(config.model, _Node2JudgeResult)
                messages = [
                    SystemMessage(content=_NODE2_SYSTEM),
                    HumanMessage(content=_node2_user_prompt(covered_groups_with_qa)),
                ]
                n2_llm: _Node2JudgeResult = await chain.ainvoke(messages)
            else:
                n2_llm = _Node2JudgeResult(criteria_evaluations=[])

            # Build synthetic NOT_FOUND evals for criteria not sent to LLM
            llm_ids = {g["id"] for g in covered_groups_with_qa}
            synthetic_evals = [
                _QACriterionEval(
                    criterion_id=g["id"],
                    criterion_description=g["description"],
                    expected_value=g["expected_value"],
                    relevant_qa_indices=[],
                    actual_answers=[],
                    value_match=False,
                    tag="NOT_FOUND",
                    resolved_state="UNKNOWN",
                    reasoning=(
                        "No relevant questions were generated in Node 1 to cover this criterion."
                        if g["id"] not in n1_covered_ids
                        else "Criterion was covered in Node 1 but no Q&A pairs were mapped to it."
                    ),
                )
                for g in criterion_groups
                if g["id"] not in llm_ids
            ]

            # Merge LLM + synthetic, preserving Node 1 order
            eval_map = {e.criterion_id: e for e in n2_llm.criteria_evaluations}
            for e in synthetic_evals:
                eval_map[e.criterion_id] = e

            ordered_evals: list[_QACriterionEval] = []
            for a in n1_assessments:
                cid = a["criterion_id"]
                ev = eval_map.get(cid)
                if ev:
                    ordered_evals.append(ev)
                else:
                    gt = gt_by_id.get(cid, {})
                    ordered_evals.append(_QACriterionEval(
                        criterion_id=cid,
                        criterion_description=gt.get("description", ""),
                        expected_value=gt.get("expected_value", ""),
                        relevant_qa_indices=[],
                        actual_answers=[],
                        value_match=False,
                        tag="NOT_FOUND",
                        resolved_state="UNKNOWN",
                        reasoning="No relevant questions were generated in Node 1 to cover this criterion.",
                    ))

            # Anti-hallucination: overwrite criterion text fields from GT
            for ev in ordered_evals:
                gt = gt_by_id.get(ev.criterion_id)
                if gt:
                    ev.criterion_description = gt["description"]
                    ev.expected_value = gt["expected_value"]

            # Deterministic tag override for LLM-evaluated criteria
            for ev in ordered_evals:
                if ev.criterion_id not in llm_ids:
                    continue
                ev.tag = "CORRECT" if ev.value_match else "INCORRECT"
                ev.resolved_by = "node2"

            # ── Node 2b: partial resolver for NOT_COVERED criteria ─────────────
            not_covered_groups = [g for g in criterion_groups if g["id"] not in n1_covered_ids]
            if not_covered_groups:
                n1_assessments_by_id = {a["criterion_id"]: a for a in n1_assessments}
                chain2b = self._dial_factory.create_llm_with_schema(config.model, _Node2bJudgeResult)
                messages2b = [
                    SystemMessage(content=_NODE2B_SYSTEM),
                    HumanMessage(content=_node2b_user_prompt(
                        not_covered_groups, qa_normalized, n1_assessments_by_id
                    )),
                ]
                n2b_result: _Node2bJudgeResult = await chain2b.ainvoke(messages2b)

                # Anti-hallucination: only keep evals for valid NOT_COVERED ids
                valid_ids = {g["id"] for g in not_covered_groups}
                valid_partial = [e for e in n2b_result.partial_evaluations if e.criterion_id in valid_ids]

                ordered_evals, criterion_groups = _apply_partial_resolver_results(
                    ordered_evals, criterion_groups, valid_partial, qa_normalized, n1_covered_ids
                )

            n2_covered_evals = [e for e in ordered_evals if e.criterion_id in n1_covered_ids]
            n2_correct = sum(1 for e in n2_covered_evals if e.tag == "CORRECT")
            n2_accuracy = round(n2_correct / len(n2_covered_evals), 3) if n2_covered_evals else 0.0

            all_correct = sum(1 for e in ordered_evals if e.tag == "CORRECT")
            n2_accuracy_all = round(all_correct / len(ordered_evals), 3) if ordered_evals else 0.0

            # Serialize the full node2 result for downstream Node 3 metric
            node2_result_blob = {
                "criteria_evaluations": [e.model_dump() for e in ordered_evals],
                "criterion_groups": criterion_groups,
            }

            counts = {
                "n2_correct": n2_correct,
                "n2_covered_total": len(n2_covered_evals),
                "all_correct": all_correct,
                "all_total": len(ordered_evals),
                "node2_result": node2_result_blob,
            }

            return self.Output(
                n2_accuracy=MetricOutputField(value=n2_accuracy, details=counts),
                n2_accuracy_all=MetricOutputField(value=n2_accuracy_all, details=counts),
            )
        except Exception as e:
            err = MetricError(message=str(e))
            return self.Output(n2_accuracy=err, n2_accuracy_all=err)
