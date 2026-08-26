"""MAF Node 3: Prior authorization decision reasoning audit metric.

Evaluates the PA agent's decision output against the contextual GT (derived
from Node 2 resolved criterion states and the GT pathway logic), and audits
the rationale for justification correctness.

Outputs:
  decision_match      — bool (1.0/0.0): actual decision == contextual GT
  justification_match — bool (1.0/0.0): rationale satisfies ≥1 acceptable path
  overall_passed      — bool (1.0/0.0): both match (or override Rule 6/7 fires)
  severity            — string label in details (NONE / LOW / MEDIUM / CRITICAL_SAFETY)
"""

import json
import re
from datetime import date
from typing import Annotated, Any

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

def _resolve_outpatient_hospital_facility(cid: str, treatment_desc: str) -> str | None:
    cid_lower = cid.lower()
    if "outpatient hospital facility" not in cid_lower:
        return None
    is_hospital_opf = treatment_desc == "outpatient"
    if "is not" in cid_lower:
        return "FALSE" if is_hospital_opf else "TRUE"
    if "is" in cid_lower:
        return "TRUE" if is_hospital_opf else "FALSE"
    return "UNKNOWN"


def _compute_contextual_gt(
    gt_criteria: list[dict],
    pathways: list[dict] | None = None,
    actual_values: dict[str, str] | None = None,
    case_inputs: dict[str, Any] | None = None,
) -> tuple[str, str]:
    if actual_values is not None:
        values_by_id: dict[str, str] = {}
        for c in gt_criteria:
            cid = c["id"]
            if cid in actual_values:
                values_by_id[cid] = actual_values[cid]
            elif cid and cid[0] in ("C", "E"):
                values_by_id[cid] = "UNKNOWN"
            else:
                values_by_id[cid] = c["expected_value"]
        mode_desc = "actual run values"
    else:
        values_by_id = {c["id"]: c["expected_value"] for c in gt_criteria}
        mode_desc = "GT expected values"

    if case_inputs:
        treatment_desc = str(case_inputs.get("treatment_setting_desc", "")).strip().lower()
        for c in gt_criteria:
            cid = c["id"]
            if not cid or cid[0] in ("C", "E"):
                continue
            resolved_val = _resolve_outpatient_hospital_facility(cid, treatment_desc)
            if resolved_val is not None:
                values_by_id[cid] = resolved_val

    trace_parts = [
        f"State-resolution mode: {mode_desc}",
        f"Resolved criteria states: {', '.join(f'{k}={v}' for k, v in values_by_id.items())}",
    ]

    def tern_and(a: str, b: str) -> str:
        if a == "FALSE" or b == "FALSE":
            return "FALSE"
        if a == "TRUE" and b == "TRUE":
            return "TRUE"
        return "UNKNOWN"

    def tern_or(a: str, b: str) -> str:
        if a == "TRUE" or b == "TRUE":
            return "TRUE"
        if a == "FALSE" and b == "FALSE":
            return "FALSE"
        return "UNKNOWN"

    def tern_not(a: str) -> str:
        if a == "TRUE":
            return "FALSE"
        if a == "FALSE":
            return "TRUE"
        return "UNKNOWN"

    def resolve_val(s: str) -> str:
        s_clean = s.strip()
        while s_clean.startswith("(") and s_clean.endswith(")"):
            depth = 0
            mismatch = False
            for i, char in enumerate(s_clean):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0 and i < len(s_clean) - 1:
                        mismatch = True
                        break
            if not mismatch:
                s_clean = s_clean[1:-1].strip()
            else:
                break
        if s_clean in values_by_id:
            return values_by_id[s_clean]
        if s_clean.upper() in values_by_id:
            return values_by_id[s_clean.upper()]
        if s_clean.lower() in values_by_id:
            return values_by_id[s_clean.lower()]
        s_norm = " ".join(s_clean.lower().split())
        for k, v in values_by_id.items():
            if " ".join(k.lower().split()) == s_norm:
                return v
        return "UNKNOWN"

    def eval_node(s: str) -> str:  # noqa: C901
        s = s.strip()
        if not s:
            return "UNKNOWN"
        depth = 0
        for i in range(len(s) - 1, -1, -1):
            char = s[i]
            if char == ")":
                depth += 1
            elif char == "(":
                depth -= 1
            elif depth == 0:
                if s[i:i + 4].upper() == " OR " or s[i:i + 4] == " or ":
                    return tern_or(eval_node(s[:i]), eval_node(s[i + 4:]))
                elif s[i] == "|":
                    return tern_or(eval_node(s[:i]), eval_node(s[i + 1:]))
        depth = 0
        for i in range(len(s) - 1, -1, -1):
            char = s[i]
            if char == ")":
                depth += 1
            elif char == "(":
                depth -= 1
            elif depth == 0:
                if s[i:i + 5].upper() == " AND " or s[i:i + 5] == " and ":
                    return tern_and(eval_node(s[:i]), eval_node(s[i + 5:]))
                elif s[i] == "&":
                    return tern_and(eval_node(s[:i]), eval_node(s[i + 1:]))
        if s.upper().startswith("NOT ") or s.startswith("not "):
            return tern_not(eval_node(s[4:]))
        if s.startswith("!"):
            return tern_not(eval_node(s[1:]))
        if s.startswith("(") and s.endswith(")"):
            depth = 0
            mismatch = False
            for i, char in enumerate(s):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0 and i < len(s) - 1:
                        mismatch = True
                        break
            if not mismatch:
                return eval_node(s[1:-1])
        return resolve_val(s)

    if pathways:
        for pw in pathways:
            pw_decision = pw["decision"]
            pw_id = pw["pathway_id"]
            logic_str = pw.get("logic") or " AND ".join(pw.get("criteria_ids", []))
            state = eval_node(logic_str)
            if pw_decision == "criteria_not_met":
                if state == "TRUE":
                    trace_parts.append(f"Exclusion pathway {pw_id} fired because exclusion condition '{logic_str}' is TRUE.")
                    return "criteria_not_met", " | ".join(trace_parts)
            elif pw_decision == "criteria_met" and state == "TRUE":
                trace_parts.append(f"Inclusion pathway {pw_id} ({logic_str}) is fully met (is TRUE).")
                return "criteria_met", " | ".join(trace_parts)

        inclusion_pathways = [pw for pw in pathways if pw["decision"] == "criteria_met"]
        all_blocked = True
        for pw in inclusion_pathways:
            pw_id = pw["pathway_id"]
            logic_str = pw.get("logic") or " AND ".join(pw.get("criteria_ids", []))
            # Extract individual criterion IDs from logic string (e.g. "C1 AND C2 AND EX1")
            cids = re.findall(r"\b[A-Z]+\d+\b", logic_str)
            blocked_by = [cid for cid in cids if eval_node(cid) == "FALSE"]
            stalled_by = [cid for cid in cids if eval_node(cid) == "UNKNOWN"]
            if blocked_by:
                trace_parts.append(f"Pathway {pw_id} is blocked by FALSE criteria: {', '.join(blocked_by)}.")
            elif stalled_by:
                all_blocked = False
                trace_parts.append(f"Pathway {pw_id} is stalled by UNKNOWN criteria: {', '.join(stalled_by)}.")
            else:
                all_blocked = False
        if inclusion_pathways and all_blocked:
            trace_parts.append("All inclusion pathways are blocked by FALSE clinical criteria. Decision is denied.")
            return "criteria_not_met", " | ".join(trace_parts)
        else:
            trace_parts.append("No inclusion pathway is met, but at least one pathway is stalled by UNKNOWN criteria. Decision is pended.")
            return "not_enough_data", " | ".join(trace_parts)

    trace_parts.append("No pathways table provided.")
    return "", " | ".join(trace_parts)


def _build_acceptable_justifications(
    gt_criteria: list[dict],
    contextual_gt: str,
    actual_values: dict[str, str] | None = None,
    pathways: list[dict] | None = None,
    case_inputs: dict[str, Any] | None = None,
) -> list[dict]:
    if actual_values is not None:
        values_by_id = {c["id"]: actual_values.get(c["id"], "UNKNOWN") for c in gt_criteria}
    else:
        values_by_id = {c["id"]: c["expected_value"] for c in gt_criteria}

    if case_inputs:
        treatment_desc = str(case_inputs.get("treatment_setting_desc", "")).strip().lower()
        for c in gt_criteria:
            cid = c["id"]
            if not cid or cid[0] in ("C", "E"):
                continue
            resolved_val = _resolve_outpatient_hospital_facility(cid, treatment_desc)
            if resolved_val is not None:
                values_by_id[cid] = resolved_val

    if contextual_gt == "criteria_met":
        return [{"pathway_name": "deterministic", "must_contain_elements": []}]

    if contextual_gt == "not_enough_data":
        unknown_ids = [cid for cid, val in values_by_id.items() if val == "UNKNOWN"]
        if unknown_ids:
            return [
                {"pathway_name": f"pend_missing_{cid}", "must_contain_elements": [[cid, "Missing"]]}
                for cid in unknown_ids
            ]
        return [{"pathway_name": "missing_data_pend", "must_contain_elements": []}]

    if contextual_gt == "criteria_not_met":
        paths = []
        exclusion_failures = [cid for cid, val in values_by_id.items() if val == "TRUE" and cid.startswith("EX")]
        for cid in exclusion_failures:
            paths.append({"pathway_name": f"denial_exclusion_{cid}", "must_contain_elements": [[cid, "Not Met"]]})

        all_p_paths_blocked = True
        failed_inclusions: set[str] = set()
        if pathways:
            approval_pws = [pw for pw in pathways if pw.get("decision") == "criteria_met"]
            if approval_pws:
                for pw in approval_pws:
                    cids = pw.get("criteria_ids", [])
                    blocked_by = [cid for cid in cids if values_by_id.get(cid) == "FALSE"]
                    if not blocked_by:
                        all_p_paths_blocked = False
                        break
                    else:
                        failed_inclusions.update(blocked_by)
                if all_p_paths_blocked and failed_inclusions:
                    paths.append({
                        "pathway_name": "denial_all_approvals_blocked",
                        "must_contain_elements": [[cid, "Not Met"] for cid in sorted(failed_inclusions)],
                    })
            else:
                all_p_paths_blocked = False
        else:
            all_p_paths_blocked = False

        if not paths or (not all_p_paths_blocked and not exclusion_failures):
            failed_inc = [cid for cid, val in values_by_id.items() if val == "FALSE" and not cid.startswith("EX")]
            if failed_inc:
                paths.append({
                    "pathway_name": "denial_all_approvals_blocked",
                    "must_contain_elements": [[cid, "Not Met"] for cid in failed_inc],
                })

        if paths:
            return paths

    return [{"pathway_name": "deterministic", "must_contain_elements": []}]


def _compute_severity(agent_decision: str, contextual_gt: str) -> str:
    if agent_decision == contextual_gt:
        return "NONE"
    if agent_decision == "criteria_met":
        return "CRITICAL_SAFETY"
    if agent_decision == "criteria_not_met":
        return "MEDIUM"
    return "LOW"


def _flatten_actual_decision(raw: Any) -> dict:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {"decision": str(raw), "rationale": "", "criteria": []}
    if not isinstance(raw, dict):
        return {"decision": "", "rationale": "", "criteria": []}
    if "full_notes" in raw:
        return {
            "decision": raw.get("decision", ""),
            "rationale": raw.get("full_notes", {}).get("rationale", ""),
            "criteria": raw.get("full_notes", {}).get("policy_criteria_assessment", []),
        }
    return {
        "decision": raw.get("decision", ""),
        "rationale": raw.get("rationale", ""),
        "criteria": raw.get("criteria", []),
    }


_NODE3_SYSTEM = """
You are a Clinical Logic Auditor evaluating a Prior Authorization agent's decision output.

You are given:
1. CONTEXTUAL_GT: the mathematically correct ground truth derived from the Q&A extraction results.
   - expected_decision: the correct decision given the criteria states actually found in the documents.
   - acceptable_justifications: a list of valid reasoning paths. Each path lists elements that MUST
     all appear in the agent's reasoning (AND logic). At least ONE path must be satisfied (OR logic).
     Each element is [criterion_id, state] — the agent's rationale must mention that criterion in
     that state (e.g. ["C2", "Missing"] means C2 must be described as missing/undocumented).
   - resolved_criteria_states: a dictionary mapping each criterion_id to its mathematically correct state (TRUE, FALSE, or UNKNOWN) resolved for the patient from the record.
   - criteria_meanings: a dictionary mapping each criterion_id to its full semantic policy description.
   - pathways: the list of policy pathways used to compute expected_decision, each with pathway_id, decision (criteria_met or criteria_not_met), criteria_ids, and logic. Use this to understand which criteria drove the contextual GT determination — in particular, which EX* criteria fired (resolved_state = TRUE → denial), and which C* inclusion criteria were FALSE (blocking all approval pathways).
2. AGENT_OUTPUT: the agent's decision and full rationale text.
3. QA_EXTRACTION_EVIDENCE (when present): the raw Q&A pairs that were used to derive the resolved
   criterion states. For each criterion this includes the mapped question-answer pairs (with the
   extractor's rationale per answer) and the Node 2 evaluator's tag, resolved_state, and reasoning.
   Q&A answers may be free-text strings or JSON-array strings of selected Multiple select options
   (e.g., the full verbatim severity definition selected from the provided option list, or a
   documentation-gap custom option generated by the agent). Use this evidence to understand the
   evidentiary basis for each criterion's resolved state and to assess whether the agent's rationale
   faithfully reflects what the clinical documents actually show.

EVALUATION RULES:
1. decision_match: Agent's decision must match expected_decision.
2. justification_match (OR logic): Agent's reasoning must satisfy at least ONE acceptable_justification.
3. Element verification (AND logic): To satisfy a justification, ALL [criterion_id, state] pairs
   must be present in the reasoning text — the criterion must be mentioned AND associated with that state.
   Use 'criteria_meanings' to map each 'criterion_id' to its semantic description. The reasoning text does NOT
   need to use the literal ID string (like 'C2'). It passes verification for 'criterion_id' if it semantically
   describes that criterion's concept in the specified state.
   State Guidelines for Verification:
   - For an inclusion criterion (C*), "Not Met" or "Failed" mean the requirement was not met/not found in the patient.
   - For an exclusion/contraindication criterion (EX*), "Not Met" or "Triggered" or "Present" mean the exclusion condition (e.g. pain modifiers, contraindications) IS PRESENT in the patient, meaning the patient fails the safety check.
   - "Missing" or "Undocumented" mean the clinical documentation lacks evidence for this criterion.
   - For multi-option criteria: if the Q&A selected an "N/A" option indicating an inactive pathway, the criterion
     is classified as UNKNOWN ("N/A Inactive"). An agent that correctly identifies a pathway as N/A/inactive passes
     verification for "Missing" state on that criterion.
4. No partial credit: if any required element is absent, that justification fails.
5. Logical Consistency and Contradiction Check:
   - Even if the agent's rationale contains all required elements for an acceptable justification, the justification match must be evaluated as FALSE if there are core clinical or logical contradictions.
   - Contradiction by Conflicting Pathways: A contradiction occurs if the agent's rationale cites different criteria that lead to conflicting clinical decisions under guidelines. For example, if there is a missing critical criterion on an unblocked pathway (which logically requires the request to be Pended / "not_enough_data"), but the agent's actual decision in AGENT_OUTPUT is "criteria_not_met" (Denial), then they are denying despite admitting that they do not have enough information to rule out other valid pathways. In this scenario, the justification match has a clinical conflict and MUST be set to FALSE.
   - Clinical State Failure Mismatch (Strict Grounding): A critical contradiction occurs if the agent's rationale asserts a standard policy criterion is "failed" or "Not Met" (thus indicating an active denial reason) when that criterion is actually expected/resolved to be `UNKNOWN` (undocumented/missing) under `resolved_criteria_states`. For example, if the patient has no record of a drug trial, its expected state is `UNKNOWN` (missing). If the agent claims this trial is actively "failed" or "Not Met" to support its reasoning, this is factually incorrect and represents a logical contradiction. In such scenarios, `justification_match` MUST be set to FALSE.
   - No Hallucinated/Phantom Criteria: If the agent's rationale bases its decision on non-existent or hallucinated medical/policy requirements (e.g., asserting the patient failed "daily oral calcium supplementation" or similar criteria which are not listed under `criteria_meanings`), this is an invalidating factual/logic contradiction, and `justification_match` MUST be set to FALSE.
   - Multi-Option Verbatim Allowance (NOT a Contradiction): When the agent's rationale references a verbatim option string selected by the Q&A extractor (e.g., quoting the full Moderate severity definition from a Multiple select answer array), this is acceptable and should be matched to the criterion semantically. Do NOT penalize the agent for using the full verbatim language of a selected option instead of a shorter paraphrase.
   - Legacy Output Language Allowance (NOT a Contradiction): Do NOT penalize or fail the justification match simply because of terminological differences between "Pended"/"Pending"/"PEND" and "criteria_not_met" in the agent's text vs. structured output. In legacy agent reasoning, these terms are often used interchangeably. Therefore, writing "determination is Pended" in the text while outputting "criteria_not_met" as the structured decision is NOT an explicit contradiction and must be ignored/allowed.
   - Non-contradictory Redundancy is Allowed: If the agent mentions additional or redundant "Not Met" or "Met" criteria that do NOT contradict the final decision (e.g., mentioning multiple failed standard therapy requirements to support a flat denial, or mentioning extra met criteria in a clear approval), that is acceptable and does NOT invalidate the justification match.
   - Blanket Exclusion/Inclusion Claims: `resolved_criteria_states` is authoritative for each criterion's actual state — use it to verify AGENT_OUTPUT's rationale, do not silently substitute your own independent clinical re-reading of the record in `explanation`. If the agent's rationale (or your own reasoning) asserts a blanket claim (e.g., "all exclusions are safely absent") that is not true for every individual criterion's `resolved_criteria_states` value, this is a hallucinated-confidence contradiction: populate `criteria_state_conflicts` with `[criterion_id, "<note describing the conflict>"]` for each conflicting criterion and reflect it in `explanation`, even when it does not change `decision_match`. If you believe a `resolved_criteria_states` value is itself wrong, record that disagreement in `criteria_state_conflicts` rather than silently overriding it in your reasoning.

6. Controlling Rule Override — Affirmative Imaging and Diagnostic Criteria (MORE_INFORMATION is the correct decision):
   The PA agent operates under a mandatory "Controlling Rule — Affirmative Diagnostic, Imaging, and Objective Testing Requirements" that instructs it to classify below-threshold or sub-grade imaging/diagnostic findings as "Missing" (Stalled), never as "Not Met", and to return MORE_INFORMATION (not_enough_data) in those cases — even when the current report explicitly shows only mild, mild-to-moderate, borderline, or equivocal findings.

   When ALL of the following conditions hold:
   a. The agent's actual output decision is "not_enough_data";
   b. The expected_decision (contextual GT) is "criteria_not_met";
   c. The contextual GT was reached because one or more affirmative imaging, diagnostic-grade, or clinical-radiographic-concordance inclusion criteria (C*) have resolved_criteria_state = FALSE — meaning the available diagnostic report showed findings below the policy-required severity threshold (e.g., mild or mild-to-moderate stenosis when moderate-to-severe is required; mild joint space narrowing when Grade 3-4 osteoarthritis is required; symptoms described as out of proportion to objective findings; concordance unestablished);
   d. The agent's rationale correctly identifies those criteria as "Missing", "Stalled", "unresolved", or "not established" because the evidence does not affirmatively demonstrate the required severity threshold or clinical-radiographic concordance;
   e. The agent explicitly requests additional or clarifying diagnostic documentation (imaging with severity detail, updated radiographic studies, level-specific findings, or correlation evidence);

   THEN: set justification_match = TRUE and overall_passed = TRUE.

   Rationale: The contextual GT computation assigns resolved_state = FALSE when the Q&A extraction confirms that the required diagnostic threshold was not met by the available evidence. However, the PA agent is explicitly instructed to treat such below-threshold findings as "Missing" (Stalled) rather than "Not Met", and to return MORE_INFORMATION rather than a denial. This is a deliberate, designed asymmetry between the Q&A extraction layer and the PA agent's decision-making layer. The agent is not wrong — it is correctly applying its mandatory Controlling Rule. Do NOT penalize the agent for this correct behavior.

7. Relative Exclusion Override — Discretionary Exclusions Requiring Case-by-Case Review (MORE_INFORMATION is the correct decision):
   The PA agent is instructed that exclusion and contraindication criteria which require case-by-case clinical evaluation, physician clearance, or psychiatric clearance — rather than constituting an absolute bar to coverage — must be classified as "Missing" (unresolved) when the condition is present but no completed clearance is documented. The agent returns MORE_INFORMATION (not_enough_data) in these cases.

   When ALL of the following conditions hold:
   a. The agent's actual output decision is "not_enough_data";
   b. The expected_decision (contextual GT) is "criteria_not_met";
   c. The contextual GT was reached because one or more exclusion criteria (EX*) have resolved_criteria_state = TRUE — meaning the condition associated with the exclusion (e.g., clinically significant depression, anxiety disorder, other psychiatric disorder, non-physiologic pain modifier, non-operative mimicking condition) is documented as present;
   d. The exclusion criterion is NOT an absolute unconditional bar to coverage; rather, the policy requires a case-by-case clinical review, psychiatric clearance, or documented rule-out evaluation before denying coverage on that basis — indicated by policy language such as "must be ruled out", "requires case-by-case review", "may not be approved", or "reviewed on a case-by-case basis";
   e. The agent's rationale correctly identifies that while the condition is present, no completed case-by-case review, clearance, or rule-out evaluation is documented in the supplied excerpts, and therefore classifies the exclusion criterion as "Missing", "Unresolved", or "requires further evaluation";
   f. The agent explicitly requests the missing clearance documentation (psychiatric evaluation, case-by-case review outcome, or differentiation from the target condition);

   THEN: set justification_match = TRUE and overall_passed = TRUE.

   Rationale: The contextual GT computation treats an EX* criterion with resolved_state = TRUE as a denial trigger because the condition is present. However, when the policy requires a case-by-case clearance process rather than making presence alone an absolute bar, the agent is correctly applying its exclusion-handling rules: a condition that is present but lacks a completed clearance is "Missing" (unresolved), not "Not Met" (definitively triggered). The agent is correct to request the missing clearance rather than issuing a denial. Do NOT penalize the agent for this correct behavior.

   Important distinction: This override does NOT apply when the policy makes the condition an absolute exclusion regardless of severity or clearance (i.e., mere presence is disqualifying). In that case, if the agent incorrectly classifies the exclusion as Missing and returns not_enough_data, that remains a genuine error and justification_match = FALSE.

In your explanation, provide a detailed step-by-step audit showing which of the acceptable justifications passed/failed, checking specifically for logical consistency, clinical state accuracy, and non-hallucination contradictions, and why. When applying Rule 6 or Rule 7, explicitly state which override triggered and confirm that all listed conditions are satisfied.

Return JSON with:
{
  "decision_match": true/false,
  "justification_match": true/false,
  "overall_passed": true/false,
  "matched_justification_index": null or integer (0-based index of first passing justification, or -1 when override Rule 6 or 7 applies),
  "missing_elements": [[criterion_id, state], ...] (elements from the best partial match that were absent; empty when an override rule applies),
  "criteria_state_conflicts": [[criterion_id, note], ...] (populated whenever AGENT_OUTPUT's rationale or your own explanation asserts a state for a criterion that conflicts with resolved_criteria_states; empty if no conflicts),
  "explanation": "Detailed step-by-step clinical audit reasoning explaining the results."
}
"""


def _node3_user_prompt(
    gt_criteria: list[dict],
    actual_decision_flat: dict,
    contextual_gt: str,
    actual_values: dict[str, str] | None = None,
    pathways: list[dict] | None = None,
    case_inputs: dict[str, Any] | None = None,
    current_date: str | None = None,
    criterion_groups: list[dict] | None = None,
    n2_evaluations: list[dict] | None = None,
) -> str:
    acceptable_justifications = _build_acceptable_justifications(
        gt_criteria, contextual_gt, actual_values=actual_values,
        pathways=pathways, case_inputs=case_inputs,
    )
    if actual_values is not None:
        resolved_states = {c["id"]: actual_values.get(c["id"], "UNKNOWN") for c in gt_criteria}
    else:
        resolved_states = {c["id"]: c["expected_value"] for c in gt_criteria}

    # Resolve non-standard criteria using actual case_inputs if available
    if case_inputs:
        treatment_desc = str(case_inputs.get("treatment_setting_desc", "")).strip().lower()
        for c in gt_criteria:
            cid = c["id"]
            if not cid or cid[0] in ("C", "E"):
                continue

            resolved_val = _resolve_outpatient_hospital_facility(cid, treatment_desc)
            if resolved_val is not None:
                resolved_states[cid] = resolved_val

    criteria_meanings = {c["id"]: c["description"] for c in gt_criteria}
    contextual_gt_payload = {
        "expected_decision": contextual_gt,
        "acceptable_justifications": acceptable_justifications,
        "resolved_criteria_states": resolved_states,
        "criteria_meanings": criteria_meanings,
        "pathways": pathways or [],
    }

    # Build per-criterion Q&A evidence (question, answer, rationale) keyed by criterion_id
    qa_evidence_section = ""
    if criterion_groups:
        evals_by_id = {ev.get("criterion_id", ""): ev for ev in (n2_evaluations or [])}
        groups_by_id = {g["id"]: g for g in criterion_groups}

        evidence_list = []
        for c in gt_criteria:
            cid = c["id"]
            group = groups_by_id.get(cid)
            if not group:
                continue
            mapped_qa = group.get("mapped_qa", [])
            if not mapped_qa:
                continue
            ev = evals_by_id.get(cid)
            qa_entries = []
            for qa in mapped_qa:
                entry: dict = {"index": qa["index"], "question": qa["question"], "answer": qa["answer"]}
                if qa.get("rationale"):
                    entry["rationale"] = qa["rationale"]
                qa_entries.append(entry)
            item: dict = {"criterion_id": cid, "mapped_qa": qa_entries}
            if ev:
                if ev.get("tag") is not None:
                    item["node2_tag"] = ev["tag"]
                if ev.get("resolved_state") is not None:
                    item["node2_resolved_state"] = ev["resolved_state"]
                if ev.get("reasoning") is not None:
                    item["node2_reasoning"] = ev["reasoning"]
                item["node2_provenance"] = ev.get("resolved_by") or "node2"
            evidence_list.append(item)
        if evidence_list:
            qa_evidence_section = f"\n<QA_EXTRACTION_EVIDENCE>\n{json.dumps(evidence_list, indent=2)}\n</QA_EXTRACTION_EVIDENCE>"

    date_section = f"\n<CURRENT_DATE>{current_date}</CURRENT_DATE>" if current_date else ""
    return f"""
<CONTEXTUAL_GT>
{json.dumps(contextual_gt_payload, indent=2)}
</CONTEXTUAL_GT>

<AGENT_OUTPUT>
Decision: {actual_decision_flat.get("decision", "")}
Reasoning: {actual_decision_flat.get("rationale", "")}
</AGENT_OUTPUT>
{qa_evidence_section}{date_section}"""

class _Node3JudgeResult(BaseModel):
    decision_match: bool
    justification_match: bool
    overall_passed: bool
    matched_justification_index: int | None = None
    missing_elements: list[list[str]] = Field(default_factory=list)
    criteria_state_conflicts: list[list[str]] = Field(default_factory=list)
    explanation: str


# ── Metric ────────────────────────────────────────────────────────────────────

class Node3DecisionMetric(BaseLLMMetric):
    """Node 3: audits the PA agent's decision and rationale against contextual GT.

    Contextual GT is derived deterministically from Node 2 resolved criterion
    states via the GT pathway logic — exactly as in the reporter pipeline.
    Returns decision_match, justification_match, and overall_passed (all 0/1).
    Severity label is included in the details dict.
    """

    name: str = "maf.node3_decision"
    display_name: str = "MAF Node 3: Decision Reasoning Audit"
    description: str = (
        "Audits the PA agent's decision and rationale against the contextual GT "
        "(derived from Node 2 resolved criterion states and GT pathway logic). "
        "Returns decision_match, justification_match, overall_passed (0/1). Requires LLM."
    )

    class Config(BaseModel):
        model: str = Field(description="The LLM deployment name for evaluation.")

    class Input(BaseModel):
        gt_criteria: Annotated[
            list[dict],
            Field(description="GT criteria list with 'id', 'description', 'expected_value'."),
        ]
        gt_pathways: Annotated[
            list[dict],
            Field(description="GT pathway list with 'pathway_id', 'decision', and either 'logic' (e.g. 'C1 AND C2') or 'criteria_ids' list."),
        ]
        actual_decision: Annotated[
            Any,
            Field(description="Agent's decision output. Dict with 'decision', 'rationale', 'criteria' (or 'full_notes'). JSON string also accepted."),
        ]
        node2_result: Annotated[
            dict,
            Field(description="The node2_result blob from maf.node2_qa_extraction output details."),
        ]
        case_inputs: Annotated[
            dict | None,
            Field(default=None, description="Optional case input fields (e.g. treatment_setting_desc) for non-standard criterion resolution."),
        ] = None

    class Output(BaseModel):
        decision_match: Annotated[
            MetricOutputField | MetricError,
            Field(discriminator="type", description="1.0 if actual decision matches contextual GT, else 0.0."),
        ]
        justification_match: Annotated[
            MetricOutputField | MetricError,
            Field(discriminator="type", description="1.0 if rationale satisfies at least one acceptable justification path."),
        ]
        overall_passed: Annotated[
            MetricOutputField | MetricError,
            Field(discriminator="type", description="1.0 if both decision and justification match (or override Rule 6/7 fires)."),
        ]

    examples = [
        MetricExample(
            name="correct decision",
            description="Agent decision matches contextual GT with valid justification",
            config=Config(model="gemini-3.5-flash"),
            input=Input(
                gt_criteria=[{"id": "C1", "description": "Patient age >= 18", "expected_value": "TRUE"}],
                gt_pathways=[{"pathway_id": "P1", "logic": "C1", "decision": "criteria_met"}],
                actual_decision={"decision": "criteria_met", "rationale": "Patient is 45 years old, meeting age criterion.", "criteria": []},
                node2_result={
                    "criteria_evaluations": [{"criterion_id": "C1", "tag": "CORRECT", "resolved_state": "TRUE", "reasoning": "Yes answer confirms age.", "resolved_by": "node2"}],
                    "criterion_groups": [],
                },
            ),
            expected_output=Output(
                decision_match=MetricOutputField(value=1.0),
                justification_match=MetricOutputField(value=1.0),
                overall_passed=MetricOutputField(value=1.0),
            ),
        ),
    ]

    def __init__(self, dial_factory: DialFactory, settings: CommonGroupSettings):
        super().__init__(dial_factory, settings)

    async def evaluate_async(self, config: Config, input: Input) -> Output:
        try:
            n2_evals: list[dict] = input.node2_result.get("criteria_evaluations", [])
            criterion_groups: list[dict] = input.node2_result.get("criterion_groups", [])

            actual_decision_flat = _flatten_actual_decision(input.actual_decision)
            actual_decision_str = actual_decision_flat.get("decision", "")

            # Derive contextual GT from Node 2 resolved states + pathways
            actual_values = {ev["criterion_id"]: ev.get("resolved_state", "UNKNOWN") for ev in n2_evals}
            contextual_gt, contextual_trace = _compute_contextual_gt(
                input.gt_criteria,
                pathways=input.gt_pathways,
                actual_values=actual_values,
                case_inputs=input.case_inputs,
            )
            if not contextual_gt:
                contextual_gt = next(
                    (c["expected_value"] for c in input.gt_criteria if c.get("id") == "__decision__"),
                    "not_enough_data",
                )

            # Deterministic decision_match (never LLM-supplied)
            dm = actual_decision_str == contextual_gt

            chain = self._dial_factory.create_llm_with_schema(config.model, _Node3JudgeResult)
            messages = [
                SystemMessage(content=_NODE3_SYSTEM),
                HumanMessage(content=_node3_user_prompt(
                    input.gt_criteria,
                    actual_decision_flat,
                    contextual_gt,
                    actual_values=actual_values,
                    pathways=input.gt_pathways,
                    case_inputs=input.case_inputs,
                    current_date=date.today().isoformat(),
                    criterion_groups=criterion_groups,
                    n2_evaluations=n2_evals,
                )),
            ]
            llm_result: _Node3JudgeResult = await chain.ainvoke(messages)

            # Always override decision_match deterministically (reporter pattern)
            llm_result.decision_match = dm

            severity = _compute_severity(actual_decision_str, contextual_gt)

            details = {
                "contextual_gt": contextual_gt,
                "contextual_trace": contextual_trace,
                "actual_decision": actual_decision_str,
                "severity": severity,
                "matched_justification_index": llm_result.matched_justification_index,
                "missing_elements": llm_result.missing_elements,
                "criteria_state_conflicts": llm_result.criteria_state_conflicts,
                "explanation": llm_result.explanation,
            }

            return self.Output(
                decision_match=MetricOutputField(value=1.0 if llm_result.decision_match else 0.0, details=details),
                justification_match=MetricOutputField(value=1.0 if llm_result.justification_match else 0.0, details=details),
                overall_passed=MetricOutputField(value=1.0 if llm_result.overall_passed else 0.0, details=details),
            )
        except Exception as e:
            err = MetricError(message=str(e))
            return self.Output(decision_match=err, justification_match=err, overall_passed=err)
