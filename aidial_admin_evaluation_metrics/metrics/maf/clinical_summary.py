"""MAF Clinical Summary NLI evaluation metric.

Single-call metric: receives the raw clinical-summary API response and a list of
target field specs, evaluates each field with the appropriate check type
(exact / nli / encounter), and returns aggregated recall metrics.

Response parsing mirrors EvaluationResponse.from_clinical_summary_json() from MAF.
NLI scoring mirrors ClinicalSummaryChecker from MAF (0.0 / 0.5 / 1.0 per fact).
Encounter localization mirrors EncounterLocalizationChecker from MAF.

Outputs:
  main_fact_recall        — fraction of GT facts whose MAIN part is entailed (0–1)
  recall_with_additionals — weighted recall: partial 0.5 credit when main ENT but
                            any additional qualifier is not ENT (0–1)
  encounter_verdict       — fraction of encounter targets that are HITs (0–1),
                            or -1.0 when there are no encounter targets
  precision               — fraction of output atomic facts that are in-scope per GT
                            (requires compute_precision=True in Config; -1.0 = N/A)
  agent_token_total       — total tokens from the clinical-summary API response
                            (sum of all token_usage entries; -1 if not present)
"""

import asyncio
import json
import re
from typing import Annotated, Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from aidial_admin_evaluation_metrics.app_config import CommonGroupSettings
from aidial_admin_evaluation_metrics.dial.llm_client import DialFactory
from aidial_admin_evaluation_metrics.metrics.common.base_llm_metric import BaseLLMMetric
from aidial_admin_evaluation_metrics.metrics.common.base_metric import MetricExample
from aidial_admin_evaluation_metrics.metrics.common.types import MetricError, MetricField, MetricOutputField


# ── Response parsing ──────────────────────────────────────────────────────────

def _slugify(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _extract_fields(response: dict) -> tuple[dict[str, Any], dict[str, dict]]:
    """Parse the clinical-summary API response into field values and per-field metadata.

    Mirrors EvaluationResponse.from_clinical_summary_json() from maf-evaluation-framework.
    field_name = "<section_name>.<slugified_label>", e.g. "chief_complaint.date_of_admission".

    Returns:
        values:   {field_name: value}
        metadata: {field_name: {"rationale": str|None, "citations": dict|None, "found": bool|None}}
    """
    payload = response.get("payload", response)
    values: dict[str, Any] = {}
    metadata: dict[str, dict] = {}

    for item in payload.get("items") or []:
        section = str(item.get("name", ""))
        for attr in item.get("attributes") or []:
            label = str(attr.get("label", ""))
            field_name = f"{section}.{_slugify(label)}"
            values[field_name] = attr.get("value")
            metadata[field_name] = {
                "rationale": attr.get("rationale"),
                "citations": attr.get("citations"),
                "found": attr.get("found"),
            }

    # last_encounter → combined localization object (used by encounter check)
    last_enc = payload.get("last_encounter") or {}
    if last_enc:
        values["last_encounter.localization"] = last_enc

    # imaging_diagnostics — section name differs by summary_type
    is_outpatient = payload.get("summary_type") == "outpatient"
    imaging_section = "objective_clinical_evidence" if is_outpatient else "objective_clinical_data"
    procedures = (payload.get("imaging_diagnostics") or {}).get("procedures")
    values[f"{imaging_section}.imaging_diagnostics"] = procedures

    return values, metadata


def _extract_token_total(response: dict) -> int:
    """Sum all total_tokens across token_usage entries in the payload. Returns -1 if absent."""
    payload = response.get("payload", response)
    usages = (payload.get("token_usage") or {}).get("token_usages") or []
    if not usages:
        return -1
    return sum(int(u.get("total_tokens", 0)) for u in usages)


# ── Encounter localization (static, no LLM) ───────────────────────────────────

def _pages_within(actual_range: list, expected_range: list) -> bool:
    """Return True if agent page_range lies within the GT page_range."""
    try:
        gt_start, gt_end = int(expected_range[0]), int(expected_range[1])
    except (IndexError, TypeError, ValueError):
        return False
    if not actual_range or actual_range[0] is None:
        return False
    agent_start = int(actual_range[0])
    agent_end = actual_range[1] if len(actual_range) > 1 else None
    within = gt_start <= agent_start <= gt_end
    if agent_end is not None:
        within = within and int(agent_end) <= gt_end
    return within


def _score_encounter(actual: Any, expected: dict) -> float:
    if isinstance(actual, str):
        try:
            actual = json.loads(actual)
        except (ValueError, TypeError):
            actual = {}
    if not isinstance(actual, dict):
        return 0.0
    file_hit = bool(expected.get("file_name")) and expected["file_name"] == actual.get("file_name")
    pages_hit = _pages_within(actual.get("page_range") or [], expected.get("page_range") or [])
    return 1.0 if (file_hit and pages_hit) else 0.0


# ── NLI scoring (LLM) ─────────────────────────────────────────────────────────

_NLI_SYSTEM = """
You are an expert in Natural Language Inference (NLI) for clinical documentation.
You are given a PREMISE (a field from a generated clinical summary) and a list of
EXPECTED FACTS. Each fact has a "main" part (the core claim) and may have
"additional" parts (qualifiers: values, dates, durations, details).

For the main part AND each additional part of every fact, independently determine:
- ENT (Entailment): the part is definitely true given the premise.
- CONT (Contradiction): the part is definitely false given the premise.
- NEUT (Neutral): the premise does not provide enough information.

Rules:
1. Judge every part independently; do not let one part's verdict influence another.
2. Be STRICT with numbers, dates, units, dosages and flags: they must match the
   premise exactly (formatting differences like "07/18/2025" vs "Jul 18, 2025"
   or "500 mg" vs "500mg" are tolerated). A different value, date or unit is CONT.
3. Phrasing differences, synonyms and abbreviations do NOT prevent entailment for
   descriptive parts.
3a. ICD-10 code annotations (e.g. "[J18.9]") are formatting, not clinical content:
   judge diagnosis matches on wording alone. This does NOT relax rule 2 for
   measurements, lab values, dates or dosages.
4. Provide a brief, one-sentence reasoning per part.

Return JSON in this exact shape:
{
  "result": [
    {
      "main": "<main part verbatim>",
      "reasoning": "<one sentence>",
      "tag": "ENT",
      "additionals": [
        {"additional": "<additional part verbatim>", "reasoning": "<one sentence>", "tag": "NEUT"}
      ]
    }
  ]
}

Return one entry per expected fact, in the same order.
Empty "additionals" list when the fact has none.
""".strip()


class _AdditionalVerdict(BaseModel):
    additional: str
    reasoning: str
    tag: Literal["ENT", "CONT", "NEUT"]


class _FactVerdict(BaseModel):
    main: str
    reasoning: str
    tag: Literal["ENT", "CONT", "NEUT"]
    additionals: list[_AdditionalVerdict] = Field(default_factory=list)

    @property
    def score(self) -> float:
        """0.0 if main not ENT; 1.0 if main ENT and all additionals ENT; 0.5 otherwise."""
        if self.tag != "ENT":
            return 0.0
        if all(a.tag == "ENT" for a in self.additionals):
            return 1.0
        return 0.5


class _NLIResult(BaseModel):
    result: list[_FactVerdict] = Field(default_factory=list)


def _normalize_facts(raw_facts: list) -> list[dict]:
    """Normalize facts: bare strings become main-only; dicts are passed through."""
    out = []
    for f in raw_facts:
        if isinstance(f, str):
            out.append({"main": f, "additional": []})
        elif isinstance(f, dict):
            out.append({"main": f.get("main", ""), "additional": list(f.get("additional") or [])})
    return out


def _field_meta_entry(meta: dict | None) -> dict:
    """Return a normalized rationale/citations/found block from per-field metadata."""
    if not meta:
        return {"rationale": None, "citations": None, "found": None}
    return {
        "rationale": meta.get("rationale"),
        "citations": meta.get("citations"),
        "found": meta.get("found"),
    }


async def _missed_field_result(
    field_name: str, norm_facts: list, actual_value: Any = None, meta: dict | None = None
) -> dict:
    """Return a zero-score result for a field not found in the response (no LLM call)."""
    return {
        "field": field_name,
        "check": "nli",
        "facts_total": len(norm_facts),
        "facts_main_ent": 0,
        "facts_score": 0.0,
        "verdicts": [],
        "extracted": actual_value,
        "reason": "field not found in response",
        **_field_meta_entry(meta),
    }


async def _run_nli(
    chain: Any, field_name: str, actual_value: str, raw_facts: list, meta: dict | None = None
) -> dict:
    """Run one NLI call for a single field. Returns a per-field result dict."""
    norm_facts = _normalize_facts(raw_facts)
    if not norm_facts:
        return {
            "field": field_name,
            "check": "nli",
            "facts_total": 0,
            "facts_main_ent": 0,
            "facts_score": 0.0,
            "verdicts": [],
            "extracted": actual_value,
            **_field_meta_entry(meta),
        }

    facts_json = json.dumps(norm_facts, ensure_ascii=False, indent=2)
    user_prompt = f"<premise>\n{actual_value}\n</premise>\n\n<expected_facts>\n{facts_json}\n</expected_facts>"
    messages = [SystemMessage(content=_NLI_SYSTEM), HumanMessage(content=user_prompt)]
    nli_result: _NLIResult = await chain.ainvoke(messages)

    facts_total = len(nli_result.result)
    facts_main_ent = sum(1 for v in nli_result.result if v.tag == "ENT")
    facts_score = sum(v.score for v in nli_result.result)
    verdicts = [
        {
            "main": v.main,
            "tag": v.tag,
            "score": v.score,
            "reasoning": v.reasoning,
            "additionals": [
                {"additional": a.additional, "tag": a.tag, "reasoning": a.reasoning}
                for a in v.additionals
            ],
        }
        for v in nli_result.result
    ]

    return {
        "field": field_name,
        "check": "nli",
        "facts_total": facts_total,
        "facts_main_ent": facts_main_ent,
        "facts_score": facts_score,
        "verdicts": verdicts,
        "extracted": actual_value,
        **_field_meta_entry(meta),
    }


# ── Precision (over-selection) — 2-stage LLM ────────────────────────────────

_DECOMP_SYSTEM = """
You split one field of a generated clinical summary into ATOMIC facts. The field may
be any clinical field (symptoms, history, findings, diagnostics, treatments,
comorbidities, documentation gaps, etc.).

An atomic fact is a single distinct clinical claim — e.g. one diagnosis, symptom,
finding, measurement/lab value, treatment or intervention, event, functional or
safety status, allergy, or behavior — together with its own qualifiers (severity,
laterality, values, dates, codes) kept attached.

Rules:
- If the text is already a list (semicolon- or bullet-separated), return exactly one
  fact per list item — do not split a single item further and do not merge items.
- If the text is prose, break it into the distinct clinical claims it asserts.
- Copy each fact VERBATIM from the text as much as possible. Do NOT add, infer,
  normalize, or invent anything not present. Do NOT drop qualifiers.
- Ignore pure filler ("Not found.", "None documented") — return an empty list for it.

Return JSON: {"facts": ["<fact 1>", "<fact 2>", ...]}
""".strip()

_VERIFY_SYSTEM = """
You are matching CANDIDATE facts taken from one field of a generated clinical summary
against the PREMISE, which is the GROUND-TRUTH set of facts expected for that same
field. For each candidate, decide whether it corresponds to one of the ground-truth
facts — i.e. whether the candidate is IN SCOPE.

For each candidate's "main" part, determine:
- ENT (in scope): the ground truth contains a fact about the SAME underlying clinical
  claim — the same diagnosis, symptom, finding, measurement, treatment, event,
  status, allergy, or behavior — even if the candidate adds extra detail the ground
  truth omits, or uses a synonym, abbreviation, or billing code.
- CONT (conflict): the ground truth states something directly incompatible with the
  candidate for the same item (e.g. a conflicting value, date, or laterality).
- NEUT (extra / out of scope): the ground truth has NO fact about this claim at all.

Rules:
1. Match on the underlying CLINICAL CLAIM, not wording. Synonyms, abbreviations,
   codes, and extra descriptive qualifiers on the candidate do NOT prevent ENT.
2. A candidate about a DIFFERENT claim than any ground-truth fact is NEUT, even if
   clinically related.
3. Do not be generous: NEUT whenever the ground truth simply does not include the
   candidate's underlying claim.
4. One-sentence reasoning per candidate.

Return JSON in this exact shape:
{
  "result": [
    {"main": "<candidate verbatim>", "reasoning": "<one sentence>", "tag": "ENT", "additionals": []}
  ]
}

Return one entry per candidate fact, in the same order, each with an empty "additionals" list.
""".strip()


class _DecompResult(BaseModel):
    facts: list[str] = Field(default_factory=list)


def _render_gt_premise(norm_facts: list[dict]) -> str:
    lines = []
    for f in norm_facts:
        adds = f.get("additional") or []
        line = f"- {f['main']}"
        if adds:
            line += " [" + " | ".join(adds) + "]"
        lines.append(line)
    return "\n".join(lines)


async def _run_precision_for_field(
    chain_decomp: Any,
    chain_verify: Any,
    field_name: str,
    actual_value: str,
    norm_facts: list[dict],
) -> dict:
    """2-stage precision for one NLI field: decompose output → verify against GT."""
    if not actual_value.strip() or not norm_facts:
        return {"field": field_name, "n_output": 0, "n_in_scope": 0, "verdicts": []}

    # Stage 1: decompose output text into atomic facts
    decomp_messages = [
        SystemMessage(content=_DECOMP_SYSTEM),
        HumanMessage(content=f"<field>\n{actual_value}\n</field>"),
    ]
    decomp_result: _DecompResult = await chain_decomp.ainvoke(decomp_messages)
    out_facts = [f for f in decomp_result.facts if f.strip()]
    if not out_facts:
        return {"field": field_name, "n_output": 0, "n_in_scope": 0, "verdicts": []}

    # Stage 2: verify each output fact against GT premise
    candidates_json = json.dumps(
        [{"main": f, "additional": []} for f in out_facts], ensure_ascii=False, indent=2
    )
    gt_premise = _render_gt_premise(norm_facts)
    verify_messages = [
        SystemMessage(content=_VERIFY_SYSTEM),
        HumanMessage(
            content=f"<premise>\n{gt_premise}\n</premise>\n\n"
                    f"<candidate_facts>\n{candidates_json}\n</candidate_facts>"
        ),
    ]
    verify_result: _NLIResult = await chain_verify.ainvoke(verify_messages)

    n_in_scope = sum(1 for v in verify_result.result if v.tag == "ENT")
    return {
        "field": field_name,
        "n_output": len(out_facts),
        "n_in_scope": n_in_scope,
        "verdicts": [
            {"fact": v.main, "tag": v.tag, "reasoning": v.reasoning}
            for v in verify_result.result
        ],
    }


# ── Metric ────────────────────────────────────────────────────────────────────

class ClinicalSummaryMetric(BaseLLMMetric):
    """MAF Clinical Summary NLI evaluation metric.

    Evaluates a generated clinical summary against ground-truth facts using
    structured NLI (main + additional qualifier scoring), exact-match checks,
    and binary encounter localization.
    """

    name: str = "maf.clinical_summary"
    display_name: str = "MAF Clinical Summary Evaluation"
    description: str = (
        "Evaluates a clinical-summary API response against curated ground-truth facts. "
        "Supports three check types per field: 'nli' (structured NLI with partial 0.5 scoring), "
        "'exact' (string equality; null expected = agent should return null/empty), and "
        "'encounter' (binary encounter-localization: file_name exact + page_range containment). "
        "Returns main_fact_recall, recall_with_additionals, encounter_verdict, "
        "precision (optional, requires compute_precision=True), and agent_token_total."
    )

    class Config(BaseModel):
        model: str = Field(description="The LLM deployment name for NLI evaluation.")
        compute_precision: bool = Field(
            default=False,
            description=(
                "Also compute precision (over-selection): fraction of output atomic facts "
                "that are in-scope per GT. Adds 2 extra LLM calls per NLI field. "
                "When False, precision output value is -1.0 (N/A sentinel)."
            ),
        )

    class Input(BaseModel):
        endpoint_response: Annotated[
            dict,
            Field(
                description=(
                    "Raw JSON response from the clinical-summary API. "
                    "Expected shape: {\"payload\": {\"items\": [{\"name\": \"<section>\", "
                    "\"attributes\": [{\"label\": \"...\", \"value\": \"...\"}]}], "
                    "\"last_encounter\": {...}, \"imaging_diagnostics\": {...}, "
                    "\"token_usage\": {\"token_usages\": [{\"total_tokens\": N, ...}]}}}. "
                    "The 'payload' wrapper is optional — the dict is also accepted directly."
                )
            ),
        ]
        target_fields: Annotated[
            list[dict],
            Field(
                description=(
                    "List of target field specs. Each dict must have: "
                    "'field_name' (str, e.g. 'chief_complaint.date_of_admission'), "
                    "'check' ('exact' | 'nli' | 'encounter'). "
                    "For 'nli': 'facts' (list of strings or {main, additional[]} dicts). "
                    "For 'exact': 'expected' (str or null — null means the agent should "
                    "return null/empty for this field). "
                    "For 'encounter': 'expected' ({file_name: str, page_range: [start, end]})."
                )
            ),
        ]

    class Output(BaseModel):
        main_fact_recall: Annotated[
            MetricField,
            Field(
                description=(
                    "[Recall] Fraction of GT facts whose MAIN part is entailed by the response (0–1). "
                    "Exact/null fields count as one implicit fact. Encounter checks excluded."
                ),
            ),
        ]
        recall_with_additionals: Annotated[
            MetricField,
            Field(
                description=(
                    "[Recall] Weighted recall: main ENT + all additionals ENT → 1.0; "
                    "main ENT + any additional not ENT → 0.5; main not ENT → 0.0. "
                    "Mean across all facts. Range 0–1."
                ),
            ),
        ]
        encounter_verdict: Annotated[
            MetricField,
            Field(
                description=(
                    "[Encounter] Fraction of encounter-localization targets that are HITs "
                    "(file_name exact match AND agent page_range ⊆ GT page_range). "
                    "1.0 = all HIT, 0.0 = all MISS, -1.0 = N/A (no encounter targets in input)."
                ),
            ),
        ]
        precision: Annotated[
            MetricField,
            Field(
                description=(
                    "[Precision] Fraction of output atomic facts that are in-scope per GT "
                    "(scope adherence / over-selection). Computed only when Config.compute_precision=True. "
                    "-1.0 = N/A (disabled or no NLI fields with non-empty output)."
                ),
            ),
        ]
        agent_token_total: Annotated[
            MetricField,
            Field(
                description=(
                    "Total tokens used by the clinical-summary agent across all token_usage entries "
                    "in the API response. -1.0 = not present in response."
                ),
            ),
        ]

    examples = [
        MetricExample(
            name="perfect clinical summary",
            description="All facts entailed, exact match passes, encounter is a HIT",
            config=Config(model="gemini-3.5-flash"),
            input=Input(
                endpoint_response={
                    "payload": {
                        "summary_type": "inpatient",
                        "items": [
                            {
                                "name": "chief_complaint",
                                "attributes": [
                                    {"label": "Date of Admission", "value": "Nov 26, 2025", "found": True},
                                    {"label": "Admission Details", "value": "Patient admitted via ED with fever and cough.", "found": True},
                                ],
                            }
                        ],
                        "last_encounter": {"file_name": "notes.pdf", "page_range": [2, 4]},
                        "imaging_diagnostics": {"procedures": None},
                        "token_usage": {
                            "token_usages": [
                                {"total_tokens": 1500, "prompt_tokens": 1200, "completion_tokens": 300, "deployment_id": "gemini-3.5-flash"}
                            ]
                        },
                    }
                },
                target_fields=[
                    {"field_name": "chief_complaint.date_of_admission", "check": "exact", "expected": "Nov 26, 2025"},
                    {"field_name": "chief_complaint.admission_details", "check": "nli", "facts": ["admitted via ED"]},
                    {"field_name": "last_encounter.localization", "check": "encounter", "expected": {"file_name": "notes.pdf", "page_range": [2, 4]}},
                ],
            ),
            expected_output=Output(
                main_fact_recall=MetricOutputField(value=1.0),
                recall_with_additionals=MetricOutputField(value=1.0),
                encounter_verdict=MetricOutputField(value=1.0),
                precision=MetricOutputField(value=-1.0),
                agent_token_total=MetricOutputField(value=1500.0),
            ),
        ),
    ]

    def __init__(self, dial_factory: DialFactory, settings: CommonGroupSettings):
        super().__init__(dial_factory, settings)

    async def evaluate_async(self, config: Config, input: Input) -> Output:  # type: ignore[override]
        # ── Parse response ────────────────────────────────────────────────────
        try:
            extracted, field_meta = _extract_fields(input.endpoint_response)
        except Exception as e:
            err = MetricError(message=f"Failed to parse endpoint_response: {e}")
            return self.Output(
                main_fact_recall=err,
                recall_with_additionals=err,
                encounter_verdict=err,
                precision=err,
                agent_token_total=err,
            )

        token_total = _extract_token_total(input.endpoint_response)
        chain_nli = self._dial_factory.create_llm_with_schema(config.model, _NLIResult)

        # ── Separate targets by check type ────────────────────────────────────
        exact_targets = [t for t in input.target_fields if str(t.get("check", "")).lower() == "exact"]
        nli_targets = [t for t in input.target_fields if str(t.get("check", "")).lower() == "nli"]
        encounter_targets = [t for t in input.target_fields if str(t.get("check", "")).lower() == "encounter"]

        # ── Exact checks (synchronous) ────────────────────────────────────────
        # When expected is None, GT says the agent should return null/empty for this field.
        # Score 1.0 when actual is also None; 0.0 otherwise.
        exact_results = []
        for t in exact_targets:
            field_name = t.get("field_name", "")
            actual = extracted.get(field_name)
            expected = t.get("expected")
            if expected is None:
                hit = actual is None
            else:
                hit = (actual is not None) and (str(actual) == str(expected))
            exact_results.append({
                "field": field_name,
                "check": "exact",
                "hit": hit,
                "expected": expected,
                "extracted": actual,
                **_field_meta_entry(field_meta.get(field_name)),
            })

        # ── Encounter checks (synchronous) ────────────────────────────────────
        encounter_results = []
        for t in encounter_targets:
            field_name = t.get("field_name", "")
            actual = extracted.get(field_name)
            expected = t.get("expected") or {}
            score = _score_encounter(actual, expected)
            encounter_results.append({
                "field": field_name,
                "check": "encounter",
                "hit": score >= 1.0,
                "expected": expected,
                "extracted": actual,
            })

        # ── NLI checks (concurrent LLM calls) ────────────────────────────────
        nli_coros = []
        nli_norm_facts: list[list[dict]] = []  # parallel list for precision
        nli_actual_values: list[Any] = []
        for t in nli_targets:
            field_name = t.get("field_name", "")
            actual = extracted.get(field_name)
            raw_facts = t.get("facts") or []
            if not raw_facts:
                continue
            norm = _normalize_facts(raw_facts)
            nli_norm_facts.append(norm)
            nli_actual_values.append(actual)
            meta = field_meta.get(field_name)
            if actual is None:
                nli_coros.append(_missed_field_result(field_name, norm, actual_value=actual, meta=meta))
            else:
                nli_coros.append(_run_nli(chain_nli, field_name, str(actual), raw_facts, meta=meta))

        try:
            nli_results = list(await asyncio.gather(*nli_coros))
        except Exception as e:
            err = MetricError(message=f"NLI evaluation failed: {e}")
            return self.Output(
                main_fact_recall=err,
                recall_with_additionals=err,
                encounter_verdict=err,
                precision=err,
                agent_token_total=MetricOutputField(value=float(token_total)),
            )

        # ── Precision (optional 2-stage LLM) ─────────────────────────────────
        prec_value = -1.0
        prec_details: dict[str, Any] = {}
        if config.compute_precision:
            try:
                chain_decomp = self._dial_factory.create_llm_with_schema(config.model, _DecompResult)
                chain_verify = self._dial_factory.create_llm_with_schema(config.model, _NLIResult)
                prec_coros = []
                prec_field_index = []
                for i, (r, norm, actual) in enumerate(zip(nli_results, nli_norm_facts, nli_actual_values)):
                    if actual is not None:
                        prec_coros.append(
                            _run_precision_for_field(
                                chain_decomp, chain_verify, r["field"], str(actual), norm
                            )
                        )
                        prec_field_index.append(i)
                prec_field_results = list(await asyncio.gather(*prec_coros))
                total_output = sum(p["n_output"] for p in prec_field_results)
                total_in_scope = sum(p["n_in_scope"] for p in prec_field_results)
                if total_output > 0:
                    prec_value = round(total_in_scope / total_output, 3)
                prec_details = {
                    "n_output_facts": total_output,
                    "n_in_scope": total_in_scope,
                    "fields": prec_field_results,
                }
            except Exception as e:
                prec_details = {"error": f"Precision computation failed: {e}"}

        # ── Accumulate recall metrics ─────────────────────────────────────────
        facts_total = 0
        facts_main_ent = 0
        facts_score = 0.0
        field_details = []

        # Exact: one implicit fact per field (score 1.0 if hit, 0.0 otherwise)
        for r in exact_results:
            facts_total += 1
            if r["hit"]:
                facts_main_ent += 1
                facts_score += 1.0
            field_details.append(r)

        # NLI: one entry per fact
        for r in nli_results:
            facts_total += r["facts_total"]
            facts_main_ent += r["facts_main_ent"]
            facts_score += r["facts_score"]
            field_details.append(r)

        main_recall = round(facts_main_ent / facts_total, 3) if facts_total else None
        weighted_recall = round(facts_score / facts_total, 3) if facts_total else None

        # Encounter
        loc_hits = sum(1 for r in encounter_results if r["hit"])
        loc_total = len(encounter_results)
        if loc_total:
            enc_value = round(loc_hits / loc_total, 3)
        else:
            enc_value = -1.0  # sentinel: no encounter targets

        return self.Output(
            main_fact_recall=MetricOutputField(
                value=main_recall if main_recall is not None else 0.0,
                details={
                    "facts_main_ent": facts_main_ent,
                    "facts_total": facts_total,
                    "fields": field_details,
                },
            ),
            recall_with_additionals=MetricOutputField(
                value=weighted_recall if weighted_recall is not None else 0.0,
                details={"facts_score": round(facts_score, 3), "facts_total": facts_total},
            ),
            encounter_verdict=MetricOutputField(
                value=enc_value,
                details={
                    "loc_hits": loc_hits,
                    "loc_total": loc_total,
                    "fields": encounter_results,
                },
            ),
            precision=MetricOutputField(
                value=prec_value,
                details=prec_details,
            ),
            agent_token_total=MetricOutputField(
                value=float(token_total),
                details={},
            ),
        )
