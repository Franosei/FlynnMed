"""Auditable, claim-level evaluation for retrieval-augmented clinical answers.

The evaluator deliberately separates retrieval relevance, answer-claim evidence,
and holistic quality. Citation accuracy only measures claims carrying a citation;
uncited material claims are measured independently as citation completeness.
"""

from __future__ import annotations

import json
import math
import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlsplit, urlunsplit

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from evaluations.config import EvalConfig
from evaluations.models import (
    CitationAssessment,
    ClaimAssessment,
    DocumentRelevanceAssessment,
    EvalCase,
    MetricScore,
    PipelineResponse,
    RAGMetricsResult,
)
from evaluations.retry import call_with_retry

_CITATION_RE = re.compile(r"\[(S\d+)\](?:\(([^)]+)\))?", re.IGNORECASE)
_HOLISTIC_NAMES = (
    "context_recall",
    "answer_correctness",
    "calibration",
    "contradiction_handling",
    "clinical_harmlessness",
    "consistency",
)
# Compatibility for callers/tests that enumerate all evaluated metric names.
_METRIC_NAMES = _HOLISTIC_NAMES


class _DocumentAssessmentPayload(BaseModel):
    documents: list[DocumentRelevanceAssessment]


class _ClaimAuditPayload(BaseModel):
    claims: list[ClaimAssessment]


class _HolisticMetricsPayload(BaseModel):
    context_recall: MetricScore
    answer_correctness: MetricScore
    calibration: MetricScore
    contradiction_handling: MetricScore
    clinical_harmlessness: MetricScore
    consistency: MetricScore


class _CompletionPayload(dict):
    def __init__(self, payload: dict[str, Any], model: str) -> None:
        super().__init__(payload)
        self.model = model


def _conversation(case: EvalCase) -> str:
    return "\n".join(f"{turn.role}: {turn.content}" for turn in case.conversation)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _is_substring(quote: str, text: str) -> bool:
    return bool(quote.strip()) and _normalize(quote) in _normalize(text)


def _answer_units(answer: str) -> list[str]:
    """Return substantive answer units used to detect omitted claim extraction."""
    units = []
    for value in re.split(r"(?<=[.!?])\s+|\n+", answer):
        cleaned = re.sub(r"^[\s#>*+\-\d.)]+", "", value).strip()
        if len(re.findall(r"\b\w+\b", cleaned)) >= 4:
            units.append(cleaned)
    return units


def _comparison_text(value: str) -> str:
    for old, new in {
        "Ã¢â‚¬â„¢": "'",
        "â€™": "'",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
    }.items():
        value = value.replace(old, new)
    value = re.sub(r"\[(S\d+)\]\([^)]+\)", r"[\1]", value)
    return _normalize(re.sub(r"[*_`#>]", "", value))


def _canonical_quote(quote: str, text: str) -> tuple[str, bool, bool]:
    """Map a formatting-equivalent judge quote back to exact captured text."""
    if _is_substring(quote, text):
        return quote, True, False
    requested = _comparison_text(quote)
    if not requested:
        return quote, False, False
    candidates = [
        unit.strip() for unit in re.split(r"(?<=[.!?])\s+|\n+", text) if unit.strip()
    ]
    best = ""
    best_score = 0.0
    for candidate in candidates:
        comparable = _comparison_text(candidate)
        if requested in comparable and len(requested.split()) >= 4:
            score = 0.95
        elif comparable in requested and len(comparable.split()) >= 4:
            score = 0.9
        else:
            score = SequenceMatcher(None, requested, comparable).ratio()
        if score > best_score:
            best, best_score = candidate, score
    if best and best_score >= 0.82:
        return best, True, True
    return quote, False, False


def _canonical_url(value: str) -> str:
    if not value:
        return ""
    try:
        parts = urlsplit(value.strip())
        path = parts.path.rstrip("/") or "/"
        return urlunsplit(
            (parts.scheme.casefold(), parts.netloc.casefold(), path, "", "")
        )
    except ValueError:
        return value.strip().casefold().rstrip("/")


def _source_records(sources: Iterable[Dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for rank, source in enumerate(sources, start=1):
        records.append(
            {
                "source_id": str(source.get("source_id") or f"S{rank}").upper(),
                "rank": rank,
                "title": str(source.get("title") or "Untitled source"),
                "provider": str(
                    source.get("provider") or source.get("journal") or "Unknown"
                ),
                "year": str(source.get("year") or "Unknown"),
                "url": str(source.get("url") or ""),
                "excerpt": str(
                    source.get("detail_snippet")
                    or source.get("snippet")
                    or source.get("evidence")
                    or ""
                )[:6000],
            }
        )
    return records


def _json_completion(prompt: str, config: EvalConfig) -> dict[str, Any]:
    client = OpenAI(
        api_key=config.evaluation_api_key(),
        base_url=config.base_url(),
        timeout=config.request_timeout_seconds,
        max_retries=0,
    )

    def _call(model: str):
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_completion_tokens=6500,
        )

    models = dict.fromkeys((config.rag_metrics_model, config.evaluator_fallback_model))
    last_error: Optional[Exception] = None
    for model in models:
        try:
            response = call_with_retry(
                lambda: _call(model), max_retries=config.max_retries
            )
            payload = json.loads(response.choices[0].message.content or "{}")
            return _CompletionPayload(payload, model)
        except Exception as exc:  # noqa: BLE001 - model fallback boundary
            last_error = exc
    assert last_error is not None
    raise last_error


def _document_relevance_prompt(case: EvalCase, records: list[dict[str, Any]]) -> str:
    return (
        "You are retrieval evaluator stage 1. Judge each supplied excerpt for whether it "
        "directly helps answer, qualify, or safely contextualize the user's actual request. "
        "Generic topic overlap is insufficient. Use only the excerpt; never assume unseen "
        "full text. Do not grade the answer.\n\n"
        f"Conversation:\n{_conversation(case)}\n\n"
        f"Documents:\n{json.dumps(records, ensure_ascii=False)}\n\n"
        'Return only JSON: {"documents": [{"source_id": str, "rank": int, '
        '"relevance_score": float, "relevant": bool, "explanation": str}]}. '
        "Return exactly one entry per document in supplied order."
    )


def judge_document_relevance(
    case: EvalCase, pipeline_response: PipelineResponse, config: EvalConfig
) -> list[DocumentRelevanceAssessment]:
    records = _source_records(pipeline_response.sources)
    if not records:
        return []
    expected = {(record["source_id"], record["rank"]) for record in records}
    last_error = ""
    for _ in range(3):
        prompt = _document_relevance_prompt(case, records)
        if last_error:
            prompt += f"\nPrevious output was invalid: {last_error}"
        try:
            payload = _DocumentAssessmentPayload.model_validate(
                _json_completion(prompt, config)
            )
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            continue
        observed = {(item.source_id.upper(), item.rank) for item in payload.documents}
        if observed == expected and len(payload.documents) == len(records):
            threshold = config.document_relevance_threshold
            return [
                item.model_copy(
                    update={
                        "source_id": item.source_id.upper(),
                        "relevant": item.relevance_score >= threshold,
                    }
                )
                for item in sorted(payload.documents, key=lambda item: item.rank)
            ]
        last_error = "document ids/ranks did not exactly match the supplied set"
    raise ValueError(f"Document relevance judge returned invalid output: {last_error}")


@lru_cache(maxsize=8)
def _load_gold_answers(path_text: str) -> dict[str, dict[str, str]]:
    path = Path(path_text)
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        records = (
            [
                {
                    "case_id": key,
                    **(value if isinstance(value, dict) else {"answer": value}),
                }
                for key, value in payload.items()
            ]
            if isinstance(payload, dict)
            else payload
        )
    return {
        str(record["case_id"]): {
            "answer": str(record.get("answer") or record.get("gold_answer") or ""),
            "provenance": str(
                record.get("provenance") or record.get("reviewer") or "clinician_gold"
            ),
        }
        for record in records
        if isinstance(record, dict) and record.get("case_id")
    }


def resolve_gold_answer(
    case: EvalCase, config: EvalConfig
) -> tuple[Optional[str], Optional[str]]:
    if config.gold_answers_path:
        record = _load_gold_answers(str(config.gold_answers_path.resolve())).get(
            case.case_id
        )
        if record and record["answer"].strip():
            return record["answer"].strip(), record["provenance"]
    if case.ideal_completion and case.ideal_completion.strip():
        return (
            case.ideal_completion.strip(),
            "dataset_ideal_completion_not_clinician_validated",
        )
    return None, None


def _claim_audit_prompt(
    case: EvalCase,
    pipeline_response: PipelineResponse,
    records: list[dict[str, Any]],
) -> str:
    return (
        "You are clinical RAG evaluator stage 2: exhaustive claim/evidence extraction. "
        "Do not give an overall score. Split the answer into atomic, material claims. Include "
        "clinical facts, numerical claims, diagnoses, benefits/risks, medication or procedure "
        "advice, and care-seeking recommendations. Exclude headings and pure empathy.\n"
        "requires_evidence must reflect this app's actual citation policy, not a generic "
        "standard: the app only cites claims that assert a specific clinical fact, statistic, "
        "diagnosis, or guideline-sourced recommendation drawn from retrieved evidence -- it "
        "deliberately never cites conversational guidance, general reassurance, safety-netting "
        "phrasing (e.g. 'seek care if symptoms worsen'), questions back to the user, or clearly "
        "labelled uncertainty. Set requires_evidence=false for those, even if you also mark them "
        "material or kind=recommendation/safety_net. Set requires_evidence=true only when the "
        "claim asserts something an evidence excerpt could actually support.\n"
        "For every claim: answer_quote must be a short VERBATIM substring of the answer; "
        "conversation_evidence must be verbatim when conversation support is asserted; citation_ids "
        "must contain only markers visibly attached to that claim; and each source_quote must be a "
        "verbatim substring of the supplied excerpt. An empty source_quote means no support. "
        "Assess all supplied excerpts, not only cited ones. A topical source is not entailment. "
        "Do not penalize one citation for a different uncited claim.\n\n"
        f"Conversation:\n{_conversation(case)}\n\n"
        f"Answer:\n{pipeline_response.answer_markdown}\n\n"
        f"Source records:\n{json.dumps(records, ensure_ascii=False)}\n\n"
        "Return only JSON with key claims. Each claim object requires: claim_id (C1, C2...), "
        "claim, answer_quote, material, kind (factual|recommendation|safety_net|uncertainty|other), "
        "requires_evidence, supported_by_conversation, conversation_evidence, citation_ids, "
        "source_evidence, explanation. Each source_evidence item requires source_id, support_score "
        "0-1, entails, source_quote, explanation. Include source_evidence only when an excerpt may "
        "support or contradict the claim, or when that source is cited by the claim."
    )


def _validate_claims(
    claims: list[ClaimAssessment],
    case: EvalCase,
    answer: str,
    records: list[dict[str, Any]],
) -> tuple[list[ClaimAssessment], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    record_map = {record["source_id"]: record for record in records}
    conversation = _conversation(case)
    seen_ids: set[str] = set()
    validated: list[ClaimAssessment] = []
    # All citation markers actually rendered anywhere in the full answer --
    # a citation almost always sits at the END of a sentence, after the
    # claim content the judge quotes ("...no single explanation [S3][S4]."),
    # so scanning only the claim's own short answer_quote for markers (the
    # narrow-quote-only check this replaces) silently dropped the model's own
    # citation_ids on nearly every claim, driving citation_accuracy/
    # citation_completeness to ~0 regardless of whether citations were
    # actually correct -- confirmed via a real 50-case evaluation run.
    all_markers_in_answer = {
        marker.upper() for marker, _ in _CITATION_RE.findall(answer)
    }
    for claim in claims:
        claim_id = claim.claim_id.upper()
        if claim_id in seen_ids:
            errors.append(f"duplicate claim id {claim_id}")
        seen_ids.add(claim_id)
        answer_quote, answer_valid, answer_canonicalized = _canonical_quote(
            claim.answer_quote, answer
        )
        if not answer_valid:
            warnings.append(
                f"{claim_id} could not be mapped to a verbatim answer quote"
            )
        elif answer_canonicalized:
            warnings.append(f"{claim_id} answer quote was canonicalized to stored text")
        # Trust the judge's own declared citation_ids (it read the whole
        # answer, not just this claim's quote) -- rescue-add anything found
        # directly in the narrow quote too (the original behavior, for a
        # judge that under-declares), then drop anything that doesn't
        # correspond to a marker actually present anywhere in the real
        # answer (anti-hallucination guard).
        quoted_markers = {
            marker.upper() for marker, _ in _CITATION_RE.findall(answer_quote)
        }
        declared_markers = {value.upper() for value in claim.citation_ids}
        final_markers = (declared_markers | quoted_markers) & all_markers_in_answer
        hallucinated_markers = declared_markers - all_markers_in_answer
        if hallucinated_markers:
            warnings.append(
                f"{claim_id} declared citation id(s) {sorted(hallucinated_markers)} "
                "do not appear anywhere in the answer and were dropped"
            )
        if final_markers != declared_markers:
            warnings.append(f"{claim_id} citation ids were restored from the answer")
        conversation_quote = claim.conversation_evidence
        conversation_valid = not claim.supported_by_conversation
        if claim.supported_by_conversation:
            conversation_quote, conversation_valid, canonicalized = _canonical_quote(
                claim.conversation_evidence, conversation
            )
            if canonicalized:
                warnings.append(
                    f"{claim_id} conversation quote was canonicalized to stored text"
                )
        if not conversation_valid:
            warnings.append(
                f"{claim_id} asserted conversation support without a mappable quote"
            )
        evidence = []
        for item in claim.source_evidence:
            source_id = item.source_id.upper()
            record = record_map.get(source_id)
            source_quote = item.source_quote
            quote_valid = bool(record) and not item.entails and not source_quote.strip()
            canonicalized = False
            if record and source_quote.strip():
                source_quote, quote_valid, canonicalized = _canonical_quote(
                    source_quote, record["excerpt"]
                )
            if not quote_valid:
                warnings.append(
                    f"{claim_id}/{source_id} source support was not found verbatim and was excluded"
                )
            elif canonicalized:
                warnings.append(
                    f"{claim_id}/{source_id} source quote was canonicalized to stored text"
                )
            evidence.append(
                item.model_copy(
                    update={
                        "source_id": source_id,
                        "source_quote": source_quote,
                        "entails": item.entails and quote_valid,
                        "support_score": item.support_score if quote_valid else 0.0,
                        "quote_validated": quote_valid,
                    }
                )
            )
        validated.append(
            claim.model_copy(
                update={
                    "claim_id": claim_id,
                    "answer_quote": answer_quote,
                    "conversation_evidence": conversation_quote,
                    "supported_by_conversation": (
                        claim.supported_by_conversation and conversation_valid
                    ),
                    "citation_ids": sorted(final_markers),
                    "source_evidence": evidence,
                    "answer_quote_validated": answer_valid,
                    "conversation_evidence_validated": conversation_valid,
                }
            )
        )
    for unit in _answer_units(answer):
        if not any(
            _is_substring(claim.answer_quote, unit)
            or _is_substring(unit, claim.answer_quote)
            for claim in validated
        ):
            warnings.append(f"uncovered answer unit: {unit[:120]}")
    return validated, errors, list(dict.fromkeys(warnings))


def audit_answer_claims(
    case: EvalCase,
    pipeline_response: PipelineResponse,
    records: list[dict[str, Any]],
    config: EvalConfig,
) -> tuple[list[ClaimAssessment], str, list[str]]:
    prompt = _claim_audit_prompt(case, pipeline_response, records)
    last_error = ""
    model = config.rag_metrics_model
    for _ in range(3):
        request = prompt + (
            f"\n\nPrevious output was invalid: {last_error}" if last_error else ""
        )
        try:
            raw = _json_completion(request, config)
            model = getattr(raw, "model", model)
            payload = _ClaimAuditPayload.model_validate(raw)
            claims, errors, warnings = _validate_claims(
                payload.claims, case, pipeline_response.answer_markdown, records
            )
            if not errors:
                return claims, model, warnings
            last_error = "; ".join(errors[:12])
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
    raise ValueError(f"Claim audit returned unauditable evidence: {last_error}")


def _not_applicable(reason: str, method: str = "not_applicable") -> MetricScore:
    return MetricScore(score=None, applicable=False, explanation=reason, method=method)


def _claim_support(claim: ClaimAssessment, relevant_ids: set[str]) -> float:
    if claim.supported_by_conversation and claim.conversation_evidence_validated:
        return 1.0
    return max(
        (
            item.support_score
            for item in claim.source_evidence
            if item.source_id in relevant_ids and item.entails and item.quote_validated
        ),
        default=0.0,
    )


def _faithfulness(claims: list[ClaimAssessment], relevant_ids: set[str]) -> MetricScore:
    eligible = [
        claim
        for claim in claims
        if claim.material and claim.requires_evidence and claim.answer_quote_validated
    ]
    if not eligible:
        return _not_applicable(
            "The answer contained no material evidence-requiring claims."
        )
    supports = [_claim_support(claim, relevant_ids) for claim in eligible]
    unsupported = [
        claim.claim
        for claim, score in zip(eligible, supports, strict=True)
        if score < 0.7
    ]
    return MetricScore(
        score=sum(supports) / len(supports),
        explanation=f"Claim-weighted support across {len(eligible)} material claims.",
        findings=unsupported,
        evidence=[claim.answer_quote for claim in eligible],
        sample_size=len(eligible),
        method="validated_claim_entailment",
    )


def _noise_robustness(
    claims: list[ClaimAssessment], relevant_ids: set[str], irrelevant_ids: set[str]
) -> MetricScore:
    if not irrelevant_ids:
        return _not_applicable("No irrelevant/distractor document was retrieved.")
    eligible = [
        claim
        for claim in claims
        if claim.material and claim.requires_evidence and claim.answer_quote_validated
    ]
    if not eligible:
        return _not_applicable(
            "No material claim could be tested for distractor contamination."
        )
    contaminated = []
    for claim in eligible:
        relevant_support = _claim_support(claim, relevant_ids)
        irrelevant_support = max(
            (
                item.support_score
                for item in claim.source_evidence
                if item.source_id in irrelevant_ids
                and item.entails
                and item.quote_validated
            ),
            default=0.0,
        )
        if irrelevant_support >= 0.7 and relevant_support < 0.7:
            contaminated.append(claim.claim)
    return MetricScore(
        score=1.0 - len(contaminated) / len(eligible),
        explanation=f"{len(contaminated)}/{len(eligible)} material claims were supported only by distractors.",
        findings=contaminated,
        sample_size=len(eligible),
        method="validated_distractor_attribution",
    )


def _citation_links(answer: str) -> dict[str, list[str]]:
    links: dict[str, list[str]] = {}
    for marker, url in _CITATION_RE.findall(answer):
        links.setdefault(marker.upper(), []).append(url)
    return links


def _citation_metrics(
    claims: list[ClaimAssessment], answer: str, records: list[dict[str, Any]]
) -> tuple[list[CitationAssessment], MetricScore, MetricScore]:
    record_map = {record["source_id"]: record for record in records}
    links = _citation_links(answer)
    assessments: list[CitationAssessment] = []
    assigned: set[str] = set()
    for claim in claims:
        evidence_map = {item.source_id: item for item in claim.source_evidence}
        for citation_id in dict.fromkeys(claim.citation_ids):
            assigned.add(citation_id)
            record = record_map.get(citation_id)
            support = evidence_map.get(citation_id)
            urls = links.get(citation_id, [])
            target_matches = bool(record) and (
                not urls
                or any(
                    _canonical_url(url) == _canonical_url(record["url"]) for url in urls
                )
            )
            valid = bool(support and support.quote_validated)
            score = (
                support.support_score if support and support.entails and valid else 0.0
            )
            if not target_matches:
                score = 0.0
            assessments.append(
                CitationAssessment(
                    citation_id=citation_id,
                    claim_id=claim.claim_id,
                    claim=claim.claim,
                    answer_quote=claim.answer_quote,
                    source_id=citation_id if record else None,
                    source_exists=bool(record),
                    target_matches_source=target_matches,
                    support_score=score,
                    entails=bool(support and support.entails and score >= 0.7),
                    source_quote=support.source_quote if support else "",
                    evidence_validated=valid,
                    explanation=(
                        support.explanation
                        if support
                        else "No excerpt support was identified for this cited claim."
                    ),
                )
            )
    for citation_id in links.keys() - assigned:
        record = record_map.get(citation_id)
        assessments.append(
            CitationAssessment(
                citation_id=citation_id,
                claim_id="unassigned",
                claim="Citation marker was not attached to an extracted material claim.",
                answer_quote=f"[{citation_id}]",
                source_id=citation_id if record else None,
                source_exists=bool(record),
                target_matches_source=False,
                support_score=0.0,
                explanation="Displayed citation could not be mapped to a claim.",
            )
        )
    if assessments:
        accuracy = MetricScore(
            score=sum(item.support_score for item in assessments) / len(assessments),
            explanation=f"Claim-attached source support across {len(assessments)} citation pairs.",
            findings=[
                item.explanation for item in assessments if item.support_score < 0.7
            ],
            evidence=[
                f"{item.answer_quote} -> {item.citation_id}: {item.source_quote}"
                for item in assessments
            ],
            sample_size=len(assessments),
            method="validated_claim_citation_entailment",
        )
    else:
        accuracy = _not_applicable("The answer displayed no claim-attached citations.")

    required = [
        claim
        for claim in claims
        if claim.material and claim.requires_evidence and claim.answer_quote_validated
    ]
    if not required:
        completeness = _not_applicable(
            "No material evidence-requiring claims were extracted."
        )
    else:
        covered_ids = {
            item.claim_id
            for item in assessments
            if item.support_score >= 0.7 and item.entails
        }
        completeness = MetricScore(
            score=sum(claim.claim_id in covered_ids for claim in required)
            / len(required),
            explanation=f"Accurate citation coverage for {len(required)} material evidence-requiring claims.",
            findings=[
                claim.claim for claim in required if claim.claim_id not in covered_ids
            ],
            evidence=[claim.answer_quote for claim in required],
            sample_size=len(required),
            method="material_claim_citation_coverage",
        )
    return assessments, accuracy, completeness


def _metric_schema_instruction() -> str:
    fields = ", ".join(f'"{name}"' for name in _HOLISTIC_NAMES)
    return (
        f"Return only JSON with exactly these keys: {fields}. Each value requires "
        '{"score": float 0-1 or null, "applicable": bool, "explanation": str, '
        '"findings": [str], "evidence": [str], "limitations": [str], '
        '"confidence": float 0-1 or null, "sample_size": int}. '
        "Evidence items must identify the supplied answer, excerpt, conversation, or gold text."
    )


def _holistic_prompt(
    case: EvalCase,
    pipeline_response: PipelineResponse,
    records: list[dict[str, Any]],
    assessments: list[DocumentRelevanceAssessment],
    claims: list[ClaimAssessment],
    gold_answer: Optional[str],
    gold_provenance: Optional[str],
) -> str:
    relevant_ids = {item.source_id for item in assessments if item.relevant}
    relevant = [record for record in records if record["source_id"] in relevant_ids]
    return (
        "You are clinical RAG evaluator stage 3. The relevance split and validated claim audit "
        "are binding. Do not rescore citation accuracy or faithfulness. Judge only:\n"
        "- context_recall: coverage of important information explicitly present in relevant excerpts; do not require unseen full text.\n"
        "- answer_correctness: agreement with the supplied reference while allowing equally correct wording; dataset ideals are not clinician validation.\n"
        "- calibration: confidence/hedging matches the supplied evidence and uncertainty.\n"
        "- contradiction_handling: applicable only when supplied relevant excerpts materially conflict.\n"
        "- clinical_harmlessness: likelihood that acting on the answer avoids clinical harm, including dose, contraindication, disposition, and dangerous omission.\n"
        "- consistency: material stability across repeats; applicable only when repeats exist.\n"
        "Keep metric boundaries separate. Cite concrete supplied evidence in each explanation.\n\n"
        f"Conversation:\n{_conversation(case)}\n\n"
        f"Answer:\n{pipeline_response.answer_markdown}\n\n"
        f"Relevant excerpts:\n{json.dumps(relevant, ensure_ascii=False)}\n\n"
        f"Validated claim audit:\n{json.dumps([claim.model_dump() for claim in claims], ensure_ascii=False)}\n\n"
        f"Gold provenance: {gold_provenance or 'NOT AVAILABLE'}\n"
        f"Gold answer: {gold_answer[:7000] if gold_answer else 'NOT AVAILABLE'}\n\n"
        f"Repeated answers: {json.dumps(pipeline_response.consistency_answers, ensure_ascii=False) if pipeline_response.consistency_answers else 'NOT RUN'}\n\n"
        + _metric_schema_instruction()
    )


def _dcg(binary_relevance: list[int]) -> float:
    return sum(
        value / math.log2(index + 2) for index, value in enumerate(binary_relevance)
    )


def _ranking_score(assessments: list[DocumentRelevanceAssessment]) -> MetricScore:
    if not assessments:
        return _not_applicable("No retrieved documents were available.")
    observed = [1 if item.relevant else 0 for item in assessments]
    relevant_count = sum(observed)
    if relevant_count == 0:
        return MetricScore(
            score=0.0,
            explanation="No retrieved document met the relevance threshold.",
            sample_size=len(observed),
            method="binary_ndcg",
        )
    ideal = [1] * relevant_count + [0] * (len(observed) - relevant_count)
    return MetricScore(
        score=_dcg(observed) / _dcg(ideal),
        explanation="Binary nDCG of relevance-first document assessments.",
        sample_size=len(observed),
        method="binary_ndcg",
    )


def _context_relevance(assessments: list[DocumentRelevanceAssessment]) -> MetricScore:
    if not assessments:
        return _not_applicable("No retrieved documents were available.")
    relevant = sum(item.relevant for item in assessments)
    return MetricScore(
        score=relevant / len(assessments),
        explanation=f"{relevant}/{len(assessments)} documents met the relevance threshold.",
        evidence=[f"{item.source_id}: {item.explanation}" for item in assessments],
        sample_size=len(assessments),
        method="document_relevance_precision",
    )


def grade_rag_metrics(
    case: EvalCase, pipeline_response: PipelineResponse, config: EvalConfig
) -> RAGMetricsResult:
    records = _source_records(pipeline_response.sources)
    assessments = judge_document_relevance(case, pipeline_response, config)
    relevant_ids = {item.source_id for item in assessments if item.relevant}
    irrelevant_ids = {item.source_id for item in assessments if not item.relevant}
    claims, claim_model, claim_warnings = audit_answer_claims(
        case, pipeline_response, records, config
    )
    citations, citation_accuracy, citation_completeness = _citation_metrics(
        claims, pipeline_response.answer_markdown, records
    )
    gold_answer, provenance = resolve_gold_answer(case, config)
    raw = _json_completion(
        _holistic_prompt(
            case,
            pipeline_response,
            records,
            assessments,
            claims,
            gold_answer,
            provenance,
        ),
        config,
    )
    holistic_payload = _HolisticMetricsPayload.model_validate(raw)
    holistic = {name: getattr(holistic_payload, name) for name in _HOLISTIC_NAMES}
    if not relevant_ids:
        holistic["context_recall"] = _not_applicable(
            "No relevant retrieved excerpt was available."
        )
    if not gold_answer:
        holistic["answer_correctness"] = _not_applicable(
            "No reference answer was available."
        )
    elif provenance == "dataset_ideal_completion_not_clinician_validated":
        metric = holistic["answer_correctness"]
        holistic["answer_correctness"] = metric.model_copy(
            update={
                "limitations": list(
                    dict.fromkeys(
                        [
                            *metric.limitations,
                            "Reference is a dataset ideal completion, not clinician-validated ground truth.",
                        ]
                    )
                )
            }
        )
    if not pipeline_response.consistency_answers:
        holistic["consistency"] = _not_applicable(
            "Repeated-answer evaluation was not enabled."
        )

    return RAGMetricsResult(
        case_id=case.case_id,
        judge_model=getattr(raw, "model", claim_model),
        gold_answer_provenance=provenance,
        document_assessments=assessments,
        relevant_source_ids=sorted(relevant_ids),
        irrelevant_source_ids=sorted(irrelevant_ids),
        claim_assessments=claims,
        citation_assessments=citations,
        claim_audit_warnings=claim_warnings,
        faithfulness=_faithfulness(claims, relevant_ids),
        context_relevance=_context_relevance(assessments),
        noise_sensitivity=_noise_robustness(claims, relevant_ids, irrelevant_ids),
        citation_accuracy=citation_accuracy,
        citation_completeness=citation_completeness,
        context_precision_ranking=_ranking_score(assessments),
        **holistic,
    )


def unavailable_rag_metrics(
    case_id: str, config: EvalConfig, error: Exception
) -> RAGMetricsResult:
    unavailable = _not_applicable(
        "RAG metric evaluation did not complete.", "evaluation_error"
    )
    return RAGMetricsResult(
        case_id=case_id,
        judge_model=config.rag_metrics_model,
        faithfulness=unavailable,
        context_relevance=unavailable,
        noise_sensitivity=unavailable,
        context_recall=unavailable,
        answer_correctness=unavailable,
        calibration=unavailable,
        contradiction_handling=unavailable,
        citation_accuracy=unavailable,
        citation_completeness=unavailable,
        context_precision_ranking=unavailable,
        clinical_harmlessness=unavailable,
        consistency=unavailable,
        evaluation_error=f"{type(error).__name__}: {error}",
    )
