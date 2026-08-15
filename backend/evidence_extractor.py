"""
Evidence extractor: structured extraction layer between retrieved sources and the LLM.

For each ranked source, uses gpt-4o-mini to extract patient-specific facts
into an ArticleEvidence JSON object. Only these objects are forwarded to the
answer model -- never raw unprocessed chunks.

Why this matters:
- Prevents the LLM from hallucinating facts not present in the source
- Ensures population mismatches are surfaced before answer generation
- Reduces token noise: the LLM sees 4-6 structured extractions, not raw passages
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from backend.evidence_schema import (
    ArticleEvidence,
    ExtractedEvidenceDossier,
    PatientAlignmentFact,
    StructuredClaim,
)
from backend.models.evidence import CERTAINTY_LEVELS, STUDY_DESIGNS
from backend.user_store import compute_current_age


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _canonicalize_passage(quote: str, source_text: str) -> Optional[str]:
    """
    Maps a claimed extracted_passages quote back to genuine source text,
    tolerating whitespace/case differences the model introduces while
    copying -- never invents or paraphrases text, only recovers verbatim
    source content the model's quote closely matches. Returns None (reject)
    rather than a paraphrase when no confident match exists. Ported from
    evaluations/rag_metrics.py::_canonical_quote, the same proven approach
    already used for claim-quote validation elsewhere in this codebase --
    evaluations/ and backend/ don't share utils today, so this is a focused
    copy, not an import.
    """
    if not quote.strip():
        return None
    normalized_quote = _normalize(quote)
    normalized_text = _normalize(source_text)
    if normalized_quote in normalized_text:
        # Exact match modulo whitespace/case -- the model's own quote text
        # is already verbatim, return it as-is.
        return quote.strip()

    candidates = [
        unit.strip()
        for unit in re.split(r"(?<=[.!?])\s+|\n+", source_text)
        if unit.strip()
    ]
    best = ""
    best_score = 0.0
    for candidate in candidates:
        comparable = _normalize(candidate)
        if normalized_quote in comparable and len(normalized_quote.split()) >= 4:
            score = 0.95
        elif comparable in normalized_quote and len(comparable.split()) >= 4:
            score = 0.9
        else:
            score = SequenceMatcher(None, normalized_quote, comparable).ratio()
        if score > best_score:
            best, best_score = candidate, score
    if best and best_score >= 0.82:
        return best
    return None


def _build_patient_summary(user_profile: dict, patient_history_ctx=None) -> str:
    """Build a brief, structured patient profile string for extraction alignment."""
    parts: List[str] = []

    age = compute_current_age(user_profile.get("date_of_birth", ""))
    sex = user_profile.get("biological_sex", "")
    if age:
        parts.append(f"Age {age}")
    if sex and sex.lower() not in ("prefer not to say", ""):
        parts.append(sex)

    if patient_history_ctx and not patient_history_ctx.is_empty():
        block = patient_history_ctx.as_prompt_block()
        # Take first 400 chars -- enough for alignment, keeps token cost low
        parts.append(block[:400])

    return "; ".join(parts) if parts else "Patient profile not recorded"


def _extract_one_article(
    llm,
    source: Dict,
    question: str,
    patient_summary: str,
    medications: List[str],
    conditions: List[str],
) -> ArticleEvidence:
    """
    Call gpt-4o-mini to fill ArticleEvidence for one source.
    Falls back to a minimal structural extraction if the LLM call fails.
    """
    from backend.summarizer import LLMHelper

    snippet = (
        source.get("snippet")
        or source.get("detail_snippet")
        or source.get("text", "")
    )[:800]
    title = source.get("title", "Untitled")
    source_id = source.get("source_id", "S?")

    if not snippet:
        return ArticleEvidence(
            source_id=source_id,
            title=title,
            journal=source.get("journal"),
            year=str(source.get("year", "")),
            url=source.get("url"),
            evidence_tier=source.get("evidence_tier", 3),
            tier_label=source.get("tier_label", "Research"),
            answers_question=False,
            patient_relevant_summary="No usable text available from this source.",
            source_snippet="",
        )

    prompt = (
        "You are a clinical evidence extractor. Extract patient-specific facts from the "
        "article below and return ONLY a valid JSON object.\n\n"
        f"PATIENT PROFILE: {patient_summary}\n"
        f"PATIENT CONDITIONS: {', '.join(conditions) or 'None recorded'}\n"
        f"PATIENT MEDICATIONS: {', '.join(medications) or 'None recorded'}\n"
        f"PATIENT QUESTION: {question}\n\n"
        f"ARTICLE TITLE: {title}\n"
        f"ARTICLE TEXT:\n{snippet}\n\n"
        "Return JSON with these exact fields:\n"
        "{\n"
        '  "answers_question": true/false,\n'
        '  "question_facts": ["direct fact from article that answers the question"],\n'
        '  "patient_aligned_facts": [\n'
        '    {"category": "condition|medication|vital|demographic|allergy", '
        '"patient_value": "exact value from patient profile", '
        '"article_statement": "what article says about it", '
        '"relevance_type": "direct_evidence|contraindication|drug_interaction|population_match"}\n'
        "  ],\n"
        '  "contraindications": ["contraindication relevant to this patient"],\n'
        '  "drug_interactions": ["interaction involving this patient medications"],\n'
        '  "patient_relevant_summary": "2-3 sentences on what this article contributes for this patient",\n'
        '  "extracted_passages": ["verbatim substring of ARTICLE TEXT above that supports '
        'question_facts/contraindications/drug_interactions -- copy the exact wording, do not '
        'paraphrase or summarise"],\n'
        '  "structured_claims": [\n'
        '    {"claim_text": "normalised statement of the finding", '
        '"population": "who the finding applies to", '
        '"intervention": "the treatment/exposure studied", '
        '"comparator": "what it was compared against, if any", '
        '"outcome": "the measured result", '
        '"study_design": "systematic_review|meta_analysis|rct|cohort_study|case_control|'
        'case_report|clinical_guideline|narrative_review|expert_opinion|unknown", '
        '"certainty": "high|moderate|low|very_low|unknown", '
        '"exact_quote": "verbatim substring of ARTICLE TEXT supporting this claim"}\n'
        "  ],\n"
        '  "alignment_confidence": 0.0-1.0,\n'
        '  "specialty_mismatch": true/false,\n'
        '  "specialty_mismatch_reason": "one sentence, only if specialty_mismatch is true"\n'
        "}\n\n"
        "RULES:\n"
        "- Only include facts explicitly in the article text -- never infer\n"
        "- extracted_passages entries MUST be an exact, verbatim substring of ARTICLE TEXT -- "
        "copy-paste the wording exactly, including punctuation. A passage that is not an exact "
        "substring will be discarded, so do not summarise, combine, or lightly reword it.\n"
        "- structured_claims: ONLY populate this when the article states a genuine comparative or "
        "interventional finding with a clear population/intervention/outcome (e.g. a trial or "
        "guideline recommendation) -- return an empty list for anything else (dosing "
        "instructions, general advice, a page with no comparator). Do not force a claim that "
        "isn't really there. exact_quote MUST be an exact, verbatim substring of ARTICLE TEXT, "
        "same rule as extracted_passages -- a claim whose quote cannot be verified will be "
        "discarded entirely.\n"
        "- patient_aligned_facts must reference actual values from the patient profile\n"
        "- If article does not match patient's conditions/meds, set patient_aligned_facts: []\n"
        "- SPECIALTY/MEANING MISMATCH: set specialty_mismatch to true ONLY if the PATIENT PROFILE "
        "above affirmatively confirms one specific clinical meaning of an ambiguous term (e.g. it "
        "states a urology peak urinary flow rate reading) AND this article discusses a different "
        "clinical meaning of that same ambiguous term (e.g. respiratory peak-flow guidance). This "
        "requires an actual stated conflict between two concrete facts -- never infer or guess a "
        "mismatch. If PATIENT PROFILE is 'Patient profile not recorded' or otherwise gives no "
        "concrete condition/measurement to conflict with, there is nothing for the article to "
        "mismatch against: specialty_mismatch MUST be false in that case, regardless of how "
        "unrelated the article may seem to the question -- use alignment_confidence and "
        "answers_question for topical relevance instead, never specialty_mismatch. This is a hard "
        "exclusion signal, independent of confidence scoring -- set it true whenever the mismatch "
        "is real and concretely stated, even if the article otherwise reads as well-written or "
        "superficially on-topic. When true, also set answers_question to false, "
        "patient_aligned_facts to [], alignment_confidence to 0.0, and patient_relevant_summary "
        "to state plainly that this source concerns a different measurement/condition and does "
        "not apply to this patient's confirmed reading.\n"
        "- alignment_confidence: 1.0 = directly addresses patient's conditions; 0.0 = irrelevant\n"
        "- NO PATIENT PROFILE: if PATIENT PROFILE above is 'Patient profile not recorded', this is "
        "a general/professional evidence question with no specific patient in view (e.g. a "
        "clinician asking about guidelines in the abstract) -- there is no patient to align facts "
        "to. In that case patient_aligned_facts must be [], but answers_question and "
        "alignment_confidence must still be scored on topical relevance to PATIENT QUESTION alone: "
        "set answers_question true and alignment_confidence high (0.7+) whenever the article "
        "directly covers the same clinical topic the question asks about (same condition, "
        "procedure, or guideline area), even though there are no patient-specific facts to match. "
        "Do not lower these scores merely because there is no patient profile to align to.\n"
        "- Return ONLY the JSON object"
    )

    try:
        response = llm.client.chat.completions.create(
            model=LLMHelper.AUX_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_completion_tokens=700,
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)

        aligned_facts: List[PatientAlignmentFact] = []
        for fact_dict in data.get("patient_aligned_facts", []):
            try:
                aligned_facts.append(PatientAlignmentFact(**fact_dict))
            except Exception:
                pass

        # Evidence Ledger Phase 1: a claimed passage that isn't an exact
        # substring of the article text is dropped, not kept in softened/
        # paraphrased form -- excluded rather than passed through as
        # unverified background text, matching the same principle already
        # applied to specialty_mismatch exclusion above.
        verified_passages: List[str] = []
        for quote in data.get("extracted_passages", []):
            if not isinstance(quote, str):
                continue
            canonical = _canonicalize_passage(quote, snippet)
            if canonical and canonical not in verified_passages:
                verified_passages.append(canonical)
        verified_passages = verified_passages[:6]

        # Evidence Ledger Phase 2: same verbatim-quote discipline as
        # extracted_passages above -- a structured claim whose exact_quote
        # can't be verified against the article text is dropped entirely,
        # never kept with an unverified quote. Unrecognised study_design/
        # certainty values are coerced to "unknown" rather than rejected,
        # since the LLM's classification is a best-effort label, not a gate.
        structured_claims: List[StructuredClaim] = []
        for claim_dict in data.get("structured_claims", []):
            if not isinstance(claim_dict, dict):
                continue
            quote = claim_dict.get("exact_quote", "")
            if not isinstance(quote, str):
                continue
            canonical_quote = _canonicalize_passage(quote, snippet)
            if not canonical_quote:
                continue
            claim_text = str(claim_dict.get("claim_text", "")).strip()
            if not claim_text:
                continue
            study_design = str(claim_dict.get("study_design", "unknown")).strip().lower()
            if study_design not in STUDY_DESIGNS:
                study_design = "unknown"
            certainty = str(claim_dict.get("certainty", "unknown")).strip().lower()
            if certainty not in CERTAINTY_LEVELS:
                certainty = "unknown"
            structured_claims.append(
                StructuredClaim(
                    claim_text=claim_text[:500],
                    population=str(claim_dict.get("population", ""))[:512],
                    intervention=str(claim_dict.get("intervention", ""))[:512],
                    comparator=str(claim_dict.get("comparator", ""))[:512],
                    outcome=str(claim_dict.get("outcome", ""))[:512],
                    study_design=study_design,
                    certainty=certainty,
                    exact_quote=canonical_quote,
                )
            )
        structured_claims = structured_claims[:4]

        return ArticleEvidence(
            source_id=source_id,
            title=title,
            journal=source.get("journal"),
            year=str(source.get("year", "")),
            url=source.get("url"),
            evidence_tier=source.get("evidence_tier", 3),
            tier_label=source.get("tier_label", "Research"),
            answers_question=bool(data.get("answers_question", False)),
            question_facts=data.get("question_facts", [])[:6],
            patient_aligned_facts=aligned_facts[:4],
            contraindications=data.get("contraindications", [])[:4],
            drug_interactions=data.get("drug_interactions", [])[:4],
            patient_relevant_summary=str(data.get("patient_relevant_summary", ""))[:500],
            extracted_passages=verified_passages,
            structured_claims=structured_claims,
            alignment_confidence=float(data.get("alignment_confidence", 0.5)),
            specialty_mismatch=bool(data.get("specialty_mismatch", False)),
            specialty_mismatch_reason=str(data.get("specialty_mismatch_reason", ""))[:300],
            source_snippet=snippet,
        )

    except Exception as exc:
        print(f"[EvidenceExtractor] Extraction failed for {source_id}: {exc}")
        return ArticleEvidence(
            source_id=source_id,
            title=title,
            journal=source.get("journal"),
            year=str(source.get("year", "")),
            url=source.get("url"),
            evidence_tier=source.get("evidence_tier", 3),
            tier_label=source.get("tier_label", "Research"),
            answers_question=False,
            patient_relevant_summary=(snippet[:200] + "…") if snippet else "Extraction unavailable.",
            source_snippet=snippet,
        )


def build_evidence_dossier(
    llm,
    sources: List[Dict],
    question: str,
    user_profile: dict,
    patient_history_ctx=None,
    medications: Optional[List[Dict]] = None,
    conditions: Optional[List[Dict]] = None,
) -> ExtractedEvidenceDossier:
    """
    Build a complete structured evidence dossier for the given question + patient.
    Runs extraction for up to 6 sources in parallel (ThreadPoolExecutor).
    """
    patient_summary = _build_patient_summary(user_profile, patient_history_ctx)
    has_patient_profile = patient_summary != "Patient profile not recorded"
    med_names = [m.get("name", "") for m in (medications or []) if m.get("name")]
    cond_names = [c.get("name", "") for c in (conditions or []) if c.get("name")]

    articles: List[ArticleEvidence] = []
    top_sources = sources[:6]

    if top_sources:
        with ThreadPoolExecutor(max_workers=min(4, len(top_sources))) as executor:
            futures = [
                executor.submit(
                    _extract_one_article,
                    llm, source, question, patient_summary, med_names, cond_names,
                )
                for source in top_sources
            ]
            for future in futures:
                try:
                    articles.append(future.result())
                except Exception as exc:
                    print(f"[EvidenceExtractor] Worker failed: {exc}")

    # specialty_mismatch means "conflicts with a concrete fact in the patient's own profile" --
    # with no profile on record there's nothing to conflict with, so a true flag here can only be
    # an aux-model misfire (belt-and-suspenders alongside the prompt instruction, which the model
    # doesn't always follow). The same extraction call is also instructed to zero out
    # answers_question/alignment_confidence as a side effect whenever it sets specialty_mismatch,
    # so undoing the flag alone isn't enough -- those forced-low values would still trip the
    # separate low-confidence exclusion below. Reset the whole cascade: treat the source as
    # topically relevant (it was retrieved by a query derived from this exact question) with
    # neutral, unverified patient-alignment confidence.
    if not has_patient_profile:
        for a in articles:
            if a.specialty_mismatch:
                a.specialty_mismatch = False
                a.specialty_mismatch_reason = ""
                a.answers_question = True
                a.alignment_confidence = 0.5
                if not a.patient_relevant_summary:
                    a.patient_relevant_summary = "General topical context for this question."

    # Sort: highest alignment confidence first
    articles.sort(key=lambda a: a.alignment_confidence, reverse=True)

    # Sources the extractor confirmed concern a different clinical meaning of an ambiguous term
    # (or are otherwise irrelevant) are excluded entirely -- they must never reach the answer
    # prompt, even as "general context", since that's exactly how wrong-specialty guidance leaks
    # into an otherwise-correct answer. specialty_mismatch is a hard, explicit signal from the
    # extractor and is checked independently of alignment_confidence -- a mismatched source must
    # never survive just because it scored a middling confidence.
    MISMATCH_THRESHOLD = 0.1
    # specialty_mismatch means "conflicts with a concrete fact in the patient's own
    # profile" -- with no profile on record there is nothing for an article to
    # conflict with, so the flag is meaningless here and must never be honored,
    # regardless of what the extractor returned (belt-and-suspenders alongside the
    # prompt instruction: an aux-model misfire on this signal silently zeroes out
    # every source, which then also disables the downstream claim-alignment gate).
    mismatched = [
        a for a in articles
        if (a.specialty_mismatch and has_patient_profile)
        or (a.alignment_confidence < MISMATCH_THRESHOLD and not a.answers_question)
    ]
    excluded_source_ids: List[str] = []
    if mismatched:
        mismatched_ids = {id(a) for a in mismatched}
        excluded_source_ids = [a.source_id for a in mismatched]
        articles = [a for a in articles if id(a) not in mismatched_ids]

    low_conf = [a for a in articles if a.alignment_confidence < 0.3]
    notes = []
    if mismatched:
        notes.append(
            f"{len(mismatched)} source(s) excluded -- confirmed to concern a different "
            "condition/measurement meaning than this patient's profile."
        )
    if low_conf:
        notes.append(
            f"{len(low_conf)} source(s) had low patient alignment confidence (<0.3) -- used for general context only."
        )
    extraction_notes = " ".join(notes)

    return ExtractedEvidenceDossier(
        question=question,
        patient_profile_summary=patient_summary,
        articles=articles,
        extraction_notes=extraction_notes,
        excluded_source_ids=excluded_source_ids,
    )
