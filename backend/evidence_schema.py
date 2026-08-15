"""
Evidence extraction schema.

Every retrieved article/source is processed through the evidence extractor
to produce an ArticleEvidence instance before being passed to the LLM.

This prevents:
- Hallucination (model inventing facts not in the source)
- Wrong-patient facts (population mismatches silently reaching the LLM)
- Information overload (only patient-relevant facts are forwarded)
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class PatientAlignmentFact(BaseModel):
    """One fact from the article that maps to a specific value in this patient's profile."""
    category: str  # condition | medication | vital | demographic | allergy | symptom
    patient_value: str  # e.g. "Type 2 diabetes", "Metformin 500mg", "HbA1c 72mmol/mol"
    article_statement: str  # what the article actually says about this
    relevance_type: str  # direct_evidence | contraindication | drug_interaction | population_match


class StructuredClaim(BaseModel):
    """
    Evidence Ledger Phase 2: a normalised clinical claim with an explicit
    PICO decomposition (Population, Intervention, Comparator, Outcome), study
    design, and certainty -- only populated when the article states a
    genuine comparative/interventional finding (see
    backend/evidence_extractor.py's prompt); most sources won't have one,
    and an empty list is the correct, common case, not a gap. exact_quote is
    validated the same way as ArticleEvidence.extracted_passages before
    being accepted (backend/evidence_extractor.py::_canonicalize_passage) --
    a claim whose quote can't be verified is dropped entirely.
    """
    claim_text: str
    population: str = ""
    intervention: str = ""
    comparator: str = ""
    outcome: str = ""
    study_design: str = "unknown"
    certainty: str = "unknown"
    exact_quote: str = ""


class ArticleEvidence(BaseModel):
    """
    Structured evidence extracted from one source, filtered and mapped to this patient.
    All LLM prompts receive this schema -- never raw unprocessed snippets.
    """
    source_id: str
    title: str
    journal: Optional[str] = None
    year: Optional[str] = None
    url: Optional[str] = None
    evidence_tier: int = Field(default=3, description="1=NHS/NICE, 2=systematic reviews, 3=primary research")
    tier_label: str = ""

    # Does the article directly address the patient's question?
    answers_question: bool = False

    # Specific facts that answer the current question
    question_facts: List[str] = Field(
        default_factory=list,
        description="Precise facts from this article that answer the patient's question"
    )

    # Patient profile alignment
    patient_aligned_facts: List[PatientAlignmentFact] = Field(
        default_factory=list,
        description="Facts relevant to this specific patient's conditions, meds, vitals, demographics"
    )

    # Evidence Ledger Phase 1: verbatim quotes from source_snippet, each
    # validated as an exact substring before being accepted (see
    # backend/evidence_extractor.py) -- these become EvidencePassage rows
    # (backend/evidence_ledger.py), so a citation traces to an exact passage
    # instead of just "this source was used". Not tied 1:1 to individual
    # facts above -- a flat, verified quote pool for this article.
    extracted_passages: List[str] = Field(
        default_factory=list,
        description="Verbatim substrings of source_snippet backing this article's extracted facts"
    )

    # Evidence Ledger Phase 2: normalised PICO claims, only populated for a
    # genuine comparative/interventional finding -- most articles won't have
    # one, and an empty list is the expected common case (see StructuredClaim).
    structured_claims: List[StructuredClaim] = Field(
        default_factory=list,
        description="Normalised PICO clinical claims with study design and certainty, only populated for genuine comparative/interventional findings"
    )

    # Safety
    contraindications: List[str] = Field(
        default_factory=list,
        description="Contraindications or warnings relevant to this patient specifically"
    )
    drug_interactions: List[str] = Field(
        default_factory=list,
        description="Drug interactions mentioned that involve this patient's medications"
    )

    # Synthesis for the LLM prompt
    patient_relevant_summary: str = Field(
        default="",
        description="2-3 sentence summary of what this article contributes for this patient"
    )

    # Quality signal
    alignment_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # True when this source discusses a different clinical meaning of an ambiguous
    # term than the one confirmed by the patient's own profile (e.g. respiratory
    # peak flow vs. urology peak flow) -- an explicit signal rather than something
    # inferred from alignment_confidence, so a source can never leak through as
    # "general context" just because it scored a middling confidence.
    specialty_mismatch: bool = False
    specialty_mismatch_reason: str = ""

    # Raw source passage used for extraction
    source_snippet: str = ""


class ExtractedEvidenceDossier(BaseModel):
    """
    The complete set of structured evidence for one question + patient.
    Serialised into the LLM system prompt as the evidence context block.
    """
    question: str
    patient_profile_summary: str
    articles: List[ArticleEvidence] = Field(default_factory=list)
    extraction_notes: str = ""
    # source_ids dropped from `articles` because they were confirmed to concern a
    # different specialty/meaning of an ambiguous term. The orchestrator uses this
    # to also strip them from combined_sources/evidence_quality_report so they
    # can't reappear as citations or "general context" permission elsewhere.
    excluded_source_ids: List[str] = Field(default_factory=list)

    def to_prompt_context(self) -> str:
        """Render the dossier as a structured LLM context block."""
        if not self.articles:
            return "No retrieved evidence was available for this question."

        lines = [
            f"PATIENT CONTEXT: {self.patient_profile_summary}",
            f"QUESTION: {self.question}",
            "",
            "EVIDENCE DOSSIER -- structured extraction matched to this patient.",
            "RULE: Only cite facts explicitly present below. Do not invent or extend.",
            "",
        ]
        for art in self.articles:
            meta = " | ".join(filter(None, [art.journal, str(art.year or ""), f"Tier {art.evidence_tier}: {art.tier_label}"]))
            # Label with the article's real source_id (matching combined_sources / the
            # clickable Sources panel), not a positional index -- the dossier is sorted
            # by alignment confidence, so a positional [S1..] label here would drift from
            # the actual S-number the UI resolves citations against.
            lines.append(f"[{art.source_id}] {art.title}")
            if meta:
                lines.append(f"     {meta}")
            if art.question_facts:
                lines.append(f"     ANSWER FACTS: {' | '.join(art.question_facts[:5])}")
            for fact in art.patient_aligned_facts[:3]:
                lines.append(
                    f"     PATIENT MATCH [{fact.category} -- {fact.patient_value}]: {fact.article_statement}"
                )
            if art.contraindications:
                lines.append(f"     CONTRAINDICATIONS: {'; '.join(art.contraindications[:3])}")
            if art.drug_interactions:
                lines.append(f"     DRUG INTERACTIONS: {'; '.join(art.drug_interactions[:3])}")
            if art.patient_relevant_summary:
                lines.append(f"     SUMMARY: {art.patient_relevant_summary}")
            if not art.answers_question:
                lines.append("     NOTE: This source provides background context, not direct answer evidence.")
            lines.append("")

        if self.extraction_notes:
            lines.append(f"EXTRACTION NOTE: {self.extraction_notes}")

        return "\n".join(lines)
