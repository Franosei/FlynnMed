"""
Evidence Ledger v2, #11: read-side assembly of the full five-object lineage
(AnswerClaim -> EvidenceClaim/PatientFact -> EvidencePassage -> SourceArtifact)
plus any EvidenceContradiction rows for one trace_id, for the new
GET /api/evidence/trace/{trace_id} endpoint. Every one of these tables has
been written on every Health Chat answer since Phase 1-4 (see
backend/evidence_ledger.py, backend/patient_fact_ledger.py,
backend/answer_claim_ledger.py) but had no read path until now.
"""
from __future__ import annotations

from typing import Any, Dict, List
from uuid import UUID

from sqlalchemy import select

from backend.db import get_session_factory
from backend.models.answer_claim import AnswerClaim
from backend.models.evidence import EvidenceClaim, EvidencePassage, SourceArtifact
from backend.models.evidence_contradiction import EvidenceContradiction
from backend.models.patient_fact import PatientFact


def _parse_uuid(value: Any) -> Any:
    try:
        return UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


def build_evidence_trace(trace_id: str) -> Dict:
    """Returns {"trace_id", "claims": [...], "contradictions": [...]}. An
    unknown/empty trace_id returns an empty result rather than raising --
    callers (the API route) are responsible for verifying the trace belongs
    to the requesting user before calling this."""
    session_factory = get_session_factory()
    with session_factory() as db:
        answer_claims = db.execute(
            select(AnswerClaim)
            .where(AnswerClaim.trace_id == trace_id)
            .order_by(AnswerClaim.created_at)
        ).scalars().all()

        evidence_claim_ids = {
            uid for c in answer_claims for i in c.evidence_claim_ids if (uid := _parse_uuid(i))
        }
        patient_fact_ids = {
            uid for c in answer_claims for i in c.patient_fact_ids if (uid := _parse_uuid(i))
        }

        evidence_claims_by_id: Dict = {}
        if evidence_claim_ids:
            rows = db.execute(
                select(EvidenceClaim).where(EvidenceClaim.id.in_(evidence_claim_ids))
            ).scalars().all()
            evidence_claims_by_id = {row.id: row for row in rows}

        passage_ids = {ec.passage_id for ec in evidence_claims_by_id.values() if ec.passage_id}
        passages_by_id: Dict = {}
        if passage_ids:
            rows = db.execute(
                select(EvidencePassage).where(EvidencePassage.id.in_(passage_ids))
            ).scalars().all()
            passages_by_id = {row.id: row for row in rows}

        contradiction_rows = db.execute(
            select(EvidenceContradiction).where(EvidenceContradiction.trace_id == trace_id)
        ).scalars().all()

        source_ids = {p.source_artifact_id for p in passages_by_id.values()}
        source_ids |= {row.source_a_id for row in contradiction_rows}
        source_ids |= {row.source_b_id for row in contradiction_rows}
        sources_by_id: Dict = {}
        if source_ids:
            rows = db.execute(
                select(SourceArtifact).where(SourceArtifact.id.in_(source_ids))
            ).scalars().all()
            sources_by_id = {row.id: row for row in rows}

        patient_facts_by_id: Dict = {}
        if patient_fact_ids:
            rows = db.execute(
                select(PatientFact).where(PatientFact.id.in_(patient_fact_ids))
            ).scalars().all()
            patient_facts_by_id = {row.id: row for row in rows}

        def source_dict(source: SourceArtifact) -> Dict:
            return {
                "title": source.title,
                "url": source.url,
                "source_version": source.content_hash[:12],
                "retrieved_at": source.retrieved_at.isoformat(),
                "is_full_document": source.is_full_document,
            }

        def evidence_claim_dict(ec: EvidenceClaim) -> Dict:
            passage = passages_by_id.get(ec.passage_id) if ec.passage_id else None
            source = sources_by_id.get(passage.source_artifact_id) if passage else None
            return {
                "claim_text": ec.claim_text,
                "study_design": ec.study_design,
                "certainty": ec.certainty,
                "risk_of_bias": ec.risk_of_bias,
                "passage": (
                    {
                        "exact_text": passage.exact_text,
                        "locator": passage.locator,
                        "source": source_dict(source) if source else None,
                    }
                    if passage
                    else None
                ),
            }

        def patient_fact_dict(fact: PatientFact) -> Dict:
            return {
                "label": fact.label,
                "value": fact.value,
                "status": fact.status,
                "source": fact.source,
                "previous_fact_id": str(fact.previous_fact_id) if fact.previous_fact_id else None,
            }

        claims_out: List[Dict] = []
        for claim in answer_claims:
            claims_out.append(
                {
                    "claim_text": claim.claim_text,
                    "status": claim.status,
                    "requires_evidence": claim.requires_evidence,
                    "module": claim.module,
                    "llm_only_support": claim.llm_only_support,
                    "evidence_claims": [
                        evidence_claim_dict(evidence_claims_by_id[uid])
                        for i in claim.evidence_claim_ids
                        if (uid := _parse_uuid(i)) in evidence_claims_by_id
                    ],
                    "patient_facts": [
                        patient_fact_dict(patient_facts_by_id[uid])
                        for i in claim.patient_fact_ids
                        if (uid := _parse_uuid(i)) in patient_facts_by_id
                    ],
                }
            )

        contradictions_out = [
            {
                "topic": row.topic,
                "claim_a": row.claim_a,
                "claim_b": row.claim_b,
                "description": row.description,
                "source_a": source_dict(sources_by_id[row.source_a_id]) if row.source_a_id in sources_by_id else None,
                "source_b": source_dict(sources_by_id[row.source_b_id]) if row.source_b_id in sources_by_id else None,
            }
            for row in contradiction_rows
        ]

        return {"trace_id": trace_id, "claims": claims_out, "contradictions": contradictions_out}
