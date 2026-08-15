"""
Evidence Ledger v2 (#4) tests for backend/contradiction_detector.py: grouping
StructuredClaim entries by intervention similarity across sources, and the
batched LLM disagreement check. No DB; mocks the LLM client the same way
backend/test_evidence_extractor.py does.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from backend.contradiction_detector import _group_claims_by_intervention, detect_contradictions
from backend.evidence_schema import ArticleEvidence, ExtractedEvidenceDossier, StructuredClaim


class _FakeCompletions:
    def __init__(self, payload: dict):
        self.calls = []
        self._payload = payload

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self._payload)))]
        )


class _FakeLLM:
    def __init__(self, payload: dict):
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions(payload)))
        self.model = "test-model"


def _article(source_id: str, claims: list) -> ArticleEvidence:
    return ArticleEvidence(
        source_id=source_id, title=f"Article {source_id}", structured_claims=claims
    )


def _claim(intervention: str, outcome: str, claim_text: str) -> StructuredClaim:
    return StructuredClaim(
        claim_text=claim_text, intervention=intervention, outcome=outcome, exact_quote=claim_text
    )


def test_group_claims_by_intervention_groups_similar_interventions_across_sources():
    dossier = ExtractedEvidenceDossier(
        question="q",
        patient_profile_summary="",
        articles=[
            _article("S1", [_claim("aspirin", "stroke risk", "Aspirin reduces stroke risk.")]),
            _article("S2", [_claim("aspirin", "stroke risk", "Aspirin does not reduce stroke risk.")]),
        ],
    )

    groups = _group_claims_by_intervention(dossier)

    assert len(groups) == 1
    assert {c["source_id"] for c in groups[0]} == {"S1", "S2"}


def test_group_claims_by_intervention_ignores_single_source_claims():
    dossier = ExtractedEvidenceDossier(
        question="q",
        patient_profile_summary="",
        articles=[_article("S1", [_claim("aspirin", "stroke risk", "Aspirin reduces stroke risk.")])],
    )

    assert _group_claims_by_intervention(dossier) == []


def test_group_claims_by_intervention_ignores_claims_with_no_intervention():
    dossier = ExtractedEvidenceDossier(
        question="q",
        patient_profile_summary="",
        articles=[
            _article("S1", [_claim("", "outcome", "A claim with no intervention.")]),
            _article("S2", [_claim("", "outcome", "Another claim with no intervention.")]),
        ],
    )

    assert _group_claims_by_intervention(dossier) == []


def test_detect_contradictions_returns_empty_list_when_no_groups():
    llm = _FakeLLM({"contradictions": []})
    dossier = ExtractedEvidenceDossier(
        question="q",
        patient_profile_summary="",
        articles=[_article("S1", [_claim("aspirin", "stroke risk", "Aspirin reduces stroke risk.")])],
    )

    assert detect_contradictions(llm, dossier) == []
    assert llm.client.chat.completions.calls == []


def test_detect_contradictions_parses_valid_pair():
    llm = _FakeLLM(
        {
            "contradictions": [
                {
                    "source_a_id": "S1",
                    "claim_a": "Aspirin reduces stroke risk.",
                    "source_b_id": "S2",
                    "claim_b": "Aspirin does not reduce stroke risk.",
                    "topic": "Aspirin and stroke risk",
                    "description": "Sources disagree on whether aspirin reduces stroke risk.",
                }
            ]
        }
    )
    dossier = ExtractedEvidenceDossier(
        question="q",
        patient_profile_summary="",
        articles=[
            _article("S1", [_claim("aspirin", "stroke risk", "Aspirin reduces stroke risk.")]),
            _article("S2", [_claim("aspirin", "stroke risk", "Aspirin does not reduce stroke risk.")]),
        ],
    )

    result = detect_contradictions(llm, dossier)

    assert len(result) == 1
    assert result[0]["source_a_id"] == "S1"
    assert result[0]["source_b_id"] == "S2"


def test_detect_contradictions_drops_pair_referencing_unknown_source_id():
    llm = _FakeLLM(
        {
            "contradictions": [
                {
                    "source_a_id": "S1",
                    "claim_a": "x",
                    "source_b_id": "S99",  # not in this group
                    "claim_b": "y",
                    "topic": "t",
                    "description": "d",
                }
            ]
        }
    )
    dossier = ExtractedEvidenceDossier(
        question="q",
        patient_profile_summary="",
        articles=[
            _article("S1", [_claim("aspirin", "stroke risk", "Aspirin reduces stroke risk.")]),
            _article("S2", [_claim("aspirin", "stroke risk", "Aspirin does not reduce stroke risk.")]),
        ],
    )

    assert detect_contradictions(llm, dossier) == []
