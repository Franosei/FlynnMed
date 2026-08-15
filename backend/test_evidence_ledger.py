"""
Evidence Ledger Phase 1 tests: SourceArtifact/EvidencePassage persistence and
dedup. Follows test_clinician_access.py's exact conventions (skipif-gated on
a live Postgres with migrations applied, rollback-isolated db_session
fixture).
"""
from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from backend.db import get_session_factory
from backend.evidence_ledger import (
    persist_evidence_claim,
    persist_evidence_for_bundle,
    persist_evidence_passage,
    persist_source_artifact,
)


def _db_available() -> bool:
    if not os.getenv("DATABASE_URL"):
        return False
    try:
        with get_session_factory()() as session:
            session.execute(text("SELECT 1"))
        return True
    except OperationalError:
        return False


pytestmark = pytest.mark.skipif(
    not _db_available(),
    reason="requires a live Postgres (DATABASE_URL) with migrations applied",
)


@pytest.fixture()
def db_session():
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


def _unique_url() -> str:
    return f"https://test.example/evidence-ledger-{uuid.uuid4().hex[:12]}"


def test_persist_source_artifact_dedupes_unchanged_source(db_session):
    url = _unique_url()
    source = {"url": url, "title": "Example guidance", "detail_snippet": "Some excerpt text."}

    first = persist_source_artifact(db_session, source)
    second = persist_source_artifact(db_session, dict(source))

    assert first is not None
    assert second is not None
    assert first.id == second.id


def test_persist_source_artifact_creates_new_row_for_changed_content(db_session):
    url = _unique_url()
    original = persist_source_artifact(
        db_session, {"url": url, "title": "Guidance v1", "detail_snippet": "Original text."}
    )
    updated = persist_source_artifact(
        db_session, {"url": url, "title": "Guidance v1", "detail_snippet": "Updated text."}
    )

    assert original is not None
    assert updated is not None
    assert original.id != updated.id
    # Immutability: the original row's stored text is untouched by the update.
    assert original.stored_snapshot_text == "Original text."
    assert updated.stored_snapshot_text == "Updated text."


def test_persist_source_artifact_returns_none_without_url_or_text(db_session):
    assert persist_source_artifact(db_session, {"title": "No URL"}) is None
    assert persist_source_artifact(db_session, {"url": _unique_url()}) is None


def test_persist_evidence_passage_dedupes_same_quote(db_session):
    artifact = persist_source_artifact(
        db_session,
        {"url": _unique_url(), "title": "Guidance", "detail_snippet": "Full source text here."},
    )
    assert artifact is not None

    first = persist_evidence_passage(db_session, artifact, "Full source text", "passage 1 of 1")
    second = persist_evidence_passage(db_session, artifact, "Full source text", "passage 1 of 1")

    assert first is not None
    assert second is not None
    assert first.id == second.id


def test_persist_evidence_passage_creates_distinct_rows_for_distinct_quotes(db_session):
    artifact = persist_source_artifact(
        db_session,
        {"url": _unique_url(), "title": "Guidance", "detail_snippet": "Full source text here."},
    )
    assert artifact is not None

    first = persist_evidence_passage(db_session, artifact, "Full source", "passage 1 of 2")
    second = persist_evidence_passage(db_session, artifact, "text here", "passage 2 of 2")

    assert first is not None
    assert second is not None
    assert first.id != second.id


def test_persist_evidence_for_bundle_enriches_sources_in_place():
    """
    Integration test through the real short-lived-session write path (this
    one does commit -- matches this codebase's existing precedent of
    account/patient test fixtures creating real, randomized-URL rows rather
    than being rollback-isolated, since persist_evidence_for_bundle manages
    its own session by design and is not given an external one to roll back).
    """
    from backend.evidence_schema import ArticleEvidence, ExtractedEvidenceDossier

    url = _unique_url()
    combined_sources = [
        {
            "source_id": "S1",
            "url": url,
            "title": "Flucloxacillin prescribing guidance",
            "detail_snippet": "Flucloxacillin can potentiate warfarin's anticoagulant effect.",
        }
    ]
    dossier = ExtractedEvidenceDossier(
        question="Can I take flucloxacillin with warfarin?",
        patient_profile_summary="On warfarin.",
        articles=[
            ArticleEvidence(
                source_id="S1",
                title="Flucloxacillin prescribing guidance",
                extracted_passages=[
                    "Flucloxacillin can potentiate warfarin's anticoagulant effect."
                ],
            )
        ],
    )

    persist_evidence_for_bundle(combined_sources, dossier)

    enriched = combined_sources[0]
    assert enriched.get("source_version")
    assert enriched.get("retrieved_at")
    assert enriched.get("passage_id")
    assert (
        enriched.get("exact_passage")
        == "Flucloxacillin can potentiate warfarin's anticoagulant effect."
    )
    assert enriched.get("passage_locator") == "passage 1 of 1 extracted from this source"


def test_persist_evidence_for_bundle_never_raises_on_bad_input():
    """A source missing url/text (nothing to persist) must be skipped
    silently, not crash the whole answer pipeline."""
    combined_sources = [{"source_id": "S1", "title": "No URL or text"}]
    persist_evidence_for_bundle(combined_sources, None)
    assert "source_version" not in combined_sources[0]


def _fake_claim(**overrides):
    from backend.evidence_schema import StructuredClaim

    defaults = dict(
        claim_text="Drug X reduced relapse rate compared to placebo.",
        population="Adults with condition Y",
        intervention="Drug X",
        comparator="Placebo",
        outcome="Relapse rate at 12 months",
        study_design="rct",
        certainty="moderate",
        exact_quote="Drug X reduced relapse rate compared to placebo.",
    )
    defaults.update(overrides)
    return StructuredClaim(**defaults)


def test_persist_evidence_claim_dedupes_same_claim(db_session):
    artifact = persist_source_artifact(
        db_session,
        {"url": _unique_url(), "title": "Trial guidance", "detail_snippet": "Trial text here."},
    )
    assert artifact is not None
    claim = _fake_claim()

    first = persist_evidence_claim(db_session, artifact, None, claim)
    second = persist_evidence_claim(db_session, artifact, None, claim)

    assert first is not None
    assert second is not None
    assert first.id == second.id


def test_persist_evidence_claim_links_to_source_and_passage(db_session):
    artifact = persist_source_artifact(
        db_session,
        {"url": _unique_url(), "title": "Trial guidance", "detail_snippet": "Trial text here."},
    )
    assert artifact is not None
    passage = persist_evidence_passage(db_session, artifact, "Trial text", "passage 1 of 1")
    assert passage is not None

    claim = persist_evidence_claim(db_session, artifact, passage, _fake_claim())

    assert claim is not None
    assert claim.source_artifact_id == artifact.id
    assert claim.passage_id == passage.id


def test_persist_evidence_claim_coerces_unrecognised_study_design_and_certainty(db_session):
    artifact = persist_source_artifact(
        db_session,
        {"url": _unique_url(), "title": "Trial guidance", "detail_snippet": "Trial text here."},
    )
    assert artifact is not None
    claim = _fake_claim(study_design="not_a_real_design", certainty="extremely_high")

    persisted = persist_evidence_claim(db_session, artifact, None, claim)

    assert persisted is not None
    assert persisted.study_design == "unknown"
    assert persisted.certainty == "unknown"


def test_persist_evidence_claim_returns_none_without_claim_text(db_session):
    artifact = persist_source_artifact(
        db_session,
        {"url": _unique_url(), "title": "Trial guidance", "detail_snippet": "Trial text here."},
    )
    assert artifact is not None

    assert persist_evidence_claim(db_session, artifact, None, _fake_claim(claim_text="")) is None


def test_persist_evidence_for_bundle_persists_structured_claims():
    """Integration test: a dossier article with a populated structured_claims
    list should result in a linked EvidencePassage + EvidenceClaim, discovered
    the same way passage-only sources already are (see the passage-only
    integration test above)."""
    from backend.evidence_schema import ArticleEvidence, ExtractedEvidenceDossier

    url = _unique_url()
    combined_sources = [
        {
            "source_id": "S1",
            "url": url,
            "title": "Drug X trial guidance",
            "detail_snippet": "Drug X reduced relapse rate compared to placebo.",
        }
    ]
    dossier = ExtractedEvidenceDossier(
        question="How effective is Drug X compared to placebo?",
        patient_profile_summary="Adult with condition Y.",
        articles=[
            ArticleEvidence(
                source_id="S1",
                title="Drug X trial guidance",
                structured_claims=[_fake_claim()],
            )
        ],
    )

    persist_evidence_for_bundle(combined_sources, dossier)

    session_factory = get_session_factory()
    with session_factory() as db:
        artifact = db.execute(
            text("SELECT id FROM source_artifacts WHERE url = :url"), {"url": url}
        ).scalar_one()
        claim_row = db.execute(
            text("SELECT source_artifact_id, passage_id, claim_text FROM evidence_claims WHERE source_artifact_id = :aid"),
            {"aid": artifact},
        ).mappings().one()
        assert claim_row["source_artifact_id"] == artifact
        assert claim_row["passage_id"] is not None
        assert claim_row["claim_text"] == "Drug X reduced relapse rate compared to placebo."
