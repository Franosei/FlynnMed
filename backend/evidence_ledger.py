"""
Evidence Ledger Phase 1 write path: persists retrieved sources and the exact
passages extracted facts were grounded in, so a citation can be traced to a
specific version of a specific source, checked at a specific time -- not just
"this source was used somewhere".

Deliberately isolated from the retrieval/orchestrator call chain (no `db:
Session` threaded through ClinicalOrchestrator/AgenticRetrievalLoop) --
persistence happens once, after a bundle's sources are finalized, via a
short-lived session opened here. This matches the per-call-session convention
already used elsewhere in this codebase (see backend/api.py's
_save_previsit_chat_message) rather than plumbing a session through many
layers of the retrieval hot path.

A persistence failure must never block answer generation -- callers wrap
persist_evidence_for_bundle in a broad try/except and continue without the
enriched fields on failure (see backend/rag_system.py's _prepare_answer_bundle).
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db import get_session_factory
from backend.models.evidence import EvidencePassage, SourceArtifact


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stored_snapshot_text(source: Dict[str, Any]) -> str:
    """The fullest available text for a source -- prefers the longer detail
    excerpt over the short display snippet, since the hash/passages should be
    computed from as much of the actual fetched content as we have."""
    return str(
        source.get("detail_snippet")
        or source.get("snippet")
        or source.get("text")
        or ""
    ).strip()


def persist_source_artifact(db: Session, source: Dict[str, Any]) -> Optional[SourceArtifact]:
    """Dedupes on (url, content_hash): refetching an unchanged source reuses
    its existing row; a changed source (e.g. a guideline update) gets a new
    row rather than overwriting the old one, so a past citation keeps
    pointing at the exact version it was actually checked against."""
    url = str(source.get("url") or "").strip()
    snapshot_text = _stored_snapshot_text(source)
    if not url or not snapshot_text:
        return None

    content_hash = _hash_text(snapshot_text)
    existing = db.execute(
        select(SourceArtifact).where(
            SourceArtifact.url == url, SourceArtifact.content_hash == content_hash
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    artifact = SourceArtifact(
        url=url,
        title=str(source.get("title") or "")[:1024],
        publisher=str(source.get("provider") or source.get("journal") or "")[:256],
        jurisdiction="",  # best-effort; not derivable from most retrieved sources today
        source_type=str(source.get("source_type") or "")[:32],
        published_date=str(source.get("year") or "").strip()[:32] or None,
        retrieved_at=datetime.now(timezone.utc),
        content_hash=content_hash,
        stored_snapshot_text=snapshot_text,
    )
    db.add(artifact)
    db.flush()  # populate artifact.id for the passage rows below, no commit yet
    return artifact


def persist_evidence_passage(
    db: Session, source_artifact: SourceArtifact, exact_text: str, locator: str
) -> Optional[EvidencePassage]:
    """Dedupes on (source_artifact_id, passage_hash): the same exact quote
    extracted twice from the same source version reuses one row."""
    cleaned = exact_text.strip()
    if not cleaned:
        return None

    passage_hash = _hash_text(cleaned)
    existing = db.execute(
        select(EvidencePassage).where(
            EvidencePassage.source_artifact_id == source_artifact.id,
            EvidencePassage.passage_hash == passage_hash,
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    passage = EvidencePassage(
        source_artifact_id=source_artifact.id,
        exact_text=cleaned,
        locator=locator[:256],
        passage_hash=passage_hash,
    )
    db.add(passage)
    db.flush()
    return passage


def _dossier_quotes_by_source_id(evidence_dossier: Any) -> Dict[str, List[str]]:
    """Pulls validated extracted_passages per source_id out of the
    orchestrator's ArticleEvidence dossier (see backend/evidence_extractor.py,
    which verifies each is a verbatim substring of the source before it's
    ever set). Returns {} for a dossier from before this field existed --
    callers fall back to no passage-level detail for that source rather than
    failing."""
    quotes_by_source: Dict[str, List[str]] = {}
    if evidence_dossier is None:
        return quotes_by_source
    for article in getattr(evidence_dossier, "articles", []) or []:
        quotes = [
            q.strip()
            for q in (getattr(article, "extracted_passages", None) or [])
            if isinstance(q, str) and q.strip()
        ]
        if quotes:
            quotes_by_source[article.source_id] = quotes
    return quotes_by_source


def persist_evidence_for_bundle(
    combined_sources: List[Dict[str, Any]], evidence_dossier: Any = None
) -> None:
    """Persists a SourceArtifact per retrieved source and an EvidencePassage
    per validated exact_quote, then mutates combined_sources IN PLACE with
    the new fields the API/frontend need: source_version, retrieved_at, and
    (when an extracted quote exists for that source) exact_passage/
    passage_locator/passage_id. Silent no-op per-source on any failure --
    never raises, since a persistence failure must not block the answer
    already being generated from these same sources.
    """
    if not combined_sources:
        return

    quotes_by_source = _dossier_quotes_by_source_id(evidence_dossier)
    session_factory = get_session_factory()
    with session_factory() as db:
        for source in combined_sources:
            source_id = source.get("source_id")
            try:
                artifact = persist_source_artifact(db, source)
                if artifact is None:
                    continue

                source["source_version"] = artifact.content_hash[:12]
                source["retrieved_at"] = artifact.retrieved_at.isoformat()

                quotes = quotes_by_source.get(source_id, [])
                if quotes:
                    primary_quote = quotes[0]
                    locator = f"passage 1 of {len(quotes)} extracted from this source"
                    passage = persist_evidence_passage(db, artifact, primary_quote, locator)
                    if passage is not None:
                        source["passage_id"] = str(passage.id)
                        source["exact_passage"] = passage.exact_text
                        source["passage_locator"] = passage.locator
                    for extra_quote in quotes[1:]:
                        persist_evidence_passage(
                            db,
                            artifact,
                            extra_quote,
                            "additional passage extracted from this source",
                        )
            except Exception as exc:  # persistence must never block the answer
                print(f"[EvidenceLedger] persistence failed for {source_id}: {exc}")
                continue
        db.commit()
