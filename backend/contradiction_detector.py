"""
Evidence Ledger v2, #4: scoped, same-intervention cross-source contradiction
detection.

NOT a general reconciliation engine. No jurisdiction-aware comparison
(SourceArtifact.jurisdiction is mostly blank -- see backend/models/evidence.py,
best-effort by design) and no narrative synthesis beyond a single pairwise
judgment. What this module does: groups StructuredClaim entries
(backend/evidence_schema.py, already extracted per source by
backend/evidence_extractor.py) across DIFFERENT sources by normalized
intervention-string similarity, then makes one batched LLM call per group
asking specifically whether any of the claims in it directly disagree on the
same intervention/outcome. Most questions retrieve claims about only one
intervention (or none with a genuine PICO decomposition at all -- most
sources don't have one, see StructuredClaim's docstring), so this is
typically zero or one extra LLM call, not one per source pair.
"""
from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Any, Dict, List


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().casefold().split())


def _group_claims_by_intervention(evidence_dossier: Any) -> List[List[Dict]]:
    """Flattens every source's structured_claims, then groups claims whose
    intervention text is similar (ratio >= 0.6) across at least two distinct
    source_ids. A claim with no intervention/claim_text, or the only claim
    for its intervention, never forms a group -- there's nothing to compare
    it against."""
    flat: List[Dict] = []
    for article in getattr(evidence_dossier, "articles", []) or []:
        for claim in getattr(article, "structured_claims", []) or []:
            intervention = getattr(claim, "intervention", "") or ""
            claim_text = getattr(claim, "claim_text", "") or ""
            if not intervention.strip() or not claim_text.strip():
                continue
            flat.append(
                {
                    "source_id": article.source_id,
                    "claim_text": claim_text,
                    "intervention": intervention,
                    "outcome": getattr(claim, "outcome", "") or "",
                }
            )

    groups: List[List[Dict]] = []
    used = [False] * len(flat)
    for i, claim in enumerate(flat):
        if used[i]:
            continue
        group = [claim]
        used[i] = True
        norm_i = _normalize(claim["intervention"])
        for j in range(i + 1, len(flat)):
            if used[j]:
                continue
            if SequenceMatcher(None, norm_i, _normalize(flat[j]["intervention"])).ratio() >= 0.6:
                group.append(flat[j])
                used[j] = True
        if len({c["source_id"] for c in group}) >= 2:
            groups.append(group)
    return groups


def detect_contradictions(llm: Any, evidence_dossier: Any) -> List[Dict]:
    """Returns a list of
    {source_a_id, claim_a, source_b_id, claim_b, topic, description} dicts,
    one per genuine disagreement found. `llm` is anything exposing `.client`
    (an OpenAI client) and `.model` -- an LLMHelper instance in practice.
    Never raises: a parse/request failure for one group degrades to "no
    contradictions found for that group" rather than propagating, matching
    the rest of the Evidence Ledger's never-blocks-the-answer discipline."""
    groups = [g for g in _group_claims_by_intervention(evidence_dossier) if len(g) >= 2]
    if not groups:
        return []

    results: List[Dict] = []
    for group in groups:
        try:
            listing = "\n".join(
                f"[{c['source_id']}] intervention={c['intervention']!r} outcome={c['outcome']!r}: {c['claim_text']}"
                for c in group
            )
            response = llm.client.chat.completions.create(
                model=llm.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "These are clinical claims about the same intervention, from "
                            "different sources. Identify only claims that DIRECTLY disagree "
                            "(an opposite recommendation, or a contradictory finding on the "
                            "same outcome) -- not claims that are simply worded differently or "
                            "address different outcomes/populations; those are not "
                            "contradictions. Return a JSON object with one key: contradictions, "
                            "a list of {\"source_a_id\": str, \"claim_a\": str, "
                            "\"source_b_id\": str, \"claim_b\": str, \"topic\": str, "
                            "\"description\": \"one sentence explaining the disagreement\"}. "
                            "source_a_id/source_b_id must be one of the exact bracketed ids "
                            "shown before each claim. Return an empty list if nothing "
                            "genuinely disagrees -- most groups won't."
                        ),
                    },
                    {"role": "user", "content": listing},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            payload = json.loads((response.choices[0].message.content or "").strip())
            items = payload.get("contradictions", [])
            valid_ids = {c["source_id"] for c in group}
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                source_a_id = str(item.get("source_a_id", ""))
                source_b_id = str(item.get("source_b_id", ""))
                if source_a_id not in valid_ids or source_b_id not in valid_ids or source_a_id == source_b_id:
                    continue
                results.append(
                    {
                        "source_a_id": source_a_id,
                        "claim_a": str(item.get("claim_a", ""))[:2000],
                        "source_b_id": source_b_id,
                        "claim_b": str(item.get("claim_b", ""))[:2000],
                        "topic": str(item.get("topic", ""))[:500],
                        "description": str(item.get("description", ""))[:1000],
                    }
                )
        except Exception as exc:
            print(f"[ContradictionDetector] detection failed for one intervention group: {exc}")
            continue
    return results


def persist_contradictions_for_bundle(
    trace_id: str, contradictions: List[Dict], combined_sources: List[Dict]
) -> None:
    """Persists detect_contradictions' output into evidence_contradictions,
    resolving each result's source_id back to the SourceArtifact row
    persist_evidence_for_bundle already wrote for this bundle (see its
    source["source_artifact_id"] field). A pair whose source_id can't be
    resolved (e.g. persistence for that source failed earlier) is skipped
    rather than persisted with a dangling reference. Never raises."""
    if not contradictions or not trace_id:
        return

    artifact_id_by_source = {
        s.get("source_id"): s.get("source_artifact_id")
        for s in combined_sources
        if s.get("source_id") and s.get("source_artifact_id")
    }
    if not artifact_id_by_source:
        return

    from backend.db import get_session_factory
    from backend.models.evidence_contradiction import EvidenceContradiction

    session_factory = get_session_factory()
    with session_factory() as db:
        for item in contradictions:
            try:
                source_a_id = artifact_id_by_source.get(item.get("source_a_id"))
                source_b_id = artifact_id_by_source.get(item.get("source_b_id"))
                if not source_a_id or not source_b_id:
                    continue
                db.add(
                    EvidenceContradiction(
                        trace_id=trace_id[:64],
                        source_a_id=source_a_id,
                        source_b_id=source_b_id,
                        topic=item.get("topic", ""),
                        claim_a=item.get("claim_a", ""),
                        claim_b=item.get("claim_b", ""),
                        description=item.get("description", ""),
                    )
                )
            except Exception as exc:
                print(f"[ContradictionDetector] contradiction persist failed: {exc}")
        db.commit()
