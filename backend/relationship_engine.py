"""Shared clinical relationship normalization and deterministic derivation.

Relationships here are graph context, not proof of medical causation. They
retain provenance and certainty so every consumer can distinguish a recorded
fact, a patient suspicion, a temporal association, and a clinical decision.
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, Iterable, List, Optional


ALLOWED_RELATIONS = {
    "taken_for",
    "causes",
    "triggers",
    "worsens",
    "improves",
    "started_after",
    "allergic_reaction",
    "associated_with",
    "led_to",
    "recorded_in",
    "recommended_for",
}

RELATION_CLASS = {
    "causes": "causal",
    "triggers": "causal",
    "worsens": "causal",
    "improves": "therapeutic_effect",
    "taken_for": "treatment",
    "allergic_reaction": "adverse_reaction",
    "started_after": "temporal",
    "associated_with": "association",
    "led_to": "clinical_decision",
    "recorded_in": "provenance",
    "recommended_for": "care_recommendation",
}

_EXPLICIT_LINK_RE = re.compile(
    r"(?P<relation>caused\s+by|due\s+to|triggered\s+by|worsened\s+by|"
    r"improved\s+by|after\s+starting|associated\s+with)\s+(?P<target>[^.;,]+)",
    re.IGNORECASE,
)


def _stable_id(parts: Iterable[str]) -> str:
    raw = "|".join(str(part).strip().lower() for part in parts)
    return "rel-derived-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def normalize_relationship(
    item: Dict,
    *,
    default_source: str = "structured_record",
    default_certainty: str = "recorded_association",
) -> Optional[Dict]:
    source_name = str(item.get("source_name") or "").strip()
    target_name = str(item.get("target_name") or "").strip()
    relation = str(item.get("relation") or "").strip().lower()
    if not source_name or not target_name or relation not in ALLOWED_RELATIONS:
        return None
    certainty = str(item.get("certainty") or default_certainty).strip().lower()
    if certainty not in {
        "documented",
        "user_reported",
        "user_suspected",
        "recorded_association",
    }:
        certainty = default_certainty
    source_type = str(item.get("source_type") or "other").strip().lower()
    target_type = str(item.get("target_type") or "other").strip().lower()
    relationship_id = item.get("relationship_id") or _stable_id(
        (source_type, source_name, relation, target_type, target_name, default_source)
    )
    return {
        "relationship_id": relationship_id,
        "source_type": source_type,
        "source_name": source_name,
        "relation": relation,
        "relation_class": RELATION_CLASS.get(relation, "association"),
        "target_type": target_type,
        "target_name": target_name,
        "certainty": certainty,
        "evidence": str(item.get("evidence") or "").strip(),
        "source": str(item.get("source") or default_source).strip(),
        "recorded_at": str(item.get("recorded_at") or "").strip(),
    }


def _split_values(value: object) -> List[str]:
    return [
        item.strip()
        for item in re.split(r"[,;/]|\band\b", str(value or ""), flags=re.IGNORECASE)
        if item.strip()
    ]


def _item_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("title", "task", "action", "goal", "description", "name"):
            text = str(value.get(key) or "").strip()
            if text:
                return text
    return ""


def _note_relationships(
    record_type: str,
    record_name: str,
    notes: str,
    source: str,
    certainty: str,
) -> List[Dict]:
    results = []
    relation_map = {
        "caused by": "causes",
        "due to": "causes",
        "triggered by": "triggers",
        "worsened by": "worsens",
        "improved by": "improves",
        "after starting": "started_after",
        "associated with": "associated_with",
    }
    for match in _EXPLICIT_LINK_RE.finditer(notes or ""):
        phrase = match.group("relation").lower()
        target = match.group("target").strip()
        relation = relation_map[phrase]
        # "Condition due to exposure" means exposure causes condition, while
        # association/temporal links retain the record as their source.
        if relation in {"causes", "triggers", "worsens", "improves"}:
            source_name, target_name = target, record_name
            source_type, target_type = "factor", record_type
        else:
            source_name, target_name = record_name, target
            source_type, target_type = record_type, "factor"
        results.append(
            {
                "source_type": source_type,
                "source_name": source_name,
                "relation": relation,
                "target_type": target_type,
                "target_name": target_name,
                "certainty": certainty,
                "evidence": match.group(0),
                "source": source,
            }
        )
    return results


def derive_relationships(
    *,
    medications: Optional[List[Dict]] = None,
    allergies: Optional[List[Dict]] = None,
    conditions: Optional[List[Dict]] = None,
    symptom_logs: Optional[List[Dict]] = None,
    vitals: Optional[List[Dict]] = None,
    triage_summaries: Optional[List[Dict]] = None,
    care_plans: Optional[List[Dict]] = None,
    clinical_notes: Optional[List[Dict]] = None,
    safety_reviews: Optional[List[Dict]] = None,
    source: str = "structured_record",
) -> List[Dict]:
    candidates: List[Dict] = []
    if source in {"conversation", "manual_record"}:
        record_certainty = "user_reported"
    elif source.startswith("document:"):
        record_certainty = "documented"
    else:
        record_certainty = "recorded_association"

    for medication in medications or []:
        name = str(medication.get("name") or "").strip()
        reason = str(medication.get("reason") or "").strip()
        if name and reason:
            candidates.append(
                {
                    "source_type": "medication",
                    "source_name": name,
                    "relation": "taken_for",
                    "target_type": "condition",
                    "target_name": reason,
                    "certainty": record_certainty,
                    "evidence": f"Medication reason: {reason}",
                    "source": source,
                }
            )
        candidates.extend(
            _note_relationships(
                "medication",
                name,
                medication.get("notes", ""),
                source,
                record_certainty,
            )
            if name
            else []
        )

    for allergy in allergies or []:
        name = str(allergy.get("name") or "").strip()
        reaction = str(allergy.get("reaction") or "").strip()
        if name and reaction:
            candidates.append(
                {
                    "source_type": "allergy",
                    "source_name": name,
                    "relation": "allergic_reaction",
                    "target_type": "symptom",
                    "target_name": reaction,
                    "certainty": record_certainty,
                    "evidence": f"Recorded reaction: {reaction}",
                    "source": source,
                }
            )

    for symptom in symptom_logs or []:
        name = str(symptom.get("symptom") or "").strip()
        for trigger in _split_values(symptom.get("triggers")):
            if name:
                candidates.append(
                    {
                        "source_type": "factor",
                        "source_name": trigger,
                        "relation": "triggers",
                        "target_type": "symptom",
                        "target_name": name,
                        "certainty": record_certainty,
                        "evidence": f"Recorded trigger: {trigger}",
                        "source": source,
                    }
                )
        candidates.extend(
            _note_relationships(
                "symptom",
                name,
                symptom.get("notes", ""),
                source,
                record_certainty,
            )
            if name
            else []
        )

    for condition in conditions or []:
        name = str(condition.get("name") or "").strip()
        if name:
            candidates.extend(
                _note_relationships(
                    "condition",
                    name,
                    condition.get("notes", ""),
                    source,
                    record_certainty,
                )
            )

    for vital in vitals or []:
        name = str(vital.get("type") or "").strip()
        if name:
            candidates.extend(
                _note_relationships(
                    "vital",
                    name,
                    vital.get("notes", ""),
                    source,
                    record_certainty,
                )
            )

    for triage in triage_summaries or []:
        concern = str(
            triage.get("impression")
            or triage.get("decision_summary")
            or triage.get("question")
            or ""
        ).strip()
        next_step = str(triage.get("next_step") or "").strip()
        if concern and next_step:
            candidates.append(
                {
                    "source_type": "triage",
                    "source_name": concern[:180],
                    "relation": "led_to",
                    "target_type": "care_action",
                    "target_name": next_step,
                    "certainty": "documented",
                    "evidence": "Recorded triage decision",
                    "source": source,
                }
            )

    for plan in care_plans or []:
        condition = str(plan.get("condition") or plan.get("title") or "").strip()
        actions: List[str] = []
        for field in ("goals", "daily_tasks", "weekly_tasks", "medication_reminders"):
            for value in plan.get(field) or []:
                text = _item_text(value)
                if text and text not in actions:
                    actions.append(text)
        if condition:
            for action in actions[:12]:
                candidates.append(
                    {
                        "source_type": "care_action",
                        "source_name": action[:240],
                        "relation": "recommended_for",
                        "target_type": "condition",
                        "target_name": condition,
                        "certainty": "documented",
                        "evidence": "Saved care-plan action",
                        "source": source,
                    }
                )

    for note in clinical_notes or []:
        assessment = str(note.get("assessment") or "").strip()
        plan = str(note.get("plan") or "").strip()
        if assessment and plan:
            candidates.append(
                {
                    "source_type": "clinical_assessment",
                    "source_name": assessment[:240],
                    "relation": "led_to",
                    "target_type": "care_action",
                    "target_name": plan[:240],
                    "certainty": "documented",
                    "evidence": "Saved clinical note assessment and plan",
                    "source": source,
                }
            )
        for section in ("subjective", "objective", "assessment", "plan"):
            text = str(note.get(section) or "").strip()
            if text:
                candidates.extend(
                    _note_relationships(
                        "clinical_note",
                        f"{section.title()} section",
                        text,
                        source,
                        "documented",
                    )
                )

    for review in safety_reviews or []:
        action = str(review.get("proposed_action") or "").strip()
        if not action:
            continue
        for fact in review.get("patient_facts") or []:
            fact_name = str(fact.get("value") or fact.get("label") or "").strip()
            if not fact_name:
                continue
            candidates.append(
                {
                    "source_type": str(fact.get("record_type") or "patient_fact").lower(),
                    "source_name": fact_name[:240],
                    "relation": "led_to",
                    "target_type": "safety_action",
                    "target_name": action[:240],
                    "certainty": "documented",
                    "evidence": str(review.get("category") or "Safety review"),
                    "source": source,
                }
            )

    normalized: List[Dict] = []
    seen = set()
    for candidate in candidates:
        item = normalize_relationship(candidate, default_source=source)
        if not item:
            continue
        key = (
            item["source_type"], item["source_name"].lower(), item["relation"],
            item["target_type"], item["target_name"].lower(),
        )
        if key not in seen:
            normalized.append(item)
            seen.add(key)
    return normalized


def merge_relationships(*groups: Iterable[Dict]) -> List[Dict]:
    merged: List[Dict] = []
    seen = set()
    for group in groups:
        for raw in group or []:
            item = normalize_relationship(raw)
            if not item:
                continue
            key = (
                item["source_type"], item["source_name"].lower(), item["relation"],
                item["target_type"], item["target_name"].lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def relationship_summary(relationships: Iterable[Dict], max_items: int = 12) -> str:
    lines = []
    for item in merge_relationships(relationships):
        certainty = str(item.get("certainty") or "recorded_association").replace("_", " ")
        relation = str(item.get("relation") or "associated_with").replace("_", " ")
        lines.append(
            f"- [{certainty}] {item['source_name']} {relation} {item['target_name']}"
        )
        if len(lines) >= max_items:
            break
    if not lines:
        return ""
    return "Recorded clinical relationships (association does not prove causation):\n" + "\n".join(lines)
