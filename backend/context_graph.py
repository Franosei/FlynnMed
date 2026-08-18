"""
ContextGraph: fast, no-LLM relevance graph over a user's prior health records.

On each query:
  1. Extract medical entities from the question (keyword matching, no LLM).
  2. Score every prior record (conditions, medications, symptoms, vitals, etc.)
     by entity overlap + recency.
  3. Return a ranked set of nodes, a focused prompt block for the LLM, and
     patient-specific search hints to sharpen PubMed retrieval.

No external API calls -- this runs in < 50 ms.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from backend.relationship_engine import (
    GENERIC_TRIAGE_CONCERN_LABEL,
    truncate_at_word_boundary,
)
from backend.utils import render_vital_for_prompt


_SKIP_WORDS: frozenset = frozenset({
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "this", "that", "these", "those", "there", "here", "then", "than",
    "have", "has", "had", "does", "did", "will", "would", "could", "should",
    "shall", "may", "might", "must", "been", "being", "was", "were",
    "the", "and", "but", "for", "not", "with", "from", "into", "about",
    "are", "can", "also", "any", "all", "some", "more", "such", "other",
    "our", "your", "their", "its", "his", "her", "him", "she", "they",
    "you", "him", "her", "his", "hers", "them", "we", "us", "my", "me",
    "give", "take", "get", "got", "put", "set", "let", "see", "know",
    "think", "want", "need", "going", "feel", "felt", "make", "made",
    "tell", "told", "said", "come", "came", "look", "use", "used",
    "ask", "asked", "find", "help", "show", "start", "stop", "now",
    "very", "just", "still", "even", "much", "most", "many", "too",
    "yes", "yet", "well", "else", "both", "each", "few", "same",
    "because", "since", "while", "after", "before", "until", "though",
    "although", "however", "therefore", "thus", "hence", "also", "so",
    "often", "sometimes", "always", "never", "usually", "normally",
    "please", "really", "something", "anything", "nothing", "everything",
    "cause", "causes", "caused", "causing", "effect", "effects", "affect",
    "affecting", "affected", "having", "getting", "feeling", "been",
    "like", "when", "over", "under", "around", "between", "through",
    "about", "against", "along", "among", "within", "without", "during",
    # common question/hedging words that are not medical search terms
    "worried", "worry", "concern", "concerned", "wonder", "wondering",
    "understand", "understanding", "given", "history", "problem",
    "issue", "question", "things", "think", "thinking", "maybe",
    "perhaps", "possible", "probably", "actually", "basically",
    "normal", "okay", "fine", "right", "wrong", "serious", "bad",
})


@dataclass
class ContextNode:
    node_id: str
    node_type: str   # condition|medication|symptom|vital|allergy|triage
    label: str
    detail: str
    relevance_score: float   # 0.0–1.0 keyword overlap
    recency_weight: float    # 1.0=very recent → 0.3=old
    source: str

    @property
    def weighted_score(self) -> float:
        # Relevance dominates; recency breaks ties and boosts very recent records
        return self.relevance_score * 0.75 + self.recency_weight * 0.25


@dataclass
class ContextEdge:
    source_type: str
    source_name: str
    relation: str
    target_type: str
    target_name: str
    relation_class: str = "association"
    certainty: str = "user_reported"
    evidence: str = ""
    relevance_score: float = 0.0

    def prompt_line(self) -> str:
        certainty = {
            "user_suspected": "user suspects",
            "user_reported": "user reports",
            "documented": "documented",
            "recorded_association": "recorded association",
        }.get(self.certainty, "recorded association")
        relation = self.relation.replace("_", " ")
        return (
            f"[{self.relation_class}; {certainty}] "
            f"{self.source_name} {relation} {self.target_name}"
            + (f" ({self.evidence})" if self.evidence else "")
        )


def format_relationships_for_user(
    edges: List[ContextEdge], max_edges: int = 3
) -> str:
    """Render graph edges as plain record context, never as internal prompt syntax."""
    if not edges or max_edges <= 0:
        return ""

    relation_templates = {
        "taken_for": lambda source, target: f"{source} is taken for {target}",
        "causes": lambda source, target: f"{source} may have caused {target}",
        "triggers": lambda source, target: f"{source} may trigger {target}",
        "worsens": lambda source, target: f"{source} may worsen {target}",
        "improves": lambda source, target: f"{source} may improve {target}",
        "started_after": lambda source, target: f"{source} started after {target}",
        "allergic_reaction": lambda source, target: (
            f"{target} was recorded as a reaction to {source}"
        ),
        "associated_with": lambda source, target: (
            f"{source} was associated with {target}"
        ),
        "led_to": lambda source, target: (
            f"{source} led to the recorded next step, {target}"
        ),
        "recorded_in": lambda source, target: f"{source} was recorded in {target}",
        "recommended_for": lambda source, target: (
            f"{source} was recommended for {target}"
        ),
    }
    certainty_openings = {
        "user_suspected": "You previously described this as a possible link: ",
        "user_reported": "You previously reported that ",
        "documented": "Your record notes that ",
        "recorded_association": "Your record contains a possible association: ",
    }

    sentences = []
    ranked_edges = sorted(
        edges, key=lambda edge: edge.relevance_score, reverse=True
    )[:max_edges]
    for edge in ranked_edges:
        source = truncate_at_word_boundary(edge.source_name, 160)
        target = truncate_at_word_boundary(edge.target_name, 160)
        if not source or not target:
            continue
        # GENERIC_TRIAGE_CONCERN_LABEL is an internal sentinel (see
        # relationship_engine.py), not real record content -- rendering it
        # verbatim reads as leaked system state ("Recorded triage concern
        # led to..."). Present it as what it is: an unspecified prior note.
        if source == GENERIC_TRIAGE_CONCERN_LABEL:
            source = "a previous triage note"
        if target == GENERIC_TRIAGE_CONCERN_LABEL:
            target = "a previous triage note"
        formatter = relation_templates.get(
            edge.relation,
            lambda source, target: f"{source} was recorded in relation to {target}",
        )
        opening = certainty_openings.get(
            edge.certainty, "Your record contains a possible association: "
        )
        sentences.append(opening + formatter(source, target) + ".")

    if not sentences:
        return ""
    return (
        "Relevant record context: "
        + " ".join(sentences)
        + " These entries are reports or recorded associations, not proof that one "
        "thing caused another."
    )


@dataclass
class ContextGraph:
    question: str = ""
    question_entities: List[str] = field(default_factory=list)
    nodes: List[ContextNode] = field(default_factory=list)
    edges: List[ContextEdge] = field(default_factory=list)
    search_hints: List[str] = field(default_factory=list)

    def top_nodes(self, max_nodes: int = 10) -> List[ContextNode]:
        return sorted(self.nodes, key=lambda n: n.weighted_score, reverse=True)[:max_nodes]

    def to_prompt_block(self, max_nodes: int = 10) -> str:
        nodes = self.top_nodes(max_nodes)
        if not nodes:
            return ""
        lines = [f"[{n.node_type.upper()}] {n.detail}" for n in nodes]
        return "Patient history most relevant to this question:\n" + "\n".join(lines)

    def has_conditions(self) -> bool:
        return any(n.node_type == "condition" for n in self.nodes)

    def top_condition_labels(self, n: int = 3) -> List[str]:
        return [
            node.label for node in self.nodes
            if node.node_type == "condition"
        ][:n]

    def top_medication_labels(self, n: int = 3) -> List[str]:
        return [
            node.label for node in self.nodes
            if node.node_type == "medication" and node.relevance_score > 0.2
        ][:n]

    def relationship_prompt_block(self, max_edges: int = 8) -> str:
        if not self.edges:
            return ""
        edges = sorted(
            self.edges, key=lambda edge: edge.relevance_score, reverse=True
        )[:max_edges]
        lines = "\n".join(f"- {edge.prompt_line()}" for edge in edges)
        return (
            "Explicit patient-reported clinical relationships:\n"
            f"{lines}\n"
            "Treat these links as patient-reported context, not proven medical causation. "
            "Distinguish temporal association or suspicion from evidence-supported causality."
        )


def build_context_graph(
    question: str,
    conditions: Optional[List[Dict]] = None,
    medications: Optional[List[Dict]] = None,
    symptom_logs: Optional[List[Dict]] = None,
    vitals: Optional[List[Dict]] = None,
    allergies: Optional[List[Dict]] = None,
    triage_summaries: Optional[List[Dict]] = None,
    relationships: Optional[List[Dict]] = None,
    longitudinal_memory: str = "",
) -> ContextGraph:
    """
    Build a relevance graph from a user's prior records vs. the current question.
    Fast: no LLM, no network calls. Runs in < 50 ms.
    """
    graph = ContextGraph(question=question)
    graph.question_entities = _extract_entities(question)
    q_words = set(graph.question_entities)
    now = datetime.now(timezone.utc)

    # ── Conditions ─────────────────────────────────────────────────────────────
    for cond in (conditions or []):
        name = str(cond.get("name") or "").strip()
        if not name:
            continue
        status = str(cond.get("status") or "").strip()
        notes = str(cond.get("notes") or "").strip()
        score = _score(q_words, f"{name} {status} {notes}")
        recency = _recency_weight(cond.get("recorded_on"), now)
        detail = name + (f" ({status})" if status and status not in ("unknown", "") else "")
        graph.nodes.append(ContextNode(
            node_id=f"cond:{name.lower()}",
            node_type="condition",
            label=name,
            detail=detail,
            relevance_score=max(score, 0.25),
            recency_weight=recency,
            source="condition history",
        ))

    # ── Medications ─────────────────────────────────────────────────────────────
    for med in (medications or []):
        name = str(med.get("name") or "").strip()
        if not name:
            continue
        dose = str(med.get("dose") or "").strip()
        reason = str(med.get("reason") or "").strip()
        score = _score(q_words, f"{name} {dose} {reason}")
        detail = name + (f" {dose}" if dose else "") + (f" for {reason}" if reason else "")
        graph.nodes.append(ContextNode(
            node_id=f"med:{name.lower()}",
            node_type="medication",
            label=name,
            detail=detail,
            relevance_score=max(score, 0.15),
            recency_weight=0.9,
            source="medication list",
        ))

    # ── Allergies ────────────────────────────────────────────────────────────────
    for allergy in (allergies or []):
        name = str(allergy.get("name") or "").strip()
        if not name:
            continue
        reaction = str(allergy.get("reaction") or "").strip()
        severity = str(allergy.get("severity") or "").strip()
        score = _score(q_words, f"{name} {reaction}")
        detail = (
            f"Allergy to {name}"
            + (f" -- {reaction}" if reaction else "")
            + (f" ({severity})" if severity else "")
        )
        graph.nodes.append(ContextNode(
            node_id=f"allergy:{name.lower()}",
            node_type="allergy",
            label=name,
            detail=detail,
            relevance_score=max(score, 0.1),
            recency_weight=0.8,
            source="allergy record",
        ))

    # ── Recent symptom logs (last 20) ─────────────────────────────────────────
    for log in (symptom_logs or [])[-20:]:
        symptom = str(log.get("symptom") or "").strip()
        if not symptom:
            continue
        severity = str(log.get("severity") or "").strip()
        notes = str(log.get("notes") or "").strip()
        score = _score(q_words, f"{symptom} {notes}")
        recency = _recency_weight(log.get("logged_on") or log.get("date"), now)
        if score > 0 or recency >= 0.75:
            detail = symptom + (f" (severity {severity})" if severity else "")
            graph.nodes.append(ContextNode(
                node_id=f"symptom:{symptom.lower()}:{log.get('logged_on', '')}",
                node_type="symptom",
                label=symptom,
                detail=detail,
                relevance_score=score,
                recency_weight=recency,
                source="symptom log",
            ))

    # ── Vitals (last 10) ─────────────────────────────────────────────────────
    for vital in (vitals or [])[-10:]:
        vtype = str(vital.get("type") or "").strip()
        vval = str(vital.get("value") or "").strip()
        if not vtype or not vval:
            continue
        score = _score(q_words, vtype)
        recency = _recency_weight(vital.get("recorded_on"), now)
        if score > 0 or recency >= 0.7:
            detail = render_vital_for_prompt(vital, include_date=False)
            graph.nodes.append(ContextNode(
                node_id=f"vital:{vtype.lower()}",
                node_type="vital",
                label=vtype,
                detail=detail,
                relevance_score=score,
                recency_weight=recency,
                source="vitals record",
            ))

    # ── Triage summaries (last 5) ─────────────────────────────────────────────
    for triage in (triage_summaries or [])[-5:]:
        impression = str(triage.get("impression") or "").strip()
        if not impression:
            continue
        question_text = str(triage.get("question") or "").strip()
        score = _score(q_words, f"{impression} {question_text}")
        recency = _recency_weight(
            triage.get("created_at") or triage.get("date"), now
        )
        if score > 0.15 or recency >= 0.85:
            graph.nodes.append(ContextNode(
                node_id=f"triage:{triage.get('trace_id', id(triage))}",
                node_type="triage",
                label="Previous assessment",
                detail=impression[:200],
                relevance_score=score,
                recency_weight=recency,
                source="prior consultation",
            ))

    # Explicit links captured from the patient's own statements. These are
    # edges, not diagnoses: their certainty remains visible to every response.
    for item in relationships or []:
        source_name = str(item.get("source_name") or "").strip()
        target_name = str(item.get("target_name") or "").strip()
        relation = str(item.get("relation") or "").strip().lower()
        if not source_name or not target_name or not relation:
            continue
        relevance = _score(
            q_words,
            f"{source_name} {relation.replace('_', ' ')} {target_name}",
        )
        graph.edges.append(
            ContextEdge(
                source_type=str(item.get("source_type") or "other").strip().lower(),
                source_name=source_name,
                relation=relation,
                target_type=str(item.get("target_type") or "other").strip().lower(),
                target_name=target_name,
                relation_class=str(
                    item.get("relation_class") or "association"
                ).strip().lower(),
                certainty=str(item.get("certainty") or "user_reported").strip().lower(),
                evidence=str(item.get("evidence") or "").strip(),
                relevance_score=max(relevance, 0.1),
            )
        )

    # ── Derive search hints from patient context × question entities ──────────
    graph.search_hints = _build_search_hints(graph, q_words)

    return graph


def _build_search_hints(graph: ContextGraph, q_words: set) -> List[str]:
    """
    Generate patient-specific PubMed search phrases by combining the patient's
    conditions/medications with the key terms in their question.
    """
    hints: List[str] = []
    seen: set = set()

    top_conditions = [n.label for n in graph.nodes if n.node_type == "condition"][:3]
    top_meds = [
        n.label for n in graph.nodes
        if n.node_type == "medication" and n.relevance_score > 0.25
    ][:2]

    content_words = [w for w in graph.question_entities if len(w) > 4][:4]

    for cond in top_conditions[:2]:
        for word in content_words[:3]:
            cond_l = cond.lower()
            if word not in cond_l and cond_l not in word:
                hint = f"{cond} {word}"
                if hint not in seen:
                    hints.append(hint)
                    seen.add(hint)

    for med in top_meds[:1]:
        for word in content_words[:2]:
            med_l = med.lower()
            if word not in med_l and med_l not in word:
                hint = f"{med} {word}"
                if hint not in seen:
                    hints.append(hint)
                    seen.add(hint)

    return hints[:4]


def _extract_entities(text: str) -> List[str]:
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return [w for w in words if w not in _SKIP_WORDS]


def _score(q_words: set, node_text: str) -> float:
    if not q_words or not node_text:
        return 0.0
    node_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", node_text.lower()))
    exact = len(q_words & node_words)
    partial = sum(
        1 for qw in q_words
        for nw in node_words
        if len(qw) > 4 and qw != nw and (qw in nw or nw in qw)
    )
    return min(1.0, (exact + 0.4 * partial) / max(1, len(q_words)))


def _recency_weight(date_str: Optional[str], now: datetime) -> float:
    if not date_str:
        return 0.5
    try:
        date_str = str(date_str).strip()
        date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        days = max(0, (now - date).days)
        if days < 7:
            return 1.0
        if days < 30:
            return 0.9
        if days < 90:
            return 0.75
        if days < 180:
            return 0.6
        if days < 365:
            return 0.5
        return 0.3
    except Exception:
        return 0.5
