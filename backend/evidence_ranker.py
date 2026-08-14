"""
Evidence quality, tiering, and role-aware ranking.

The ranker does more than sort sources. It validates whether retrieved
literature is actually usable for the current question and the stored patient
profile before the answer is generated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from backend.intent_risk_classifier import IntentClassification
from backend.response_templates import build_tier_badge, get_tier_description
from backend.role_router import RoleConfig

if TYPE_CHECKING:
    from backend.context_graph import ContextGraph
    from backend.memory_store import MemoryStore
    from backend.patient_history import PatientHistoryContext


# Tier 1: formal guidance providers.
_TIER1_PROVIDERS = {
    "nhs",
    "nice",
    "mhra",
    "sign",
    "bnf",
    "nice cks",
    "rcog",
    "phe",
    "public health england",
    "public health wales",
    "uk health security agency",
    "ukhsa",
    "gov.uk",
}

# Tier 2 signals in article titles / journals.
_TIER2_TITLE_PATTERNS = [
    re.compile(r"\b(systematic review|meta.?analysis|cochrane|scoping review)\b", re.IGNORECASE),
]
_TIER2_JOURNALS = {
    "lancet",
    "bmj",
    "new england journal of medicine",
    "nejm",
    "jama",
    "annals of internal medicine",
    "plos medicine",
    "british medical journal",
    "nature medicine",
}

_CONTENT_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "been",
    "being",
    "because",
    "before",
    "could",
    "does",
    "doing",
    "during",
    "from",
    "give",
    "have",
    "having",
    "help",
    "here",
    "into",
    "just",
    "like",
    "make",
    "might",
    "more",
    "need",
    "only",
    "patient",
    "patients",
    "people",
    "person",
    "question",
    "really",
    "should",
    "some",
    "still",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "using",
    "want",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "your",
}

_PEDIATRIC_RE = re.compile(r"\b(child|children|paediatric|pediatric|infant|neonate|adolescent|teenager)\b", re.IGNORECASE)
_ADULT_RE = re.compile(r"\b(adult|adults)\b", re.IGNORECASE)
_OLDER_ADULT_RE = re.compile(r"\b(older adult|older adults|elderly|geriatric|aged)\b", re.IGNORECASE)
_PREGNANCY_RE = re.compile(r"\b(pregnancy|pregnant|maternity|antenatal|postnatal|obstetric)\b", re.IGNORECASE)
_MALE_ONLY_RE = re.compile(r"\b(prostate|testicular|male-only|men only|in men|male patients)\b", re.IGNORECASE)


@dataclass
class EvidenceQualityAssessment:
    status: str = "question_aligned"
    quality_score: float = 0.0
    question_alignment_score: float = 0.0
    patient_alignment_score: float = 0.0
    patient_alignment_facts: List[str] = field(default_factory=list)
    mismatch_flags: List[str] = field(default_factory=list)
    quality_reasons: List[str] = field(default_factory=list)
    source_validation_status: str = "valid"
    currency_status: str = "unknown"
    usable_for_patient_specific_guidance: bool = False

    def as_dict(self) -> Dict:
        return {
            "status": self.status,
            "quality_score": round(self.quality_score, 3),
            "question_alignment_score": round(self.question_alignment_score, 3),
            "patient_alignment_score": round(self.patient_alignment_score, 3),
            "patient_alignment_facts": self.patient_alignment_facts,
            "mismatch_flags": self.mismatch_flags,
            "quality_reasons": self.quality_reasons,
            "source_validation_status": self.source_validation_status,
            "currency_status": self.currency_status,
            "usable_for_patient_specific_guidance": self.usable_for_patient_specific_guidance,
        }


@dataclass
class TieredSource:
    """A source dict enriched with evidence tier, ranking, and quality metadata."""

    source_id: str = ""
    title: str = ""
    journal: str = ""
    year: str = ""
    authors: str = ""
    section: str = ""
    url: str = ""
    snippet: str = ""
    detail_snippet: str = ""
    source_type: str = ""
    provider: str = ""
    pmcid: str = ""
    query: str = ""
    similarity: float = 0.0
    relevance: float = 0.0

    evidence_tier: int = 3
    tier_label: str = ""
    tier_description: str = ""
    tier_badge: str = ""
    role_boost: float = 0.0
    combined_score: float = 0.0

    evidence_quality_status: str = "question_aligned"
    evidence_quality_score: float = 0.0
    question_alignment_score: float = 0.0
    patient_alignment_score: float = 0.0
    patient_alignment_facts: List[str] = field(default_factory=list)
    evidence_quality_reasons: List[str] = field(default_factory=list)
    patient_mismatch_flags: List[str] = field(default_factory=list)
    source_validation_status: str = "valid"
    currency_status: str = "unknown"
    usable_for_patient_specific_guidance: bool = False

    @classmethod
    def from_dict(cls, source: Dict) -> "TieredSource":
        return cls(
            source_id=source.get("source_id", ""),
            title=source.get("title", ""),
            journal=source.get("journal", ""),
            year=source.get("year", ""),
            authors=source.get("authors", ""),
            section=source.get("section", ""),
            url=source.get("url", ""),
            snippet=source.get("snippet", ""),
            detail_snippet=source.get("detail_snippet", source.get("snippet", "")),
            source_type=source.get("source_type", ""),
            provider=source.get("provider", ""),
            pmcid=source.get("pmcid", ""),
            query=source.get("query", ""),
            similarity=float(source.get("similarity", source.get("relevance", 0.0))),
            relevance=float(source.get("relevance", source.get("similarity", 0.0))),
        )

    def apply_quality(self, assessment: EvidenceQualityAssessment) -> None:
        self.evidence_quality_status = assessment.status
        self.evidence_quality_score = assessment.quality_score
        self.question_alignment_score = assessment.question_alignment_score
        self.patient_alignment_score = assessment.patient_alignment_score
        self.patient_alignment_facts = list(assessment.patient_alignment_facts)
        self.evidence_quality_reasons = list(assessment.quality_reasons)
        self.patient_mismatch_flags = list(assessment.mismatch_flags)
        self.source_validation_status = assessment.source_validation_status
        self.currency_status = assessment.currency_status
        self.usable_for_patient_specific_guidance = assessment.usable_for_patient_specific_guidance

    def as_dict(self) -> Dict:
        """Returns a dict fully compatible with the existing UI source rendering."""
        return {
            "source_id": self.source_id,
            "title": self.title,
            "journal": self.journal,
            "year": self.year,
            "authors": self.authors,
            "section": self.section,
            "url": self.url,
            "snippet": self.snippet,
            "detail_snippet": self.detail_snippet,
            "source_type": self.source_type,
            "provider": self.provider,
            "pmcid": self.pmcid,
            "query": self.query,
            "similarity": self.similarity,
            "relevance": self.combined_score or self.relevance,
            "evidence_tier": self.evidence_tier,
            "tier_label": self.tier_label,
            "tier_description": self.tier_description,
            "tier_badge": self.tier_badge,
            "role_boost": self.role_boost,
            "evidence_quality_status": self.evidence_quality_status,
            "evidence_quality_score": round(self.evidence_quality_score, 3),
            "question_alignment_score": round(self.question_alignment_score, 3),
            "patient_alignment_score": round(self.patient_alignment_score, 3),
            "patient_alignment_facts": self.patient_alignment_facts,
            "evidence_quality_reasons": self.evidence_quality_reasons,
            "patient_mismatch_flags": self.patient_mismatch_flags,
            "source_validation_status": self.source_validation_status,
            "currency_status": self.currency_status,
            "usable_for_patient_specific_guidance": self.usable_for_patient_specific_guidance,
        }


class EvidenceRanker:
    """
    Assigns evidence tiers and re-ranks sources using semantic similarity,
    source authority, role preferences, and patient-profile fit.
    """

    QUALITY_CRITERIA = [
        "source provenance",
        "current-question relevance",
        "stored patient-profile alignment",
        "population/profile mismatch",
        "publication currency",
    ]

    def rank_and_tier(
        self,
        sources: List[Dict],
        question: str,
        role_config: RoleConfig,
        intent: IntentClassification,
        memory_store: "MemoryStore",
        top_k: int = 6,
        patient_history: Optional["PatientHistoryContext"] = None,
        context_graph: Optional["ContextGraph"] = None,
    ) -> List[Dict]:
        ranked, _ = self.rank_and_tier_with_report(
            sources=sources,
            question=question,
            role_config=role_config,
            intent=intent,
            memory_store=memory_store,
            top_k=top_k,
            patient_history=patient_history,
            context_graph=context_graph,
        )
        return ranked

    def rank_and_tier_with_report(
        self,
        sources: List[Dict],
        question: str,
        role_config: RoleConfig,
        intent: IntentClassification,
        memory_store: "MemoryStore",
        top_k: int = 6,
        patient_history: Optional["PatientHistoryContext"] = None,
        context_graph: Optional["ContextGraph"] = None,
    ) -> Tuple[List[Dict], Dict]:
        """
        Returns ranked source dicts plus an audit report for the evidence-quality gate.
        Sources that fail provenance, relevance, or population-fit checks are excluded
        before answer generation.
        """
        if not sources:
            return [], self._build_quality_report([], [], patient_history, context_graph)

        try:
            query_vector = memory_store._embed_text(question)
            source_texts = [self._source_text(source) for source in sources]
            source_vectors = memory_store._embed_texts(source_texts)
            semantic_scores = [float(np.dot(query_vector, sv)) for sv in source_vectors]
        except Exception as exc:
            print(f"EvidenceRanker embedding failed, using fallback scores: {exc}")
            semantic_scores = [
                float(source.get("relevance", source.get("similarity", 0.5)))
                for source in sources
            ]

        tiered: List[TieredSource] = []
        excluded: List[Dict] = []
        for source, semantic_score in zip(sources, semantic_scores):
            ts = TieredSource.from_dict(source)
            ts.evidence_tier = self._assign_tier(source)
            ts.tier_label = f"Tier {ts.evidence_tier}"
            ts.tier_description = get_tier_description(ts.evidence_tier)
            ts.tier_badge = build_tier_badge(ts.evidence_tier)
            ts.role_boost = self._compute_role_boost(ts.evidence_tier, role_config)
            ts.relevance = round(self._clamp_score(semantic_score), 3)

            assessment = self._assess_evidence_quality(
                source=source,
                question=question,
                patient_history=patient_history,
                context_graph=context_graph,
                semantic_score=semantic_score,
                evidence_tier=ts.evidence_tier,
            )

            if assessment.status == "excluded":
                excluded.append(
                    {
                        "title": source.get("title", "Retrieved source"),
                        "source_type": source.get("source_type", ""),
                        "provider": source.get("provider", ""),
                        "query": source.get("query", ""),
                        "quality_score": round(assessment.quality_score, 3),
                        "reasons": assessment.quality_reasons,
                        "mismatch_flags": assessment.mismatch_flags,
                    }
                )
                continue

            ts.apply_quality(assessment)

            authority_weight = {1: 1.0, 2: 0.85, 3: 0.70}.get(ts.evidence_tier, 0.70)
            semantic_component = self._clamp_score(semantic_score) * authority_weight
            role_component = ts.role_boost * 0.15
            quality_component = assessment.quality_score * 0.35
            profile_component = 0.12 if assessment.usable_for_patient_specific_guidance else 0.0
            background_penalty = 0.88 if assessment.status == "background_only" else 1.0
            ts.combined_score = round(
                (semantic_component + role_component + quality_component + profile_component)
                * background_penalty,
                4,
            )
            tiered.append(ts)

        tiered.sort(key=lambda source: source.combined_score, reverse=True)
        ranked = tiered[:top_k]

        result = []
        for index, ts in enumerate(ranked, start=1):
            ts.source_id = f"S{index}"
            result.append(ts.as_dict())

        report = self._build_quality_report(result, excluded, patient_history, context_graph)
        return result, report

    def _assess_evidence_quality(
        self,
        source: Dict,
        question: str,
        patient_history: Optional["PatientHistoryContext"],
        context_graph: Optional["ContextGraph"],
        semantic_score: float,
        evidence_tier: int,
    ) -> EvidenceQualityAssessment:
        core_text = self._source_core_text(source)
        query_text = str(source.get("query") or "")
        question_terms = self._content_terms(question)
        content_overlap = self._term_overlap(question_terms, core_text)
        query_overlap = self._term_overlap(question_terms, query_text)
        semantic_signal = self._clamp_score(semantic_score)
        question_alignment = max(content_overlap, query_overlap * 0.5, semantic_signal * 0.75)

        validation_status, validation_score, validation_reason = self._validate_source(source)
        currency_status, currency_score, currency_reason = self._assess_currency(source, evidence_tier)

        patient_facts = self._collect_patient_facts(patient_history, context_graph)
        profile_matches, profile_score = self._profile_alignment(patient_facts, core_text)
        query_profile_matches, query_profile_score = self._profile_alignment(patient_facts, query_text)
        profile_specific_query = bool(query_profile_matches)
        patient_alignment = max(profile_score, min(query_profile_score, 0.5))

        mismatch_flags = self._detect_population_mismatch(core_text, query_text, question, patient_history)

        quality_score = (
            question_alignment * 0.45
            + validation_score * 0.35
            + currency_score * 0.20
        )
        if profile_matches:
            quality_score += profile_score * 0.10
        elif profile_specific_query:
            quality_score -= 0.20
        if mismatch_flags:
            quality_score -= 0.35
        quality_score = self._clamp_score(quality_score)

        reasons = []
        if question_alignment >= 0.45:
            reasons.append("Evidence text is strongly aligned with the current question.")
        elif question_alignment >= 0.20:
            reasons.append("Evidence text has partial alignment with the current question.")
        else:
            reasons.append("Evidence text has weak alignment with the current question.")

        if profile_matches:
            reasons.append("Matches stored patient-profile facts: " + ", ".join(profile_matches[:4]) + ".")
        elif profile_specific_query:
            reasons.append(
                "Retrieved by a patient-specific query, but the source text did not explicitly confirm that profile fit."
            )
        elif patient_facts:
            reasons.append("No explicit stored-profile match detected; use only for general context.")

        reasons.append(validation_reason)
        if currency_reason:
            reasons.append(currency_reason)
        reasons.extend(mismatch_flags)

        status = "question_aligned"
        usable_for_patient_specific_guidance = False
        if validation_status == "invalid":
            status = "excluded"
        elif question_alignment < 0.16 and semantic_signal < 0.22:
            status = "excluded"
        elif mismatch_flags:
            status = "excluded"
        elif quality_score < 0.25:
            status = "excluded"
        elif profile_matches:
            status = "patient_aligned"
            usable_for_patient_specific_guidance = True
        elif profile_specific_query:
            status = "background_only"

        return EvidenceQualityAssessment(
            status=status,
            quality_score=quality_score,
            question_alignment_score=question_alignment,
            patient_alignment_score=patient_alignment,
            patient_alignment_facts=profile_matches,
            mismatch_flags=mismatch_flags,
            quality_reasons=reasons,
            source_validation_status=validation_status,
            currency_status=currency_status,
            usable_for_patient_specific_guidance=usable_for_patient_specific_guidance,
        )

    def _assign_tier(self, source: Dict) -> int:
        """Assign Tier 1, 2, or 3 based on source metadata."""
        source_type = source.get("source_type", "")
        provider = (source.get("provider", "") or "").lower()
        title = (source.get("title", "") or "").lower()
        journal = (source.get("journal", "") or "").lower()

        if source_type == "official_guidance":
            return 1
        if any(provider_name in provider for provider_name in _TIER1_PROVIDERS):
            return 1

        for pattern in _TIER2_TITLE_PATTERNS:
            if pattern.search(title):
                return 2
        if any(journal_name in journal for journal_name in _TIER2_JOURNALS):
            return 2

        pub_type = (source.get("publication_type", "") or "").lower()
        if any(item in pub_type for item in ("review", "meta-analysis", "systematic")):
            return 2

        return 3

    @staticmethod
    def _compute_role_boost(tier: int, role_config: RoleConfig) -> float:
        preferred = role_config.preferred_evidence_tiers
        if tier not in preferred:
            return 0.0
        position = preferred.index(tier)
        return max(0.0, 1.0 - position * 0.2)

    @classmethod
    def _validate_source(cls, source: Dict) -> Tuple[str, float, str]:
        title = str(source.get("title") or "").strip()
        url = str(source.get("url") or "").strip()
        provider = str(source.get("provider") or "").strip()
        source_type = str(source.get("source_type") or "").strip()
        pmcid = str(source.get("pmcid") or "").strip()
        has_text = bool(str(source.get("detail_snippet") or source.get("snippet") or "").strip())

        if not title:
            return "invalid", 0.0, "Source failed validation: missing title."
        if not (url or provider or pmcid):
            return "invalid", 0.0, "Source failed validation: missing provenance."
        if not has_text:
            return "partial", 0.65, "Source validation is partial: no retrieved evidence excerpt was available."

        if source_type == "official_guidance":
            provider_lower = provider.lower()
            if any(trusted in provider_lower for trusted in _TIER1_PROVIDERS) or cls._is_trusted_guidance_url(url):
                return "valid", 1.0, "Source provenance validated as formal guidance."
            return "partial", 0.75, "Source provenance is official guidance but provider trust could not be fully confirmed."

        if source_type == "pubmed_literature" or pmcid:
            if pmcid and url:
                return "valid", 0.95, "Source provenance validated as PubMed Central literature."
            return "partial", 0.75, "Source provenance is biomedical literature but the PMC identifier or URL is incomplete."

        return "valid", 0.85, "Source has title, provenance, and retrieved excerpt."

    @staticmethod
    def _is_trusted_guidance_url(url: str) -> bool:
        lower = (url or "").lower()
        return any(
            host in lower
            for host in (
                "nhs.uk",
                "nice.org.uk",
                "gov.uk",
                "medlineplus.gov",
                "bnf.nice.org.uk",
            )
        )

    @staticmethod
    def _assess_currency(source: Dict, evidence_tier: int) -> Tuple[str, float, str]:
        if evidence_tier == 1:
            return "current_guidance", 1.0, "Formal guidance source; use the page content as the current authority."

        year = EvidenceRanker._source_year(source)
        if year is None:
            return "unknown", 0.80, "Publication year is unknown; avoid over-weighting it for changing practice."

        current_year = datetime.now(timezone.utc).year
        age = max(0, current_year - year)
        if age <= 5:
            return "recent", 1.0, f"Publication is recent ({year})."
        if age <= 10:
            return "moderately_recent", 0.90, f"Publication is moderately recent ({year})."
        if age <= 15:
            return "older", 0.70, f"Publication is older ({year}); verify against current guidance."
        return "stale", 0.55, f"Publication is old ({year}); use only as background unless confirmed elsewhere."

    @staticmethod
    def _source_year(source: Dict) -> Optional[int]:
        candidates = [source.get("year"), source.get("published"), source.get("date")]
        text = " ".join(str(item or "") for item in candidates)
        match = re.search(r"\b(19|20)\d{2}\b", text)
        if not match:
            return None
        year = int(match.group(0))
        current_year = datetime.now(timezone.utc).year + 1
        if 1900 <= year <= current_year:
            return year
        return None

    @classmethod
    def _collect_patient_facts(
        cls,
        patient_history: Optional["PatientHistoryContext"],
        context_graph: Optional["ContextGraph"],
    ) -> List[Dict]:
        facts: List[Dict] = []

        def add(label: str, fact_type: str) -> None:
            cleaned = cls._clean_fact_label(label)
            terms = sorted(cls._content_terms(cleaned))
            if not cleaned or not terms:
                return
            key = (fact_type, cleaned.lower())
            if key in {(item["type"], item["label"].lower()) for item in facts}:
                return
            facts.append({"label": cleaned, "type": fact_type, "terms": terms})

        if patient_history:
            for item in patient_history.known_conditions[:8]:
                add(item, "condition")
            for item in patient_history.known_medications[:8]:
                add(item, "medication")
            for item in patient_history.known_allergies[:8]:
                add(item, "allergy")
            if patient_history.age is not None and patient_history.age >= 65:
                facts.append(
                    {
                        "label": "older adult age group",
                        "type": "demographic",
                        "terms": ["aged", "elderly", "geriatric", "older"],
                    }
                )
            elif patient_history.age is not None and patient_history.age < 18:
                facts.append(
                    {
                        "label": "child or adolescent age group",
                        "type": "demographic",
                        "terms": ["adolescent", "child", "children", "paediatric", "pediatric"],
                    }
                )

        if context_graph:
            for node in context_graph.top_nodes(6):
                if node.node_type in {"condition", "medication", "allergy"} and node.relevance_score >= 0.30:
                    add(node.label, node.node_type)
            for edge in sorted(
                context_graph.edges,
                key=lambda item: item.relevance_score,
                reverse=True,
            )[:6]:
                if edge.relevance_score >= 0.20:
                    add(edge.source_name, edge.source_type)
                    add(edge.target_name, edge.target_type)
                    add(
                        f"{edge.source_name} {edge.relation.replace('_', ' ')} "
                        f"{edge.target_name}",
                        "relationship",
                    )

        return facts[:16]

    @classmethod
    def _profile_alignment(cls, patient_facts: List[Dict], text: str) -> Tuple[List[str], float]:
        if not patient_facts or not text:
            return [], 0.0

        matched = []
        for fact in patient_facts:
            if cls._fact_matches_text(fact, text):
                matched.append(fact["label"])

        denominator = min(4, max(1, len(patient_facts)))
        score = min(1.0, len(matched) / denominator)
        return matched, score

    @classmethod
    def _fact_matches_text(cls, fact: Dict, text: str) -> bool:
        lower = f" {text.lower()} "
        label = str(fact.get("label") or "").lower()
        if label and re.search(rf"\b{re.escape(label)}\b", lower):
            return True

        terms = list(fact.get("terms") or [])
        if not terms:
            return False
        matches = sum(1 for term in terms if re.search(rf"\b{re.escape(term)}\b", lower))
        required = 1 if len(terms) == 1 else max(2, round(len(terms) * 0.66))
        return matches >= required

    @staticmethod
    def _detect_population_mismatch(
        core_text: str,
        query_text: str,
        question: str,
        patient_history: Optional["PatientHistoryContext"],
    ) -> List[str]:
        if not patient_history:
            return []

        flags: List[str] = []
        source_text = f"{core_text} {query_text}".lower()
        question_lower = (question or "").lower()
        age = patient_history.age
        sex = (patient_history.biological_sex or "").strip().lower()

        if age is not None:
            if age >= 18 and _PEDIATRIC_RE.search(source_text) and not _PEDIATRIC_RE.search(question_lower):
                flags.append("Population mismatch: source focuses on children, but the stored profile is adult.")
            if age < 18 and (_ADULT_RE.search(source_text) or _OLDER_ADULT_RE.search(source_text)) and not _ADULT_RE.search(question_lower):
                flags.append("Population mismatch: source focuses on adults, but the stored profile is under 18.")
            if age < 65 and _OLDER_ADULT_RE.search(source_text) and not _OLDER_ADULT_RE.search(question_lower):
                flags.append("Population mismatch: source focuses on older adults, but the stored profile is under 65.")

        if sex.startswith("male") and _PREGNANCY_RE.search(source_text) and not _PREGNANCY_RE.search(question_lower):
            flags.append("Profile mismatch: source focuses on pregnancy/maternity but the stored biological sex is male.")
        if sex.startswith("female") and _MALE_ONLY_RE.search(source_text) and not _MALE_ONLY_RE.search(question_lower):
            flags.append("Profile mismatch: source focuses on male-only anatomy but the stored biological sex is female.")

        return flags

    @classmethod
    def _build_quality_report(
        cls,
        accepted_sources: List[Dict],
        excluded_sources: List[Dict],
        patient_history: Optional["PatientHistoryContext"],
        context_graph: Optional["ContextGraph"],
    ) -> Dict:
        status_counts: Dict[str, int] = {}
        for source in accepted_sources:
            status = source.get("evidence_quality_status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        patient_aligned = status_counts.get("patient_aligned", 0)
        if accepted_sources and patient_aligned:
            overall_status = "patient_aligned_evidence_available"
        elif accepted_sources:
            overall_status = "question_aligned_only"
        elif excluded_sources:
            overall_status = "no_sources_passed_quality_gate"
        else:
            overall_status = "no_live_evidence"

        profile_facts = [
            fact["label"]
            for fact in cls._collect_patient_facts(patient_history, context_graph)
        ]
        return {
            "overall_status": overall_status,
            "criteria": list(cls.QUALITY_CRITERIA),
            "accepted_source_count": len(accepted_sources),
            "excluded_source_count": len(excluded_sources),
            "status_counts": status_counts,
            "profile_facts_checked": profile_facts[:10],
            "excluded_sources": excluded_sources[:5],
        }

    @staticmethod
    def _source_core_text(source: Dict) -> str:
        parts = [
            source.get("title", ""),
            source.get("journal", ""),
            source.get("provider", ""),
            source.get("section", ""),
            source.get("detail_snippet", ""),
            source.get("snippet", ""),
            source.get("evidence", ""),
        ]
        return " ".join(str(part) for part in parts if part) or source.get("title", "Retrieved source")

    @classmethod
    def _source_text(cls, source: Dict) -> str:
        return " ".join(
            part
            for part in (
                cls._source_core_text(source),
                str(source.get("query", "") or ""),
            )
            if part
        )

    @staticmethod
    def _clean_fact_label(value: str) -> str:
        cleaned = str(value or "").strip()
        cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)
        cleaned = re.sub(r"\([^)]*\)", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip(" -:;,.")
        return cleaned

    @staticmethod
    def _content_terms(text: str) -> set:
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+-]{2,}\b", (text or "").lower())
        return {word for word in words if word not in _CONTENT_STOPWORDS}

    @classmethod
    def _term_overlap(cls, terms: set, text: str) -> float:
        if not terms or not text:
            return 0.0
        text_terms = cls._content_terms(text)
        if not text_terms:
            return 0.0
        exact = len(terms & text_terms)
        partial = sum(
            1
            for term in terms
            for target in text_terms
            if len(term) > 4 and term != target and (term in target or target in term)
        )
        return min(1.0, (exact + 0.35 * partial) / max(1, len(terms)))

    @staticmethod
    def _clamp_score(value: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def get_tiers_present(sources: List[Dict]) -> List[int]:
        tiers = {source.get("evidence_tier", 3) for source in sources if source.get("evidence_tier")}
        return sorted(tiers)
