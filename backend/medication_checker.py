from __future__ import annotations

import re
from itertools import combinations
from typing import Dict, Iterable

import requests


OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
LOOKUP_FIELDS = (
    "openfda.generic_name",
    "openfda.brand_name",
    "openfda.substance_name",
)
INTERACTION_FIELDS = (
    ("drug_interactions", "Drug interactions"),
    ("drug_interactions_table", "Interaction table"),
    ("contraindications", "Contraindications"),
    ("warnings_and_cautions", "Warnings and cautions"),
)
HIGH_RISK_MARKERS = (
    "contraindicat",
    "avoid concomitant",
    "avoid use",
    "major interaction",
    "life-threatening",
    "serious bleeding",
    "fatal",
    "do not use",
)
MONITOR_MARKERS = (
    "monitor",
    "dose adjustment",
    "increase",
    "decrease",
    "increased risk",
    "reduced effect",
    "bleeding risk",
    "toxicity",
)


def _clean_text(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _severity_rank(value: str) -> int:
    return {"high": 3, "monitor": 2, "mentioned": 1}.get(value, 0)


class MedicationInteractionChecker:
    def __init__(self, timeout: int = 20) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self._resolution_cache: Dict[str, Dict | None] = {}

    def resolve_medication(self, medication_name: str) -> Dict | None:
        key = (medication_name or "").strip().lower()
        if not key:
            return None
        if key in self._resolution_cache:
            return self._resolution_cache[key]

        safe_name = medication_name.replace('"', "").strip()
        query_parts = [f'{field}:"{safe_name}"' for field in LOOKUP_FIELDS]
        params = {
            "search": " OR ".join(query_parts),
            "limit": "1",
        }
        try:
            response = self.session.get(OPENFDA_LABEL_URL, params=params, timeout=self.timeout)
            if response.status_code == 404:
                self._resolution_cache[key] = None
                return None
            response.raise_for_status()
            payload = response.json()
        except Exception:
            self._resolution_cache[key] = None
            return None

        results = payload.get("results", [])
        if not results:
            self._resolution_cache[key] = None
            return None

        result = results[0]
        openfda = result.get("openfda", {})
        aliases = {
            medication_name.strip(),
            *(openfda.get("generic_name") or []),
            *(openfda.get("brand_name") or []),
            *(openfda.get("substance_name") or []),
        }
        aliases = {alias.strip() for alias in aliases if alias and alias.strip()}
        pharm_classes = {
            *(openfda.get("pharm_class_epc") or []),
            *(openfda.get("pharm_class_cs") or []),
            *(openfda.get("pharm_class_moa") or []),
        }
        pharm_classes = {cls.strip() for cls in pharm_classes if cls and cls.strip()}

        sections = []
        for field_name, label in INTERACTION_FIELDS:
            values = result.get(field_name, [])
            if isinstance(values, str):
                values = [values]
            cleaned = " ".join(_clean_text(value) for value in values if _clean_text(value))
            if cleaned:
                sections.append(
                    {
                        "field": field_name,
                        "label": label,
                        "text": cleaned,
                    }
                )

        resolved = {
            "query_name": medication_name.strip(),
            "canonical_name": (
                (openfda.get("generic_name") or [])
                or (openfda.get("brand_name") or [])
                or [medication_name.strip()]
            )[0],
            "aliases": sorted(aliases),
            "pharm_classes": sorted(pharm_classes),
            "sections": sections,
            "api_url": response.url,
            "effective_time": result.get("effective_time", ""),
        }
        self._resolution_cache[key] = resolved
        return resolved

    def check_interactions(self, medication_names: Iterable[str]) -> Dict:
        unique_names = []
        for name in medication_names:
            cleaned = (name or "").strip()
            if cleaned and cleaned.lower() not in {item.lower() for item in unique_names}:
                unique_names.append(cleaned)

        resolved = []
        unresolved = []
        for name in unique_names:
            resolved_medication = self.resolve_medication(name)
            if resolved_medication:
                resolved.append(resolved_medication)
            else:
                unresolved.append(name)

        alerts = []
        for left, right in combinations(resolved, 2):
            alert = self._build_pair_alert(left, right)
            if alert:
                alerts.append(alert)

        alerts.sort(key=lambda item: (_severity_rank(item.get("severity", "")), item.get("pair", "")), reverse=True)
        return {
            "resolved_medications": resolved,
            "unresolved_medications": unresolved,
            "alerts": alerts,
        }

    def _build_pair_alert(self, left: Dict, right: Dict) -> Dict | None:
        evidence_matches = []
        for source, target in ((left, right), (right, left)):
            match = self._find_match(source, target)
            if match:
                evidence_matches.append(match)

        if not evidence_matches:
            return None

        evidence_matches.sort(key=lambda item: _severity_rank(item["severity"]), reverse=True)
        top_match = evidence_matches[0]
        return {
            "pair": f"{left['canonical_name']} + {right['canonical_name']}",
            "severity": top_match["severity"],
            "summary": top_match["summary"],
            "evidence": evidence_matches,
        }

    def _find_match(self, source: Dict, target: Dict) -> Dict | None:
        target_aliases = sorted(
            {alias for alias in target.get("aliases", []) if len(alias) >= 3},
            key=len,
            reverse=True,
        )
        for section in source.get("sections", []):
            section_text = section.get("text", "")
            for alias in target_aliases:
                pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", re.IGNORECASE)
                match = pattern.search(section_text)
                if not match:
                    continue
                excerpt = self._extract_excerpt(section_text, match.start(), match.end())
                severity = self._classify_excerpt(excerpt)
                summary = (
                    f"{source['canonical_name']} label mentions {target['canonical_name']} in "
                    f"{section['label'].lower()}: {excerpt}"
                )
                return {
                    "source_medication": source["canonical_name"],
                    "target_medication": target["canonical_name"],
                    "section": section["label"],
                    "severity": severity,
                    "summary": summary,
                    "source_url": source.get("api_url", ""),
                    "effective_time": source.get("effective_time", ""),
                }
        return None

    @staticmethod
    def _extract_excerpt(text: str, start: int, end: int, window: int = 220, max_chars: int = 420) -> str:
        left = max(0, start - window)
        right = min(len(text), end + window)

        # Prefer sentence boundaries when available so the UI does not show a
        # label fragment chopped in the middle of a thought.
        left_slice = text[left:start]
        sentence_start_candidates = [match.end() for match in re.finditer(r"[.!?;:]\s+", left_slice)]
        if sentence_start_candidates:
            left = left + sentence_start_candidates[-1]

        right_slice = text[end:right]
        sentence_end_match = re.search(r"[.!?;:](?:\s|$)", right_slice)
        if sentence_end_match:
            right = end + sentence_end_match.end()

        excerpt = text[left:right].strip()
        prefix = "... " if left > 0 else ""
        suffix = ""

        if len(excerpt) > max_chars:
            trimmed = excerpt[:max_chars].rstrip()
            last_space = trimmed.rfind(" ")
            if last_space > max_chars * 0.6:
                trimmed = trimmed[:last_space]
            excerpt = trimmed.rstrip(" ,;:")
            suffix = " ..."
        elif right < len(text):
            suffix = " ..."

        return f"{prefix}{excerpt}{suffix}".strip()

    @staticmethod
    def _classify_excerpt(excerpt: str) -> str:
        lowered = excerpt.lower()
        if any(marker in lowered for marker in HIGH_RISK_MARKERS):
            return "high"
        if any(marker in lowered for marker in MONITOR_MARKERS):
            return "monitor"
        return "mentioned"


def check_allergy_conflicts(resolved_candidate: Dict, allergies: list[dict]) -> list[dict]:
    """
    Screens a resolved candidate medication (from
    MedicationInteractionChecker.resolve_medication) against a patient's
    recorded allergies. This is a best-effort heuristic screen for clinician
    review, NOT a definitive pharmacological safety determination -- it only
    catches (a) the candidate's own name/aliases exactly matching a recorded
    allergy name, and (b) a recorded allergy name appearing inside one of the
    candidate's openFDA drug-class tags (e.g. an allergy to "penicillin"
    against a candidate whose pharm_class includes "Penicillin-class
    Antibacterial"). It cannot reason about cross-reactivity the class tags
    don't capture, and openFDA's pharm_class_epc/cs/moa fields are sparsely
    and inconsistently populated across label records in practice -- an empty
    pharm_classes list on the candidate is common and does not mean "no drug
    class exists," only that this particular label record didn't carry the
    tag. The exact-name check has no such gap. The clinician remains the
    final safety check either way -- this exists to surface likely conflicts
    prominently, not to make the call.
    """
    if not resolved_candidate or not allergies:
        return []

    candidate_names = {
        resolved_candidate.get("query_name", ""),
        resolved_candidate.get("canonical_name", ""),
        *resolved_candidate.get("aliases", []),
    }
    candidate_names = {n.strip().lower() for n in candidate_names if n and n.strip()}
    pharm_classes = resolved_candidate.get("pharm_classes", []) or []

    flags: list[dict] = []
    for allergy in allergies:
        allergy_name = (allergy.get("name") or "").strip()
        if not allergy_name:
            continue
        allergy_lower = allergy_name.lower()

        if allergy_lower in candidate_names:
            flags.append(
                {
                    "allergy_name": allergy_name,
                    "match_type": "exact_name",
                    "matched_text": allergy_name,
                    "severity": allergy.get("severity") or "unknown",
                    "summary": (
                        f"Recorded allergy '{allergy_name}' matches the candidate "
                        f"medication's own name/alias."
                    ),
                }
            )
            continue

        for pharm_class in pharm_classes:
            if allergy_lower and allergy_lower in pharm_class.lower():
                flags.append(
                    {
                        "allergy_name": allergy_name,
                        "match_type": "drug_class",
                        "matched_text": pharm_class,
                        "severity": allergy.get("severity") or "unknown",
                        "summary": (
                            f"Recorded allergy '{allergy_name}' appears in the candidate "
                            f"medication's drug class '{pharm_class}' -- possible "
                            f"cross-reactivity, confirm before prescribing."
                        ),
                    }
                )
                break

    return flags
