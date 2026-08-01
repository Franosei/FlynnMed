"""
Role routing: maps user profile role strings to structured RoleConfig bundles
that control downstream evidence tiering, escalation thresholds, and LLM persona.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List

from backend.response_templates import get_persona_block, get_section_headings


@dataclass
class RoleConfig:
    role_key: str                           # canonical role used by policy and presentation
    display_label: str
    terminology_level: str                  # "lay", "intermediate", "clinical"
    escalation_threshold: str               # "low" = escalate readily, "medium", "high"
    vulnerable_population_flags: List[str]  # always-on population flags for this role
    preferred_evidence_tiers: List[int]     # e.g. [1, 2] = prefer Tier 1 + 2 first
    no_diagnosis_enforcement: bool = True

    @property
    def system_prompt_persona(self) -> str:
        return get_persona_block(self.role_key)

    @property
    def section_headings(self) -> List[str]:
        return get_section_headings(self.role_key)

    @property
    def section_headings_text(self) -> str:
        return "\n".join(self.section_headings)


# ── Role definitions ───────────────────────────────────────────────────────────
_ROLE_CONFIGS: dict[str, RoleConfig] = {
    "patient": RoleConfig(
        role_key="patient",
        display_label="Patient",
        terminology_level="lay",
        escalation_threshold="low",          # escalate most readily
        vulnerable_population_flags=[],
        preferred_evidence_tiers=[1, 2, 3],
    ),
    "caregiver": RoleConfig(
        role_key="caregiver",
        display_label="Caregiver",
        terminology_level="lay",
        escalation_threshold="low",
        vulnerable_population_flags=[],
        preferred_evidence_tiers=[1, 2, 3],
    ),
    "doctor": RoleConfig(
        role_key="doctor",
        display_label="Doctor / Physician",
        terminology_level="clinical",
        escalation_threshold="high",         # clinical judgement assumed
        vulnerable_population_flags=[],
        preferred_evidence_tiers=[1, 2, 3],
    ),
    "nurse": RoleConfig(
        role_key="nurse",
        display_label="Nurse",
        terminology_level="intermediate",
        escalation_threshold="medium",
        vulnerable_population_flags=[],
        preferred_evidence_tiers=[1, 2, 3],
    ),
    "midwife": RoleConfig(
        role_key="midwife",
        display_label="Midwife",
        terminology_level="intermediate",
        escalation_threshold="low",          # heightened for maternal safety
        vulnerable_population_flags=["pregnancy", "postpartum", "newborn"],
        preferred_evidence_tiers=[1, 2, 3],
    ),
    "physiotherapist": RoleConfig(
        role_key="physiotherapist",
        display_label="Physiotherapist",
        terminology_level="intermediate",
        escalation_threshold="medium",
        vulnerable_population_flags=[],
        preferred_evidence_tiers=[1, 2, 3],
    ),
    "healthcare_professional": RoleConfig(
        role_key="healthcare_professional",
        display_label="Other Clinician",
        terminology_level="clinical",
        escalation_threshold="medium",
        vulnerable_population_flags=[],
        preferred_evidence_tiers=[1, 2, 3],
    ),
}

# ── Alias map: landing-page role strings → canonical role_key ─────────────────
# Covers all existing values from the old ROLES list + new clinical roles
_ALIAS_MAP: dict[str, str] = {
    # Old system (backward compat)
    "individual": "patient",
    "caregiver": "caregiver",
    "clinician / care team": "doctor",
    "clinician": "doctor",
    "care team": "nurse",

    # New clinical role strings (from updated signup form)
    "patient / individual": "patient",
    "patient": "patient",
    "doctor / physician": "doctor",
    "doctor": "doctor",
    "physician": "doctor",
    "nurse": "nurse",
    "midwife": "midwife",
    "physiotherapist": "physiotherapist",
    "physio": "physiotherapist",
    "other clinician": "healthcare_professional",
}


class RoleRouter:
    """Maps any profile role string to a structured RoleConfig."""

    def resolve(self, profile_role: str) -> RoleConfig:
        """
        Normalise → alias lookup → role config.
        Falls back to "patient" config for any unrecognised value.
        """
        key = (profile_role or "").strip().lower()
        canonical = _ALIAS_MAP.get(key)

        # Try prefix match if exact match fails
        if not canonical:
            for alias, role_key in _ALIAS_MAP.items():
                if key.startswith(alias) or alias.startswith(key):
                    canonical = role_key
                    break

        canonical = canonical or "patient"
        return _ROLE_CONFIGS[canonical]

    @staticmethod
    def get_all_clinical_roles() -> List[str]:
        """Returns display labels for the signup form dropdown."""
        return [
            "Patient / Individual",
            "Caregiver",
            "Doctor / Physician",
            "Nurse",
            "Midwife",
            "Physiotherapist",
            "Other Clinician",
        ]
