"""
Deterministic risk-of-bias tagging and a certainty sanity-check.

This is NOT a GRADE implementation. Real GRADE evaluates imprecision,
indirectness, inconsistency, and publication bias per outcome, usually from
a full evidence profile -- none of that is attempted here. What this module
adds is a single, cheap, deterministic layer on top of the LLM-self-reported
study_design/certainty already recorded on EvidenceClaim (backend/models/evidence.py):
a study-design-implied risk-of-bias tag, and one safe downgrade rule (high
risk of bias caps how high certainty is allowed to be). Both are recomputed
independently of whatever the extraction LLM claimed, the same "re-validate
at the DB boundary" discipline backend/evidence_ledger.py already applies to
study_design/certainty themselves.
"""
from __future__ import annotations

from backend.models.evidence import CERTAINTY_LEVELS

_HIGH_RISK_DESIGNS = {"case_report", "expert_opinion", "narrative_review"}
_SOME_CONCERNS_DESIGNS = {"cohort_study", "case_control"}
_LOW_RISK_DESIGNS = {"rct", "systematic_review", "meta_analysis", "clinical_guideline"}

_CERTAINTY_DOWNGRADE = {"high": "moderate", "moderate": "low"}


def assign_risk_of_bias(study_design: str) -> str:
    """Deterministic study-design -> risk-of-bias mapping. Unknown/unrecognised
    designs get "unclear" rather than a guessed value."""
    design = (study_design or "").strip().lower()
    if design in _HIGH_RISK_DESIGNS:
        return "high"
    if design in _SOME_CONCERNS_DESIGNS:
        return "some_concerns"
    if design in _LOW_RISK_DESIGNS:
        return "low"
    return "unclear"


def apply_certainty_downgrade(certainty: str, risk_of_bias: str) -> str:
    """The one GRADE rule this module is willing to enforce deterministically:
    a high risk of bias caps certainty at one level below what the LLM
    reported. Never upgrades -- only ever pulls certainty down."""
    certainty = (certainty or "unknown").strip().lower()
    if certainty not in CERTAINTY_LEVELS:
        certainty = "unknown"
    if risk_of_bias == "high" and certainty in _CERTAINTY_DOWNGRADE:
        return _CERTAINTY_DOWNGRADE[certainty]
    return certainty
