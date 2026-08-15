"""
Evidence Ledger v2 (#10) tests: the deterministic risk-of-bias tag and
certainty sanity-check in backend/evidence_quality.py. Pure functions, no
DB/LLM involved.
"""
from backend.evidence_quality import apply_certainty_downgrade, assign_risk_of_bias


def test_assign_risk_of_bias_high_for_weak_designs():
    for design in ("case_report", "expert_opinion", "narrative_review"):
        assert assign_risk_of_bias(design) == "high"


def test_assign_risk_of_bias_some_concerns_for_observational_designs():
    for design in ("cohort_study", "case_control"):
        assert assign_risk_of_bias(design) == "some_concerns"


def test_assign_risk_of_bias_low_for_strong_designs():
    for design in ("rct", "systematic_review", "meta_analysis", "clinical_guideline"):
        assert assign_risk_of_bias(design) == "low"


def test_assign_risk_of_bias_unclear_for_unknown_or_unrecognised():
    assert assign_risk_of_bias("unknown") == "unclear"
    assert assign_risk_of_bias("not_a_real_design") == "unclear"
    assert assign_risk_of_bias("") == "unclear"


def test_apply_certainty_downgrade_drops_high_to_moderate_on_high_risk():
    assert apply_certainty_downgrade("high", "high") == "moderate"


def test_apply_certainty_downgrade_drops_moderate_to_low_on_high_risk():
    assert apply_certainty_downgrade("moderate", "high") == "low"


def test_apply_certainty_downgrade_never_upgrades():
    assert apply_certainty_downgrade("low", "low") == "low"
    assert apply_certainty_downgrade("very_low", "high") == "very_low"


def test_apply_certainty_downgrade_leaves_low_risk_certainty_unchanged():
    assert apply_certainty_downgrade("high", "low") == "high"
    assert apply_certainty_downgrade("moderate", "some_concerns") == "moderate"


def test_apply_certainty_downgrade_coerces_unrecognised_certainty():
    assert apply_certainty_downgrade("extremely_sure", "low") == "unknown"
