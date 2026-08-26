import json
from pathlib import Path

import pytest

from backend.clinical_decision_support import (
    RISK_LEVEL_RANK,
    ClinicalDecisionSupportEngine,
)
from backend.intent_risk_classifier import IntentClassification, IntentRiskClassifier
from backend.role_router import RoleRouter


DATASET = (
    Path(__file__).resolve().parents[1]
    / "release"
    / "critical_presentations.v1.jsonl"
)


def _cases():
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line]


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["case_id"])
def test_locked_critical_presentation(case):
    if case.get("test_layer") in {
        "intent_crisis_prescreen",
        "intent_urgent_prescreen",
    }:
        result = object.__new__(IntentRiskClassifier).classify(
            case["input"], role_key=case.get("role", "patient")
        )
        assert result.risk_level == case["expected"]["risk_level"]
        assert result.crisis_detected is case["expected"]["crisis_detected"]
        return
    intent = IntentClassification(
        intent_category="symptom_triage",
        risk_level="routine",
        presentation_hint=case["presentation_hint_for_backstop_test"],
    )
    decision = ClinicalDecisionSupportEngine().assess(
        case["input"], intent, RoleRouter().resolve("patient")
    )
    expected = case["expected"]

    if "pathway_id" in expected:
        assert decision.pathway_id == expected["pathway_id"]
    if "pathway_id_not" in expected:
        assert decision.pathway_id != expected["pathway_id_not"]
    if "next_step" in expected:
        assert decision.next_step == expected["next_step"]
    if "minimum_risk_level" in expected:
        assert RISK_LEVEL_RANK[decision.minimum_risk_level] >= RISK_LEVEL_RANK[
            expected["minimum_risk_level"]
        ]
