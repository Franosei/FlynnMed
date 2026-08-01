"""Pathway: medication queries, drug interactions, and pharmacology."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.intent_risk_classifier import IntentClassification
    from backend.role_router import RoleConfig


@dataclass
class PathwayContext:
    pathway_name: str
    additional_search_terms: List[str] = field(default_factory=list)
    safety_rules: List[str] = field(default_factory=list)
    preferred_sources: List[str] = field(default_factory=list)
    escalation_signals: List[str] = field(default_factory=list)
    evidence_tier_override: Optional[int] = None


def get_pathway_context(
    intent: "IntentClassification",
    role_config: "RoleConfig",
) -> PathwayContext:
    clinical_role = role_config.role_key in ("doctor", "nurse", "midwife", "physiotherapist", "healthcare_professional")

    search_terms = [
        "BNF drug interaction guideline",
        "NICE drug prescribing guideline",
        "MHRA drug safety alert",
    ]
    if not clinical_role:
        search_terms.append("NHS patient medication information")

    safety_rules = [
        "Use an authoritative regulator or medicine source applicable to the exact product and jurisdiction.",
        "Do not advise abrupt discontinuation when that may be harmful; direct changes through a pharmacist or prescriber.",
        "Distinguish established interactions from uncertain or formulation-mismatched evidence.",
        "Consider renal, hepatic, pregnancy, and breastfeeding factors only when relevant or explicitly asked.",
    ]
    if not clinical_role:
        safety_rules.append(
            "Patient-facing: recommend pharmacist or prescriber verification for prescription dosage queries. "
            "Do not provide specific mg dosage for prescription-only medicines."
        )
    else:
        safety_rules.append(
            "Clinician-facing: include therapeutic range, monitoring parameters, "
            "evidence class, and the clearest safe initial management action for prescribing decisions."
        )

    return PathwayContext(
        pathway_name="medications",
        additional_search_terms=search_terms,
        safety_rules=safety_rules,
        preferred_sources=["BNF", "BNFC", "MHRA", "NICE", "NHS"],
        escalation_signals=[
            "narrow therapeutic index", "warfarin toxicity", "lithium toxicity",
            "digoxin toxicity", "methotrexate toxicity", "anaphylaxis drug reaction",
            "Stevens-Johnson syndrome", "agranulocytosis drug-induced",
            "serotonin syndrome", "neuroleptic malignant syndrome",
        ],
        evidence_tier_override=None,
    )
