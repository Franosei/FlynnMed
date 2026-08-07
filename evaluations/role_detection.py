"""Resolve the intended audience for an evaluation case.

Production permissions continue to come from the authenticated account. This
module only creates an isolated evaluation account whose role matches the case
audience. Resolution is deterministic and records its reason so benchmark role
assignment can be audited independently of answer quality.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from evaluations.models import EvalCase


@dataclass(frozen=True)
class RoleResolution:
    role: str
    reason: str
    confidence: float


_ROLE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "doctor",
        re.compile(
            r"\b(?:I(?:'m| am)|as) an?\s+(?:[a-z-]+\s+){0,4}"
            r"(?:physician|doctor|doc|GP|general practitioner|surgeon|resident|"
            r"consultant|attending|cardiologist|urologist|allergist|"
            r"anaesthetist|anesthesiologist|psychiatrist|paediatrician|pediatrician|"
            r"intensivist)\b",
            re.I,
        ),
    ),
    (
        "nurse",
        re.compile(
            r"\b(?:I(?:'m| am)|as|act like) an?\s+(?:[a-z-]+\s+){0,4}"
            r"(?:nurse|nurse practitioner|registered nurse|RN)\b",
            re.I,
        ),
    ),
    (
        "midwife",
        re.compile(
            r"\b(?:I(?:'m| am)|as) an?\s+(?:[a-z-]+\s+){0,3}midwi(?:fe|ves)\b", re.I
        ),
    ),
    (
        "physiotherapist",
        re.compile(
            r"\b(?:I(?:'m| am)|as) an?\s+(?:[a-z-]+\s+){0,3}"
            r"(?:physiotherapist|physical therapist)\b",
            re.I,
        ),
    ),
    (
        "healthcare_professional",
        re.compile(
            r"\b(?:I(?:'m| am)|as) an?\s+(?:[a-z-]+\s+){0,3}"
            r"(?:clinician|pharmacist|paramedic|healthcare professional)\b",
            re.I,
        ),
    ),
]

_NURSING_CONTEXT = re.compile(
    r"\b(?:senior nurse|staff nurse|nursing perspective|nursing assessment|"
    r"nurse-led|psych ward|on the ward)\b",
    re.I,
)
_CLINICIAN_CONTEXT = re.compile(
    r"\b(?:my patients?|one of my patients?|the patient|this patient|patient presents?|"
    r"patient has|patient with|patient is|patient was|patient started|patient taking|"
    r"we (?:admitted|prescribed|started|ordered|examined|diagnosed|treated)|"
    r"I (?:prescribed|started|ordered|examined|diagnosed|treated)|"
    r"routine exam|clinical practice|discharge (?:note|summary)|progress note|"
    r"dialysis team|Dr\.?\s+[A-Z])\b",
    re.I,
)
_PATIENT_CONTEXT = re.compile(
    r"\b(?:I have|I've had|I feel|I take|my symptoms?|my doctor|my GP|my medicine|"
    r"my child|my baby|my toddler|my husband|my wife|my labs?|my eGFR|"
    r"I gave birth|should I take)\b",
    re.I,
)
_PROFESSIONAL_SIGNALS = [
    re.compile(
        r"\b(?:guidelines?|consensus|systematic review|meta-analysis|clinical trials?|"
        r"protocol|recent evidence|latest evidence)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:contraindications?|perioperative|work-?up|differential diagnosis|"
        r"indications?|haemodynamic|hemodynamic)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:dose adjustment|dosing interval|titrate|titration|renal dosing|"
        r"creatinine clearance|eGFR|LFTs?|therapeutic monitoring|trough levels?)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:treatment approach|management approach|recommended window|"
        r"advanced imaging|first-line treatment|maintenance therapy|"
        r"clinical decision|assessment and plan)\b",
        re.I,
    ),
]

_APOSTROPHES = str.maketrans({"\u2018": "'", "\u2019": "'"})
_BROKEN_APOSTROPHES = ("\u00e2\u20ac\u02dc", "\u00e2\u20ac\u2122")


def _user_text(case: EvalCase) -> str:
    text = " ".join(
        turn.content for turn in case.conversation if turn.role.lower() == "user"
    )
    text = text.translate(_APOSTROPHES)
    for broken in _BROKEN_APOSTROPHES:
        text = text.replace(broken, "'")
    return text


def _audience_tag(case: EvalCase) -> str:
    for tag in case.tags:
        if tag == "physician_agreed_category:health-professional":
            return "professional"
        if tag == "physician_agreed_category:not-health-professional":
            return "patient"
    return ""


def resolve_case_role(case: EvalCase) -> RoleResolution:
    """Resolve role from explicit identity, dataset audience, then context."""
    text = _user_text(case)

    for role_key, pattern in _ROLE_PATTERNS:
        if pattern.search(text):
            return RoleResolution(role_key, "explicit clinical role in user text", 1.0)

    audience_tag = _audience_tag(case)
    nursing_context = bool(_NURSING_CONTEXT.search(text))
    if audience_tag == "professional":
        role = "nurse" if nursing_context else "healthcare_professional"
        return RoleResolution(
            role, "HealthBench health-professional audience tag", 0.95
        )
    if audience_tag == "patient":
        return RoleResolution(
            "patient", "HealthBench not-health-professional audience tag", 0.95
        )

    if _CLINICIAN_CONTEXT.search(text):
        role = "nurse" if nursing_context else "healthcare_professional"
        return RoleResolution(
            role, "case is framed around clinician-managed patient care", 0.9
        )

    professional_signal_count = sum(
        bool(pattern.search(text)) for pattern in _PROFESSIONAL_SIGNALS
    )
    if professional_signal_count >= 2 and not _PATIENT_CONTEXT.search(text):
        return RoleResolution(
            "healthcare_professional",
            "multiple professional clinical-language signals without a patient framing",
            0.8,
        )

    return RoleResolution("patient", "no reliable professional audience signal", 0.8)


def detect_stated_role(case: EvalCase) -> str:
    """Backward-compatible role-only interface used by existing callers."""
    return resolve_case_role(case).role


def eval_account_username(role: str, case_id: str) -> str:
    safe_case_id = re.sub(r"[^a-zA-Z0-9_-]", "", case_id)[:32]
    return f"eval-harness-{role}-{safe_case_id}"
