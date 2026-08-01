"""
Role-specific response templates, escalation banners, evidence tier labels,
and clinical safety disclaimers for the product UI.
"""
from __future__ import annotations

from typing import List

from backend.product_config import PRODUCT_NAME


CRISIS_RESPONSE = f"""\
## Urgent Safety Notice

Based on what you have described, this may be an emergency situation.

**Please act immediately:**
- **Call your local emergency number now**
- If someone is unresponsive and not breathing normally, start CPR and use an AED if available, following the emergency dispatcher's instructions
- If you can do so safely, go to the nearest emergency department
- Tell them exactly what you have told me

---

{PRODUCT_NAME} is not able to provide emergency care. Please reach out to a real person right now.
"""

CLINICAL_CRISIS_RESPONSE = """\
## Active Emergency

This describes an active emergency presentation.

- Activate your local emergency or resuscitation pathway now.
- Continue immediate assessment and treatment within your scope and current local protocol.
- Mobilize the appropriate senior, resuscitation, anaesthetic, obstetric, or specialty support without delay.
- Use verified point-of-care guidance rather than waiting for an educational evidence review.
"""

TIER_LABELS = {
    1: "Tier 1 - Formal Guidance",
    2: "Tier 2 - Review Evidence",
    3: "Tier 3 - Primary Research",
}

TIER_DESCRIPTIONS = {
    1: "NHS, NICE, MHRA or equivalent formal clinical guidance",
    2: "Systematic reviews, meta-analyses, or trusted evidence summaries",
    3: "Primary research from PubMed / Europe PMC",
}


def build_tier_badge(tier: int) -> str:
    label = TIER_LABELS.get(tier, f"Tier {tier}")
    return f"[{label}]"


def build_crisis_response(role_key: str = "patient") -> str:
    """Return emergency guidance appropriate to the established user role."""
    if role_key in ("doctor", "nurse", "midwife", "physiotherapist", "healthcare_professional"):
        return CLINICAL_CRISIS_RESPONSE
    return CRISIS_RESPONSE


def get_tier_description(tier: int) -> str:
    return TIER_DESCRIPTIONS.get(tier, "Evidence source")


def build_escalation_banner(reason: str, role_key: str = "patient") -> str:
    """Returns a prominent escalation notice to prepend to an answer."""
    if role_key in (
        "doctor", "nurse", "midwife", "physiotherapist", "healthcare_professional"
    ):
        return (
            f"> **Clinical escalation flag:** {reason}\n"
            "> **Action:** treat this as a red-flag presentation and follow the urgent local escalation pathway now.\n\n"
        )
    return (
        f"> **Important:** {reason}\n"
        "> Please seek appropriate medical help as described below.\n\n"
    )


def build_vulnerability_notice(flags: List[str]) -> str:
    """Returns a safety note for vulnerable population groups."""
    if not flags:
        return ""

    flag_text = ", ".join(flags)
    return (
        f"> **Vulnerable population notice:** This response has been reviewed with heightened "
        f"safety thresholds for: **{flag_text}**. Clinical judgement and appropriate specialist "
        f"referral should be considered.\n\n"
    )


def build_no_diagnosis_disclaimer(role_key: str = "patient") -> str:
    """Returns a role-appropriate no-diagnosis disclaimer."""
    if role_key == "patient":
        return (
            "\n\n---\n"
            f"*{PRODUCT_NAME} provides evidence-based health information and education. "
            "This is not a diagnosis and does not replace advice from your doctor or a qualified clinician. "
            "If you have concerns about your symptoms, please see a healthcare professional.*"
        )
    if role_key in (
        "doctor", "nurse", "midwife", "physiotherapist", "healthcare_professional"
    ):
        return (
            "\n\n---\n"
            "*Clinical decision support: verify applicability against the patient and current local guidance.*"
        )
    return (
        "\n\n---\n"
        "*This information is for educational purposes only and is not a substitute for "
        "professional clinical advice.*"
    )


ROLE_SECTION_HEADINGS: dict[str, List[str]] = {
    "patient": [
        "## Likely Explanation",
        "## What To Do Now",
        "## What To Monitor",
        "## Get Urgent Help If",
    ],
    "doctor": [
        "## Working Impression",
        "## Immediate Management",
        "## Investigations / Monitoring",
        "## Escalate Now If",
    ],
    "nurse": [
        "## Disposition",
        "## Immediate Nursing Actions",
        "## Monitor Right Now",
        "## Escalate Immediately If",
        "## What To Tell The Patient Or Family",
    ],
    "midwife": [
        "## Working Obstetric View",
        "## Immediate Actions",
        "## Monitor / Reassess",
        "## Escalate Immediately If",
        "## Patient Advice",
    ],
    "physiotherapist": [
        "## Working MSK Interpretation",
        "## Immediate Management",
        "## Loading / Movement Advice",
        "## Escalate Or Refer If",
    ],
    "healthcare_professional": [
        "## Prioritized Decision",
        "## Action Within Scope",
        "## Escalate / Refer If",
    ],
    "caregiver": [
        "## What This Most Likely Means",
        "## What To Do Now",
        "## What To Monitor",
        "## Get Urgent Help If",
        "## Caregiver Actions",
    ],
}

DEFAULT_SECTION_HEADINGS = [
    "## Working Impression",
    "## What To Do Now",
    "## What To Monitor",
]


def get_section_headings(role_key: str) -> List[str]:
    return ROLE_SECTION_HEADINGS.get(role_key, DEFAULT_SECTION_HEADINGS)


def get_section_headings_text(role_key: str) -> str:
    headings = get_section_headings(role_key)
    return "\n".join(headings)


ROLE_PERSONA_BLOCKS: dict[str, str] = {
    "patient": (
        "You are speaking with a patient or individual seeking health information. "
        "Use plain, accessible language and avoid unexplained medical jargon. "
        "Give a clear working explanation without overstating certainty. "
        "Be specific about what the person should do next, including timeframe and where to seek help. "
        "Prefer a clear route such as self-care, pharmacist, primary-care review, same-day review, or emergency care. "
        "Use a warm, calm, and direct tone."
    ),
    "doctor": (
        "You are supporting a qualified medical doctor. "
        "Use precise clinical terminology. "
        "Act like a safe, competent clinical colleague, not a senior specialist giving definitive consultant-level direction. "
        "Lead with one prioritized working impression, a clear route and disposition, and practical initial management supported by the evidence. "
        "Do not produce an unranked differential list. If the facts are insufficient to diagnose, state the single must-not-miss "
        "syndrome or clinical decision first, then name only the one or two discriminators that would change that decision. "
        "Be decisive about the action even when the diagnosis remains provisional. "
        "Avoid over-explaining basic clinical concepts. "
        "Attach an inline citation to the working impression and management action, using relevant NICE/SIGN or equivalent formal guidance where available. "
        "Do not overstate the strength of evidence, and defer specialty-level or definitive decisions when the evidence is thin."
    ),
    "nurse": (
        "You are supporting a registered nurse. "
        "Use intermediate clinical language appropriate for nursing practice. "
        "Lead with disposition and immediate nursing actions before explanation. "
        "Focus on monitoring parameters, escalation thresholds, and what needs doing right now. "
        "Give one clear disposition and a specific, practical first-pass management plan rather than a list of possible diagnoses. "
        "If the diagnosis is uncertain, prioritize the must-not-miss syndrome and the finding that changes escalation. "
        "Keep teaching brief and only include it when it changes the immediate decision. "
        "Include patient communication points that can be used directly with patients or families. "
        "Cite the disposition and action inline using NICE guidance or the relevant formal source where available."
    ),
    "midwife": (
        "You are supporting a registered midwife. "
        "Apply heightened safety thresholds for all pregnancy, postpartum, and newborn-related content. "
        "Use maternity-specific clinical terminology. "
        "Lead with the safest proportionate disposition and the immediate maternity actions required now. "
        "Include only presentation-relevant obstetric warning signs and referral triggers; do not give an unranked differential list. "
        "Cite the disposition and immediate action inline using formal guidance where available. "
        "For any medication or intervention question, specifically consider pregnancy safety."
    ),
    "physiotherapist": (
        "You are supporting a physiotherapist. "
        "Focus on MSK interpretation, functional movement, and rehabilitation principles. "
        "Use physiotherapy-specific terminology (ROM, load management, neural tension, etc.). "
        "Give one prioritized working interpretation and a specific initial management plan, including load advice, contraindications, and onward referral thresholds. "
        "Do not give an unranked differential list; lead with the must-not-miss non-MSK concern when one changes disposition. "
        "Cite the interpretation and onward-referral action inline using formal guidance where available. "
        "Include neurovascular or non-mechanical warning signs only when connected to the presentation."
    ),
    "healthcare_professional": (
        "You are supporting an authenticated healthcare professional whose exact regulated profession is not specified. "
        "Use concise clinical terminology and lead with one prioritized clinical decision, disposition, and immediate action. "
        "Keep every action explicitly within the user's locally authorised scope; do not assume physician prescribing, diagnostic, or discharge authority. "
        "Do not give an unranked differential list. When the diagnosis is uncertain, lead with the must-not-miss syndrome and the discriminator that changes escalation. "
        "Cite the decision and action inline using formal guidance where available, and state when local policy or senior review is required."
    ),
    "caregiver": (
        "You are speaking with a caregiver supporting a patient or family member. "
        "Use accessible language with empathetic framing. "
        "Focus on what the caregiver can practically do next and when to seek professional help. "
        "Prefer specific care routes and timeframes over general caution. "
        "Include caregiver-specific support considerations. "
        "Include escalation guidance when the presentation calls for it."
    ),
}

DEFAULT_PERSONA_BLOCK = (
    f"You are {PRODUCT_NAME}, a safe and competent clinical information assistant. "
    "Provide decisive, evidence-grounded guidance with a clear next-step plan, "
    "but do not present yourself as operating at a confident senior clinical or specialist level."
)


def get_persona_block(role_key: str) -> str:
    return ROLE_PERSONA_BLOCKS.get(role_key, DEFAULT_PERSONA_BLOCK)
