"""Deterministic request-mode routing that never changes authorization.

The authenticated role remains the sole authority for permissions and safety
thresholds. This module controls only presentation and whether a literal
transformation needs evidence retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


_DOCUMENT_TYPE = re.compile(
    r"\b(?:soap\s+note|progress\s+note|outpatient\s+note|clinic(?:al)?\s+note|"
    r"medical\s+note|discharge\s+(?:summary|letter)|clinic\s+letter|"
    r"assessment\s+and\s+plan)\b",
    re.IGNORECASE,
)
_DOCUMENT_ACTION = re.compile(
    r"\b(?:draft|write|create|prepare|format|convert|turn|put|compile)\b",
    re.IGNORECASE,
)
_TRANSLATION_ACTION = re.compile(
    r"\b(?:translate|translation|traduz(?:ir|a)?|tradu[cç][aã]o|"
    r"traduire|traduction|traducir|traducci[oó]n)\b",
    re.IGNORECASE,
)
_PROFESSIONAL_EVIDENCE = re.compile(
    r"\b(?:clinical\s+trials?|systematic\s+reviews?|meta[- ]analys(?:is|es)|"
    r"guidelines?|guidance|consensus\s+statement|latest\s+advancements?|recent\s+evidence|"
    r"landmark\s+trials?|indications?\s+and\s+contraindications?|"
    r"periprocedural|protocols?|implementation\s+framework)\b",
    re.IGNORECASE,
)
_CLINICAL_ROLE_KEYS = {"doctor", "nurse", "midwife", "physiotherapist", "healthcare_professional"}

# A clinician asking to "compile a flowsheet/checklist/template" wants a
# complete, exhaustive reference artifact -- fundamentally different from a
# quick clinical question, and the default clinician answer instructions
# (short, ~120-word, few-section) actively work against it. Deliberately
# does NOT route into "documentation" mode (_DOCUMENT_TYPE/_DOCUMENT_ACTION
# below) even though "compile" also matches that action verb -- documentation
# mode skips evidence retrieval entirely (literal_transformation=True), which
# would reopen the no-ungrounded-claims concern this session already fixed
# once for citations. This only changes response *instructions*, not the
# evidence-retrieval path.
_STRUCTURED_COMPILATION_ACTION = re.compile(
    r"\b(?:compile|put together|produce|generate|create|draft|build)\b", re.IGNORECASE
)
_STRUCTURED_ARTIFACT_TYPE = re.compile(
    r"\b(?:flowsheet|flow\s+sheet|checklist|template|tracking\s+sheet|"
    r"scoring\s+sheet|proforma|standardi[sz]ed\s+(?:form|chart|sheet)|"
    r"rounding\s+(?:chart|sheet|template))\b",
    re.IGNORECASE,
)

# A pure chart-data-lookup question ("what medication is she on", "does she have any
# allergies") is fully answerable from data already loaded into the prompt (patient_history) --
# it needs no external NHS/PubMed retrieval. _CHART_FIELD + _CHART_LOOKUP_ACTION are the
# positive signal; _ADVISORY_SIGNAL is a deliberately broad negative override, since a false
# positive here (skipping retrieval on a question that actually needed evidence) is worse than
# a false negative (an unnecessary search) -- any evaluative/advisory language at all disqualifies
# the lookup classification, regardless of how strongly the positive signal matched.
_CHART_FIELD = re.compile(
    r"\b(?:medications?|meds?|prescriptions?|drugs?|"
    r"allerg(?:y|ies)|adverse\s+reactions?|"
    r"conditions?|diagnos(?:is|es)|problem\s+list|comorbidit(?:y|ies)|"
    r"vitals?|blood\s+pressure|\bBPs?\b|heart\s+rate|\bHR\b|temperature|weight|height|"
    r"labs?|lab\s+results?|"
    r"symptoms?|"
    r"clinical\s+relationships?|recorded\s+(?:links?|connections?)|causal\s+links?|"
    r"triage\s+(?:history|summary)|"
    r"care\s+plans?)\b",
    re.IGNORECASE,
)
_CHART_LOOKUP_ACTION = re.compile(
    r"\b(?:what|which|list|show|recall|remind\s+me|"
    r"when\s+(?:was|did)|does\s+(?:she|he|the\s+patient)\s+have|"
    r"is\s+(?:she|he|the\s+patient)\s+(?:on|taking)|"
    r"am\s+i\s+(?:on|taking)|recent(?:ly)?)\b",
    re.IGNORECASE,
)
_ADVISORY_SIGNAL = re.compile(
    r"\b(?:should|is\s+it\s+safe|appropriate|contraindicat\w*|interact\w*|"
    r"adjust\w*|manage\w*|treat\w*|risk\s+of|why|caus\w*|recommend\w*|"
    r"given\s+(?:her|his|their)|need\s+to|ok\s+to|okay\s+to|change\w*)\b",
    re.IGNORECASE,
)


def _messages(
    chat_history: Sequence[Mapping[str, object]] | None,
) -> Iterable[tuple[str, str]]:
    for message in chat_history or []:
        role = str(message.get("role") or "").strip().lower()
        content = str(message.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            yield role, content


@dataclass(frozen=True)
class TaskModeDecision:
    mode: str = "clinical_answer"
    presentation_audience: str = "patient"
    requires_evidence_retrieval: bool = True
    literal_transformation: bool = False
    requests_structured_compilation: bool = False
    reason: str = "default clinical information mode"

    @property
    def is_transformation(self) -> bool:
        return self.mode in {"documentation", "translation"}

    def prompt_block(self) -> str:
        authorization_guard = (
            "This presentation decision does not change the authenticated role, permissions, "
            "safety thresholds, or access to patient data."
        )
        if self.mode == "documentation":
            return (
                "CONTROLLED TASK MODE: DOCUMENTATION\n"
                "Perform the requested documentation transformation; do not turn it into clinical advice. "
                "Follow the requested note format exactly and use only facts explicitly supplied by the user. "
                "Do not invent negative review-of-systems findings, examination findings, diagnoses, medication "
                "decisions, prior comparisons, follow-up intervals, monitoring instructions, or safety-net advice. "
                "Preserve an assessment or plan supplied by the user as their stated content. For a requested field "
                "that cannot be completed, write a short [Not provided] placeholder instead of guessing. Do not add "
                "generic health headings, evidence commentary, or a disclaimer.\n"
                + authorization_guard
            )
        if self.mode == "chart_lookup":
            return (
                "CONTROLLED TASK MODE: CHART LOOKUP\n"
                "The user is asking a direct factual question about data already provided below "
                "(medications, allergies, conditions, vitals, or similar record entries) -- this is "
                "not a clinical-evidence question, answer only from that data. If the specific fact "
                "asked for is not present in the data provided, say so plainly rather than guessing "
                "or inferring it. Do not use [S#] citation markers -- there are no external sources "
                "for this answer, it is a direct record lookup, not an evidence-grounded claim. You "
                "may still give brief, relevant clinical context about what the data means, but do "
                "not turn this into an evidence review or a treatment recommendation.\n"
                + authorization_guard
            )
        if self.mode == "translation":
            return (
                "CONTROLLED TASK MODE: TRANSLATION\n"
                "Translate only the current user text into the language and regional variant established by the "
                "request or recent conversation. Do not answer, research, fact-check, expand, summarise, or give "
                "medical advice about the text. Preserve meaning, distinctions between technical terms, formatting, "
                "and certainty. Return only the translation unless the user explicitly requests commentary.\n"
                + authorization_guard
            )
        if self.mode == "professional_evidence_review":
            return (
                "CONTROLLED TASK MODE: PROFESSIONAL EVIDENCE REVIEW\n"
                "Answer the evidence question directly at professional technical depth. Distinguish established "
                "guideline-supported practice from emerging evidence; cover relevant landmark evidence, "
                "indications, contraindications, medicine or procedural considerations, and follow-up where material. "
                "Do not replace the requested review with basic patient education or merely tell the user to ask a "
                "clinician. Keep recommendations within the authenticated role's permissions and supplied evidence.\n"
                + authorization_guard
            )
        return (
            "CONTROLLED TASK MODE: CLINICAL INFORMATION\n"
            "Answer the current health-information request directly and proportionately.\n"
            + authorization_guard
        )

    def retrieval_question(self, question: str) -> str:
        if self.mode != "professional_evidence_review":
            return question
        return (
            f"{question}\n\n"
            "Retrieval focus for a professional evidence review: current formal guidelines; landmark trials and "
            "systematic reviews; established versus emerging interventions; important indications, "
            "contraindications, peri-treatment considerations, and follow-up."
        )

    def completion_block(
        self, intent_category: str, vulnerable_flags: Sequence[str] | None = None
    ) -> str:
        """Return a bounded quality checklist that cannot change disposition."""

        if self.is_transformation or self.mode == "chart_lookup":
            return ""
        if self.mode == "professional_evidence_review":
            return (
                "Controlled response-completeness contract:\n"
                "- Lead with the requested evidence update, not generic patient advice.\n"
                "- Separate established guideline-supported care from emerging evidence.\n"
                "- Include material indications, contraindications, treatment interactions, and follow-up.\n"
                "- State important evidence limitations and dates without padding the answer.\n"
                "This checklist controls completeness only; it cannot change authorization or policy gates."
            )

        flags = {str(flag).lower() for flag in (vulnerable_flags or [])}
        if intent_category in {"symptom_triage", "maternity", "msk"}:
            lines = [
                "Controlled response-completeness contract:",
                "- Answer the user's disposition question directly using the already-decided risk level.",
                "- Frame possible explanations as possibilities, not a diagnosis.",
                "- Identify only missing facts that could change the disposition or next action.",
                "- Give proportionate self-care and a concrete review timeframe when supported.",
                "- Include concise, presentation-specific warning signs and what action each warrants.",
                "- Do not escalate merely to make the response look safer.",
            ]
            if intent_category == "maternity" or "postpartum" in flags:
                lines.extend(
                    [
                        "- For postpartum concerns, account for time since delivery, bleeding, fever or chills, "
                        "abnormal or foul-smelling discharge, pelvic bulge or heaviness, and urinary or bowel symptoms "
                        "when those facts would change advice.",
                        "- If pressure or pelvic-floor symptoms persist or recur, give a proportionate routine review "
                        "route such as the maternity team, primary care, or pelvic-health physiotherapy.",
                    ]
                )
            lines.append(
                "This checklist cannot override deterministic clinical decisions or policy gates."
            )
            return "\n".join(lines)

        if intent_category == "medication_query":
            return "\n".join(
                [
                    "Controlled response-completeness contract:",
                    "- Answer the medicine question directly without prescribing or inventing a patient-specific dose.",
                    "- State the exact missing inputs needed for safe dosing, such as weight, formulation, strength, age, allergies, and relevant conditions.",
                    "- When the supplied evidence supports it, give bounded general dosing education and show how label units map to the requested form; otherwise say what a pharmacist must verify.",
                    "- Include maximum frequency, duplicate-ingredient and overdose precautions when material and supported.",
                    "- Distinguish emergency warning signs from same-day review; never route breathing difficulty, a seizure, or inability to wake as routine or merely same-day care.",
                    "This checklist cannot override deterministic clinical decisions, medicine policy, or authorization gates.",
                ]
            )

        return ""


def decide_task_mode(
    question: str,
    chat_history: Sequence[Mapping[str, object]] | None,
    authenticated_role_key: str,
) -> TaskModeDecision:
    """Classify presentation mode without granting any new authority."""

    current = (question or "").strip()
    history = list(_messages(chat_history))
    user_history = [content for role, content in history if role == "user"][-12:]

    explicit_document = bool(
        _DOCUMENT_TYPE.search(current)
        and (_DOCUMENT_ACTION.search(current) or "soap" in current.lower())
    )
    if explicit_document:
        return TaskModeDecision(
            mode="documentation",
            presentation_audience="professional",
            requires_evidence_retrieval=False,
            literal_transformation=True,
            reason="explicit request to format a clinical document",
        )

    explicit_translation = bool(_TRANSLATION_ACTION.search(current))
    translation_continuation = any(
        _TRANSLATION_ACTION.search(text) for text in user_history
    )
    if explicit_translation or translation_continuation:
        return TaskModeDecision(
            mode="translation",
            presentation_audience="literal",
            requires_evidence_retrieval=False,
            literal_transformation=True,
            reason=(
                "explicit translation request"
                if explicit_translation
                else "continuation of an established translation task"
            ),
        )

    authenticated_clinician = authenticated_role_key in _CLINICAL_ROLE_KEYS
    requests_structured_compilation = bool(
        authenticated_clinician
        and _STRUCTURED_COMPILATION_ACTION.search(current)
        and _STRUCTURED_ARTIFACT_TYPE.search(current)
    )

    # Must be checked before the authenticated-clinician catch-all just below --
    # that check alone routes ANY authenticated clinician's question to
    # professional_evidence_review unconditionally, so chart_lookup could never
    # fire for a clinician if it were checked after. Two-sided and deliberately
    # conservative: both a chart-field noun AND lookup phrasing are required, and
    # any evaluative/advisory language at all disqualifies it -- a false positive
    # here (skipping retrieval on a question that needed real evidence) is worse
    # than a false negative (an unnecessary search).
    is_chart_lookup = bool(
        _CHART_FIELD.search(current)
        and _CHART_LOOKUP_ACTION.search(current)
        and not _ADVISORY_SIGNAL.search(current)
        and not _PROFESSIONAL_EVIDENCE.search(current)
    )
    if is_chart_lookup:
        return TaskModeDecision(
            mode="chart_lookup",
            presentation_audience=("professional" if authenticated_clinician else "patient"),
            requires_evidence_retrieval=False,
            requests_structured_compilation=requests_structured_compilation,
            reason="direct factual question about already-loaded record data",
        )

    professional_signals = sum(
        1 for text in [*user_history, current] if _PROFESSIONAL_EVIDENCE.search(text)
    )
    if (
        authenticated_clinician
        or professional_signals >= 2
        or (professional_signals >= 1 and _PROFESSIONAL_EVIDENCE.search(current))
    ):
        return TaskModeDecision(
            mode="professional_evidence_review",
            presentation_audience="professional",
            requires_evidence_retrieval=True,
            requests_structured_compilation=requests_structured_compilation,
            reason=(
                "authenticated clinical role"
                if authenticated_clinician
                else "conversation requests professional evidence synthesis"
            ),
        )

    return TaskModeDecision(
        presentation_audience=("professional" if authenticated_clinician else "patient"),
        requests_structured_compilation=requests_structured_compilation,
    )
