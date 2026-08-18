"""Clinical decision support: maps LLM-detected presentation hints to structured
ClinicalDecision objects containing evidence-based triage guidance.

Presentation detection is handled entirely by the LLM intent classifier, which
populates IntentClassification.presentation_hint using natural-language understanding.
No keyword lists, regex, or hardcoded symptom terms are used here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:
    from backend.intent_risk_classifier import IntentClassification
    from backend.role_router import RoleConfig


CLINICAL_LOGIC_VERSION = "2026.08.18-grounded-triage-v3"

RISK_LEVEL_RANK = {
    "routine": 1,
    "elevated": 2,
    "urgent": 3,
    "crisis": 4,
}

CLINICAL_ROLES = {"doctor", "nurse", "midwife", "physiotherapist", "healthcare_professional"}


_SEPSIS_CONFUSION_PATTERN = re.compile(
    r"\b(confus(?:ed|ion)|disorient(?:ed|ation)|altered\s+(?:mental\s+)?state|"
    r"new\s+(?:drowsiness|agitation)|hard\s+to\s+wake)\b",
    re.IGNORECASE,
)
_SEPSIS_FEVER_PATTERN = re.compile(
    r"\b(fever|febrile|high\s+temperature|temperature\s+(?:of\s+)?(?:3[89]|4\d)(?:\.\d+)?)\b",
    re.IGNORECASE,
)
_SEPSIS_URINE_PATTERN = re.compile(
    r"\b(reduced|decreased|low|little|minimal|no)\s+(?:urine|urinary)\s*(?:output)?\b|"
    r"\b(?:passing|passed|producing|produced)\s+(?:very\s+)?(?:little|no)\s+urine\b|"
    r"\b(?:not|hasn['\u2019]?t|haven['\u2019]?t)\s+(?:passed|passing|urinated|urinating)\b|"
    r"\b(?:oliguria|anuria)\b",
    re.IGNORECASE,
)
_FINDING_NEGATION_PATTERN = re.compile(
    r"\b(no|not|without|denies|denied|isn['\u2019]?t|aren['\u2019]?t|"
    r"wasn['\u2019]?t|hasn['\u2019]?t|haven['\u2019]?t)\b(?:\W+\w+){0,3}\W*$",
    re.IGNORECASE,
)


def _has_non_negated_finding(text: str, pattern: re.Pattern) -> bool:
    """Return true only when a required finding is affirmatively reported."""
    for match in pattern.finditer(text or ""):
        prefix = (text or "")[max(0, match.start() - 48):match.start()]
        if not _FINDING_NEGATION_PATTERN.search(prefix):
            return True
    return False


def _possible_sepsis_is_grounded(case_text: str) -> bool:
    """Require every finding claimed by the deterministic sepsis pathway."""
    return all(
        _has_non_negated_finding(case_text, pattern)
        for pattern in (
            _SEPSIS_CONFUSION_PATTERN,
            _SEPSIS_FEVER_PATTERN,
            _SEPSIS_URINE_PATTERN,
        )
    )


@dataclass
class GuidelineReference:
    authority: str
    title: str
    url: str = ""
    note: str = ""

    def as_dict(self) -> Dict:
        return {
            "authority": self.authority,
            "title": self.title,
            "url": self.url,
            "note": self.note,
        }


@dataclass
class RuleTrigger:
    rule_id: str
    finding: str
    severity: str
    rationale: str
    matched_terms: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {
            "rule_id": self.rule_id,
            "finding": self.finding,
            "severity": self.severity,
            "rationale": self.rationale,
            "matched_terms": list(self.matched_terms),
        }


@dataclass
class ClinicalDecision:
    decision_id: str
    pathway_id: str
    pathway_label: str
    summary: str
    urgency_level: str
    next_step: str
    minimum_risk_level: str
    escalation_reason: str
    rationale: str
    immediate_actions: List[str] = field(default_factory=list)
    monitoring_priorities: List[str] = field(default_factory=list)
    escalation_triggers: List[str] = field(default_factory=list)
    communication_points: List[str] = field(default_factory=list)
    likely_concerns: List[str] = field(default_factory=list)
    triggered_rules: List[RuleTrigger] = field(default_factory=list)
    guideline_references: List[GuidelineReference] = field(default_factory=list)
    search_terms: List[str] = field(default_factory=list)
    vulnerable_flags: List[str] = field(default_factory=list)
    deterministic_response: bool = False
    logic_version: str = CLINICAL_LOGIC_VERSION

    def as_dict(self) -> Dict:
        return {
            "decision_id": self.decision_id,
            "pathway_id": self.pathway_id,
            "pathway_label": self.pathway_label,
            "summary": self.summary,
            "urgency_level": self.urgency_level,
            "next_step": self.next_step,
            "minimum_risk_level": self.minimum_risk_level,
            "escalation_reason": self.escalation_reason,
            "rationale": self.rationale,
            "immediate_actions": list(self.immediate_actions),
            "monitoring_priorities": list(self.monitoring_priorities),
            "escalation_triggers": list(self.escalation_triggers),
            "communication_points": list(self.communication_points),
            "likely_concerns": list(self.likely_concerns),
            "triggered_rules": [item.as_dict() for item in self.triggered_rules],
            "guideline_references": [item.as_dict() for item in self.guideline_references],
            "search_terms": list(self.search_terms),
            "vulnerable_flags": list(self.vulnerable_flags),
            "deterministic_response": self.deterministic_response,
            "logic_version": self.logic_version,
        }

    def build_triage_summary(self, sources: Optional[List[Dict]] = None) -> Dict:
        return {
            "urgency_level": self.urgency_level,
            "next_step": self.next_step,
            "what_to_monitor": list(self.monitoring_priorities),
            "rationale": self.rationale,
            "pathway_label": self.pathway_label,
            "decision_summary": self.summary,
            "immediate_actions": list(self.immediate_actions),
            "escalation_triggers": list(self.escalation_triggers),
            "communication_points": list(self.communication_points),
            "rule_hits": [item.as_dict() for item in self.triggered_rules],
            "guideline_references": [
                item.as_dict()
                for item, _markers in self._validated_guideline_references(sources or [])
            ],
            "logic_version": self.logic_version,
        }

    def render_markdown(self, role_key: str, sources: Optional[List[Dict]] = None) -> str:
        validated_guidelines = self._validated_guideline_references(sources or [])
        evidence_line = (
            "Named pathway guidance was verified against the retrieved sources."
            if validated_guidelines
            else "No named pathway guideline was verified against the retrieved sources."
        )

        sections = []
        if role_key in CLINICAL_ROLES:
            sections.append(
                "## Disposition\n"
                f"**{self.urgency_level}: {self.summary}**\n\n"
                f"**Primary action:** {self.next_step}"
            )
            sections.append(self._bullet_section("## Immediate Actions", self.immediate_actions))
            sections.append(self._bullet_section("## Monitor Right Now", self.monitoring_priorities))
            sections.append(self._bullet_section("## Escalate Immediately If", self.escalation_triggers))

            reasoning_items = []
            if self.likely_concerns:
                reasoning_items.append("Key concerns: " + ", ".join(self.likely_concerns))
            for trigger in self.triggered_rules:
                reasoning_items.append(trigger.finding)
            reasoning_items.append(self.rationale)
            sections.append(self._bullet_section("## Why This Pathway Was Selected", reasoning_items))

            if self.communication_points:
                sections.append(self._bullet_section("## What To Tell The Patient Or Family", self.communication_points))
        else:
            sections.append(
                "## What To Do Now\n"
                f"**{self.summary}**\n\n"
                f"Recommended next step: **{self.next_step}**"
            )
            sections.append(self._bullet_section("## Why This Needs Attention", [self.rationale]))
            sections.append(self._bullet_section("## Get Help Urgently If", self.escalation_triggers))
            if self.communication_points:
                sections.append(self._bullet_section("## Key Advice", self.communication_points))

        guideline_lines = [evidence_line]
        for guideline, markers in validated_guidelines:
            line = f"{guideline.authority}: {guideline.title}"
            if guideline.note:
                line += f" - {guideline.note}"
            if markers:
                line += " " + "".join(markers)
            guideline_lines.append(line)
        guideline_lines.append(f"Logic version: {self.logic_version}")
        sections.append(self._bullet_section("## Guideline Basis", guideline_lines))
        return "\n\n".join(section for section in sections if section.strip())

    @staticmethod
    def _bullet_section(heading: str, items: List[str]) -> str:
        cleaned = [str(item).strip() for item in items if str(item).strip()]
        if not cleaned:
            return ""
        return heading + "\n" + "\n".join(f"- {item}" for item in cleaned)

    @staticmethod
    def _official_source_markers(sources: List[Dict]) -> str:
        markers = []
        for source in sources:
            if source.get("source_type") != "official_guidance":
                continue
            source_id = source.get("source_id")
            if source_id:
                markers.append(f"[{source_id}]")
            if len(markers) >= 3:
                break
        return "".join(markers)

    def _validated_guideline_references(
        self, sources: List[Dict]
    ) -> List[tuple[GuidelineReference, List[str]]]:
        validated: List[tuple[GuidelineReference, List[str]]] = []
        for guideline in self.guideline_references:
            matching = [
                source
                for source in sources
                if self._source_matches_guideline(source, guideline)
            ]
            if not matching:
                continue
            markers = [
                f"[{source['source_id']}]"
                for source in matching
                if source.get("source_id")
            ][:3]
            validated.append((guideline, markers))
        return validated

    @staticmethod
    def _source_matches_guideline(
        source: Dict, guideline: GuidelineReference
    ) -> bool:
        if source.get("source_type") != "official_guidance":
            return False

        def normalise_url(value: str) -> str:
            try:
                parts = urlsplit((value or "").strip().lower())
                return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))
            except ValueError:
                return (value or "").strip().lower().rstrip("/")

        expected_url = normalise_url(guideline.url)
        source_url = normalise_url(str(source.get("url", "")))
        if expected_url and source_url and (
            source_url == expected_url
            or source_url.startswith(expected_url + "/")
        ):
            return True

        source_text = " ".join(
            str(source.get(key, ""))
            for key in ("provider", "title", "snippet", "detail_snippet")
        ).lower()
        authority = guideline.authority.lower()
        title_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", guideline.title.lower())
            if len(token) >= 4
        }
        source_tokens = set(re.findall(r"[a-z0-9]+", source_text))
        distinctive = {
            token
            for token in title_tokens
            if re.fullmatch(r"(?:cg|ng|qs)\d+", token)
        }
        if distinctive and distinctive & source_tokens:
            return True
        overlap = len(title_tokens & source_tokens)
        return authority in source_text and overlap >= min(4, max(2, len(title_tokens) // 2))


class ClinicalDecisionSupportEngine:
    """
    Maps the LLM-detected presentation_hint from IntentClassification to a
    structured ClinicalDecision with evidence-based triage guidance.

    No text parsing, no regex, no keyword lists. The LLM intent classifier
    identifies the clinical presentation pattern using natural-language
    understanding; this engine provides the corresponding clinical response.
    """

    def assess(
        self,
        question: str,
        intent: "IntentClassification",
        role_config: "RoleConfig",
        case_text: Optional[str] = None,
    ) -> ClinicalDecision:
        presentation = getattr(intent, "presentation_hint", "none") or "none"
        if presentation == "possible_sepsis" and not _possible_sepsis_is_grounded(
            case_text or question
        ):
            # A presentation hint is only a routing suggestion. Never let it
            # manufacture the findings quoted by a deterministic response.
            intent.presentation_hint = "none"
            intent.crisis_detected = False
            if intent.risk_level == "crisis":
                intent.risk_level = "urgent"
            intent.escalation_required = intent.risk_level == "urgent"
            if intent.risk_level == "urgent":
                intent.escalation_reason = (
                    "The reported acute symptoms warrant prompt clinical assessment."
                )
            presentation = "none"
        dispatch = {
            "thunderclap_headache":      self._decision_thunderclap_headache,
            "possible_sepsis":           self._decision_possible_sepsis,
            "recurrent_blackout":        self._decision_recurrent_blackout,
            "chronic_cough_red_flags":   self._decision_chronic_cough_red_flags,
            "chronic_cough_no_red_flags": self._decision_chronic_cough_no_red_flags,
        }
        builder = dispatch.get(presentation)
        if builder:
            return builder(intent)
        return self._build_default_decision(intent, role_config)

    def apply_to_intent(
        self,
        intent: "IntentClassification",
        decision: ClinicalDecision,
    ) -> "IntentClassification":
        if RISK_LEVEL_RANK.get(decision.minimum_risk_level, 1) > RISK_LEVEL_RANK.get(intent.risk_level, 1):
            intent.risk_level = decision.minimum_risk_level
            intent.escalation_required = decision.minimum_risk_level in {"urgent", "crisis"}
            intent.escalation_reason = decision.escalation_reason
        high_acuity_pathways = {
            "thunderclap_headache", "possible_sepsis",
            "recurrent_blackout", "chronic_cough_red_flags", "chronic_cough_no_red_flags",
        }
        if decision.pathway_id in high_acuity_pathways:
            intent.pathway_hint = "general_triage"
            if intent.intent_category == "general_info":
                intent.intent_category = "symptom_triage"
        for flag in decision.vulnerable_flags:
            if flag not in intent.vulnerable_flags:
                intent.vulnerable_flags.append(flag)
        if decision.minimum_risk_level == "crisis":
            intent.crisis_detected = True
        return intent

    # ── Clinical decision builders ─────────────────────────────────────────────

    def _decision_thunderclap_headache(self, intent: "IntentClassification") -> ClinicalDecision:
        return ClinicalDecision(
            decision_id="general-triage-thunderclap-headache",
            pathway_id="thunderclap_headache",
            pathway_label="Sudden severe headache / acute neurological emergency",
            summary="Treat as a thunderclap headache requiring emergency assessment now.",
            urgency_level="Emergency",
            next_step="Immediate review",
            minimum_risk_level="urgent",
            escalation_reason="Sudden severe headache / possible intracranial emergency.",
            rationale=(
                "Sudden onset severe headache described as the worst ever should be managed as a "
                "red-flag neurological presentation until serious intracranial causes are excluded."
            ),
            immediate_actions=[
                "Escalate now for emergency department or acute medical review; do not manage as routine headache care.",
                "Record exact onset time and perform NEWS2 plus full neurological observations immediately.",
                "Keep the patient under direct observation and follow the local acute headache or suspected subarachnoid haemorrhage pathway.",
            ],
            monitoring_priorities=[
                "Level of consciousness, new confusion, speech change, or focal neurological deficit",
                "Blood pressure, pulse, respiratory rate, oxygen saturation, and temperature",
                "Vomiting, neck stiffness, photophobia, seizure activity, or worsening pain",
            ],
            escalation_triggers=[
                "Any reduced consciousness, collapse, seizure, or new focal neurology",
                "Visual disturbance, persistent vomiting, or rapidly worsening headache",
                "Signs of meningism or any deterioration in NEWS2",
            ],
            communication_points=[
                "This pattern needs emergency assessment because a sudden severe headache can signal a bleed or another acute neurological event.",
                "Please tell us immediately if there is any confusion, weakness, speech change, or visual change while help is being arranged.",
            ],
            likely_concerns=["subarachnoid haemorrhage", "other acute intracranial pathology"],
            triggered_rules=[
                RuleTrigger(
                    rule_id="thunderclap_headache_red_flag",
                    finding="Sudden severe headache -- worst-ever description identified by clinical classifier",
                    severity="urgent",
                    rationale="Red-flag headache pattern requires emergency exclusion of intracranial causes.",
                )
            ],
            guideline_references=[
                GuidelineReference(
                    authority="NHS",
                    title="Migraine - immediate action required for sudden extremely painful headache",
                    url="https://www.nhs.uk/conditions/migraine/",
                    note="NHS advises 999 for sudden extremely painful headache and associated neurological features.",
                ),
                GuidelineReference(
                    authority="NHS",
                    title="Brain aneurysm - thunderclap headache emergency advice",
                    url="https://www.nhs.uk/conditions/brain-aneurysm/",
                    note="Sudden severe thunderclap headache is treated as an emergency presentation.",
                ),
            ],
            search_terms=[
                "sudden severe headache thunderclap headache NHS NICE",
                "subarachnoid haemorrhage sudden severe headache guideline",
            ],
            deterministic_response=True,
        )

    def _decision_possible_sepsis(self, intent: "IntentClassification") -> ClinicalDecision:
        elderly = "elderly" in (intent.vulnerable_flags or [])
        return ClinicalDecision(
            decision_id="general-triage-possible-sepsis",
            pathway_id="possible_sepsis",
            pathway_label="Possible sepsis / acute delirium pathway",
            summary="Treat as possible sepsis with acute organ dysfunction and arrange immediate clinician review.",
            urgency_level="Emergency",
            next_step="Immediate review",
            minimum_risk_level="urgent",
            escalation_reason="Confusion, fever, and reduced urine output suggest possible sepsis.",
            rationale=(
                "Confusion plus fever plus reduced urine output indicates possible sepsis or another "
                "acute cause of organ dysfunction -- needs immediate escalation rather than watchful waiting."
            ),
            immediate_actions=[
                "Escalate immediately to the acute medical or senior review pathway and treat as possible sepsis until assessed.",
                "Calculate NEWS2 now and repeat observations at a frequency dictated by deterioration risk.",
                "Start the local sepsis bundle if criteria are met: cultures, bloods, fluids, oxygen, and antibiotics as directed.",
                "Strictly monitor urine output and seek the likely source of infection while escalation is underway.",
            ],
            monitoring_priorities=[
                "Mental status, respiratory rate, blood pressure, pulse, oxygen saturation, and temperature",
                "Urine output, fluid balance, and signs of acute kidney injury",
                "Rapid change in NEWS2, lactate if available, and signs of shock or poor perfusion",
            ],
            escalation_triggers=[
                "Hypotension, tachypnoea, hypoxia, mottled skin, or rising NEWS2",
                "Worsening confusion, reduced responsiveness, or new agitation",
                "No urine output, persistent vomiting, or evidence of a new infection source",
            ],
            communication_points=[
                "The combination of confusion, fever, and reduced urine output can indicate sepsis, so we are escalating immediately.",
                "We need to monitor observations closely and may need urgent blood tests, fluids, oxygen, and antibiotics.",
            ],
            likely_concerns=["sepsis", "acute delirium", "acute kidney injury"],
            triggered_rules=[
                RuleTrigger(
                    rule_id="sepsis_red_flags",
                    finding="Confusion with fever and reduced urine output -- identified by clinical classifier",
                    severity="urgent",
                    rationale="This combination suggests infection with possible organ dysfunction.",
                )
            ],
            guideline_references=[
                GuidelineReference(
                    authority="NICE",
                    title="NG253 Suspected sepsis in people aged 16 or over",
                    url="https://www.nice.org.uk/guidance/ng253",
                    note="Supports urgent assessment and treatment for suspected sepsis in adults.",
                ),
                GuidelineReference(
                    authority="NHS",
                    title="Symptoms of sepsis",
                    url="https://www.nhs.uk/conditions/sepsis/",
                    note="Confusion and passing no urine are highlighted adult red flags.",
                ),
                GuidelineReference(
                    authority="NHS England",
                    title="National Early Warning Score (NEWS)",
                    url="https://www.england.nhs.uk/ourwork/clinical-policy/sepsis/nationalearningscore/",
                    note="NEWS2 is endorsed to identify deterioration and support early escalation.",
                ),
            ],
            search_terms=[
                "suspected sepsis adults confusion reduced urine output NICE NG253",
                "sepsis confusion oliguria NEWS2 NHS",
            ],
            vulnerable_flags=["elderly"] if elderly else [],
            deterministic_response=True,
        )

    def _decision_recurrent_blackout(self, intent: "IntentClassification") -> ClinicalDecision:
        return ClinicalDecision(
            decision_id="general-triage-recurrent-blackout",
            pathway_id="recurrent_blackout",
            pathway_label="Recurrent blackout / presyncope pathway",
            summary="Arrange same-day medical assessment for recurrent blackout or presyncope until cardiac and neurological causes are excluded.",
            urgency_level="Urgent",
            next_step="Same-day review",
            minimum_risk_level="urgent",
            escalation_reason="Recurrent blackouts or near-syncope require urgent assessment.",
            rationale=(
                "Recurrent episodes of transient visual blackout or near-collapse need urgent assessment "
                "for arrhythmia, syncope, orthostatic hypotension, or neurological causes."
            ),
            immediate_actions=[
                "Keep the patient safe from falls and do not dismiss the episodes as benign without assessment.",
                "Obtain lying and standing blood pressure, pulse, capillary glucose, and a 12-lead ECG if available.",
                "Arrange same-day medical assessment or urgent referral in line with the local blackout or syncope pathway.",
            ],
            monitoring_priorities=[
                "Further blackout, collapse, chest pain, palpitations, or exertional symptoms",
                "Heart rate and rhythm, blood pressure including postural change, and injury risk",
                "Any focal neurological symptoms, persistent visual loss, or prolonged recovery",
            ],
            escalation_triggers=[
                "Loss of consciousness, injury, chest pain, palpitations, or abnormal ECG",
                "Any focal neurological deficit, persistent visual loss, or prolonged confusion",
                "Episode during exertion or without warning, especially with breathlessness",
            ],
            communication_points=[
                "Repeated blackouts or near-blackouts need urgent assessment because they can be caused by heart rhythm or blood pressure problems.",
                "Please report any chest pain, palpitations, weakness, speech change, or a full loss of consciousness immediately.",
            ],
            likely_concerns=["syncope or presyncope", "arrhythmia", "orthostatic hypotension", "neurological event"],
            triggered_rules=[
                RuleTrigger(
                    rule_id="recurrent_blackout_red_flag",
                    finding="Recurrent transient blackout or near-collapse episodes -- identified by clinical classifier",
                    severity="urgent",
                    rationale="Recurrent blackout presentations need structured syncope assessment and cardiac screening.",
                )
            ],
            guideline_references=[
                GuidelineReference(
                    authority="NICE",
                    title="CG109 Transient loss of consciousness ('blackouts') in over 16s",
                    url="https://www.nice.org.uk/guidance/cg109",
                    note="NICE recommends structured initial assessment and 12-lead ECG for blackout presentations.",
                ),
            ],
            search_terms=[
                "blackouts over 16s NICE CG109 12 lead ECG",
                "recurrent blackout near syncope same day assessment guideline",
            ],
            deterministic_response=True,
        )

    def _decision_chronic_cough_red_flags(self, intent: "IntentClassification") -> ClinicalDecision:
        return ClinicalDecision(
            decision_id="general-triage-chronic-cough-red-flags",
            pathway_id="chronic_cough_red_flags",
            pathway_label="Chronic cough with red-flag features",
            summary="Arrange urgent clinician review for persistent cough with red-flag features.",
            urgency_level="Urgent",
            next_step="Same-day review",
            minimum_risk_level="urgent",
            escalation_reason="Persistent cough with red-flag features requires urgent assessment.",
            rationale="Chronic cough plus red-flag features needs urgent assessment rather than routine follow-up.",
            immediate_actions=[
                "Arrange same-day GP or appropriate clinician review -- do not defer to routine appointment.",
                "Take a focused history for haemoptysis, weight loss, night sweats, and systemic features.",
                "Escalate faster if examination suggests respiratory compromise or new acute features.",
            ],
            monitoring_priorities=[
                "Breathlessness, chest pain, haemoptysis, fever, or systemic symptoms",
                "Any change in cough pattern, sputum, wheeze, or exercise tolerance",
                "Weight loss, voice change, or other new red-flag features",
            ],
            escalation_triggers=[
                "Coughing up blood, chest pain, breathlessness, or rapidly worsening symptoms",
                "Unexplained weight loss, night sweats, or persistent fever",
                "Oxygen desaturation, respiratory distress, or inability to maintain oral intake",
            ],
            communication_points=[
                "A persistent cough with these features needs prompt assessment to look for serious causes.",
                "Please seek help sooner if breathlessness, chest pain, or coughing blood worsens.",
            ],
            likely_concerns=["lung malignancy", "tuberculosis", "other serious respiratory pathology"],
            triggered_rules=[
                RuleTrigger(
                    rule_id="chronic_cough_red_flags",
                    finding="Chronic cough with red-flag features -- identified by clinical classifier",
                    severity="urgent",
                    rationale="Red-flag features alongside chronic cough require urgent exclusion of serious pathology.",
                )
            ],
            guideline_references=[
                GuidelineReference(
                    authority="NHS",
                    title="Cough -- when to see a GP and when urgent review is needed",
                    url="https://www.nhs.uk/symptoms/cough/",
                    note="NHS advises urgent escalation for haemoptysis, breathlessness, chest pain, and unexplained weight loss.",
                ),
            ],
            search_terms=[
                "persistent cough red flags haemoptysis weight loss NHS guideline",
                "chronic cough urgent review adult primary care",
            ],
            deterministic_response=True,
        )

    def _decision_chronic_cough_no_red_flags(self, intent: "IntentClassification") -> ClinicalDecision:
        return ClinicalDecision(
            decision_id="general-triage-chronic-cough-no-red-flags",
            pathway_id="chronic_cough_no_red_flags",
            pathway_label="Chronic cough assessment pathway",
            summary="Arrange GP review for persistent cough -- prompt but not emergency.",
            urgency_level="Prompt",
            next_step="GP",
            minimum_risk_level="elevated",
            escalation_reason="Persistent cough lasting 8+ weeks requires structured outpatient review.",
            rationale=(
                "A cough lasting 8 weeks meets the threshold for chronic cough assessment. "
                "Without red-flag features, this warrants a prompt outpatient workup rather than emergency care."
            ),
            immediate_actions=[
                "Arrange GP review for chronic cough assessment -- do not continue self-care alone.",
                "Take a focused history for asthma, rhinitis, reflux, ACE inhibitor use, and infection exposure.",
                "Escalate faster if new red flags appear or examination suggests respiratory compromise.",
            ],
            monitoring_priorities=[
                "Breathlessness, chest pain, haemoptysis, fever, or systemic symptoms",
                "Any change in cough pattern, sputum, wheeze, or exercise tolerance",
                "Weight loss, voice change, or other new red-flag features",
            ],
            escalation_triggers=[
                "Coughing up blood, chest pain, breathlessness, or rapidly worsening symptoms",
                "Unexplained weight loss, night sweats, or persistent fever",
                "Oxygen desaturation, respiratory distress, or inability to maintain oral intake",
            ],
            communication_points=[
                "A cough lasting 8 weeks needs clinician review to look for common causes such as asthma, reflux, or upper airway issues.",
                "Please seek help sooner if you develop breathlessness, chest pain, coughing blood, weight loss, or fevers.",
            ],
            likely_concerns=["upper airway cough syndrome", "asthma", "gastro-oesophageal reflux"],
            triggered_rules=[
                RuleTrigger(
                    rule_id="chronic_cough_threshold",
                    finding="Cough duration at or beyond 8 weeks -- identified by clinical classifier",
                    severity="elevated",
                    rationale="Persistent cough needs structured assessment rather than generic reassurance.",
                )
            ],
            guideline_references=[
                GuidelineReference(
                    authority="NHS",
                    title="Cough -- when to see a GP and when urgent review is needed",
                    url="https://www.nhs.uk/symptoms/cough/",
                    note="NHS advises GP review for persistent cough.",
                ),
            ],
            search_terms=[
                "persistent cough more than 8 weeks NHS guideline",
                "chronic cough adult primary care assessment NHS",
            ],
            deterministic_response=True,
        )

    def _build_default_decision(
        self,
        intent: "IntentClassification",
        role_config: "RoleConfig",
    ) -> ClinicalDecision:
        risk_level = (intent.risk_level or "routine").lower()
        if risk_level == "crisis":
            urgency_level = "Emergency"
            next_step = "Local emergency services"
        elif risk_level == "urgent":
            urgency_level = "Urgent"
            next_step = "Local urgent-care service" if role_config.role_key not in CLINICAL_ROLES else "Immediate review"
        elif risk_level == "elevated":
            urgency_level = "Prompt"
            next_step = "Primary-care clinician"
        else:
            urgency_level = "Routine"
            next_step = "Self-care"

        rationale = intent.escalation_reason or "No specific high-acuity presentation pattern identified."
        actions = {
            "crisis":   ["Activate the emergency response pathway immediately.",
                         "Do not wait for routine assessment or reassurance."],
            "urgent":   ["Arrange urgent clinician review and reassess observations promptly.",
                         "Escalate immediately if new red flags appear while awaiting assessment."],
            "elevated": ["Arrange clinician review and safety-net for deterioration."],
            "routine":  ["Provide self-care guidance with clear safety-netting."],
        }
        monitors = {
            "crisis":   ["Any deterioration while emergency help is being arranged"],
            "urgent":   ["Worsening symptoms, new red flags, or rising physiological concern"],
            "elevated": ["Persistence, progression, or new red-flag symptoms"],
            "routine":  ["Whether symptoms settle, worsen, or new warning signs appear"],
        }

        return ClinicalDecision(
            decision_id="general-triage-default",
            pathway_id="general_triage",
            pathway_label="General triage",
            summary="No specific high-acuity presentation matched; using computed acuity floor and clinical judgement.",
            urgency_level=urgency_level,
            next_step=next_step,
            minimum_risk_level=risk_level,
            escalation_reason=intent.escalation_reason or "",
            rationale=rationale,
            immediate_actions=actions.get(risk_level, actions["routine"]),
            monitoring_priorities=monitors.get(risk_level, monitors["routine"]),
            escalation_triggers=["Any new severe symptom, collapse, or rapid deterioration"],
            communication_points=[],
            likely_concerns=[],
            triggered_rules=[],
            guideline_references=[],
            search_terms=[],
            deterministic_response=False,
        )
