"""
Intent and risk classification for incoming clinical questions.
Combines a fast regex pre-screen with an LLM-based structured classifier.
"""
from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class IntentClassification:
    intent_category: str = "general_info"
    # symptom_triage | medication_query | chronic_condition | maternity |
    # msk | mental_health | general_info | crisis | administrative
    risk_level: str = "routine"
    # routine | elevated | urgent | crisis
    vulnerable_flags: List[str] = field(default_factory=list)
    # pregnancy | paediatric | elderly | renal_impairment | immunocompromised
    escalation_required: bool = False
    escalation_reason: str = ""
    crisis_detected: bool = False
    pathway_hint: str = "general_triage"
    # general_triage | maternity | msk | medications | chronic_conditions
    confidence: float = 0.8
    presentation_hint: str = "none"
    # none | thunderclap_headache | possible_sepsis | recurrent_blackout
    # | chronic_cough_red_flags | chronic_cough_no_red_flags
    ambiguous_term_detected: bool = False
    ambiguous_term: str = ""
    ambiguity_clarifying_question: str = ""
    ambiguity_reply_options: List[Dict[str, str]] = field(default_factory=list)
    # [{"display": short chip label, "prompt": full self-contained disambiguated question}, ...]
    clarification_required: bool = False
    clarification_reason: str = ""
    clarifying_questions: List[str] = field(default_factory=list)
    # Up to 3 decision-specific questions required before a useful
    # personalized answer can be given.


# ── Fast regex crisis patterns ─────────────────────────────────────────────────
_CRISIS_PATTERNS: List[re.Pattern] = [
    # Cardiac / respiratory arrest
    re.compile(
        r"(chest\s*pain.{0,30}(shortness?|difficult|breath)|"
        r"not\s*breath|stopped\s*breath|cardiac\s*arrest|heart\s*attack)",
        re.IGNORECASE,
    ),
    # Stroke
    re.compile(
        r"(face\s*drooping|arm\s*weak|slurred?\s*speech|sudden\s*(vision|headache|weakness)|"
        r"fast\s*test|FAST\s*test|stroke\s*symptoms?)",
        re.IGNORECASE,
    ),
    # Anaphylaxis
    re.compile(
        r"(anaphyla|severe\s*allergic|throat\s*(closing|swelling)|"
        r"epipen|epinephrine\s*now|can\s*t\s*breathe)",
        re.IGNORECASE,
    ),
    # Obstetric emergencies
    re.compile(
        r"(heavy\s*bleed.{0,20}pregnan|eclampsia|cord\s*prolapse|"
        r"placental?\s*abruption|baby\s*not\s*moving.{0,10}hours?)",
        re.IGNORECASE,
    ),
    # Major trauma / overdose
    re.compile(
        r"(overdosed?|taken\s*too\s*many\s*(pills|tablets)|"
        r"unconscious|unresponsive|not\s*waking|collaps(?:e|ed|ing))",
        re.IGNORECASE,
    ),
    # Meningitis
    re.compile(
        r"(meningitis|non.?blanching\s*rash|glass\s*test|"
        r"stiff\s*neck.{0,20}(fever|rash))",
        re.IGNORECASE,
    ),
]

_CLINICAL_EDUCATION_PATTERN = re.compile(
    r"\b(guidelines?|updates?|evidence|research|literature|protocol|policy|teaching|"
    r"training|simulation|quality improvement|audit|review article|recommended dosing|"
    r"bls|acls|als|approach(?:es)? to|walk me through)\b",
    re.IGNORECASE,
)
_GENERAL_EDUCATION_PATTERN = re.compile(
    r"\b(what (?:are|is)|how does|explain|learn about|information about|difference between)\b",
    re.IGNORECASE,
)
_ACTIVE_EMERGENCY_PATTERN = re.compile(
    r"\b(right now|currently (?:happening|resuscitating)|in front of me|just collapsed|"
    r"code (?:is )?in progress|patient is (?:unresponsive|not breathing|in cardiac arrest)|"
    r"we are (?:resuscitating|doing cpr)|i am (?:having|experiencing)|can't breathe now)\b",
    re.IGNORECASE,
)
_EXPLICIT_STABILITY_PATTERN = re.compile(
    r"\b(?:remains?|is|was)\s+(?:clinically\s+|haemodynamically\s+|hemodynamically\s+)?stable\b",
    re.IGNORECASE,
)
_NEGATED_CARDIORESPIRATORY_PATTERN = re.compile(
    r"\b(?:denies?|denied|no|without)\b.{0,35}\b(?:chest\s+pain|shortness\s+of\s+breath|"
    r"difficulty\s+breathing|breathlessness)\b",
    re.IGNORECASE,
)
_CLINICAL_ROLES = {"doctor", "nurse", "midwife", "physiotherapist", "healthcare_professional"}
_PERSONAL_PNEUMONIA_TREATMENT_PATTERN = re.compile(
    r"\b(i (?:have|was diagnosed with)|i've got|estou com|tenho|tengo|"
    r"me diagnosticaron|j['’]ai|diagnosed? with)\b.{0,80}"
    r"\b(pneumonia|pneumonie|neumon[ií]a)\b.{0,120}\bantibi[oó]tic",
    re.IGNORECASE,
)

# ── Intent → pathway mapping ───────────────────────────────────────────────────
_INTENT_TO_PATHWAY: dict[str, str] = {
    "symptom_triage": "general_triage",
    "medication_query": "medications",
    "chronic_condition": "chronic_conditions",
    "maternity": "maternity",
    "msk": "msk",
    "mental_health": "general_triage",
    "general_info": "general_triage",
    "crisis": "general_triage",
    "administrative": "general_triage",
}


class IntentRiskClassifier:
    """
    Two-stage classifier:
    1. Fast regex crisis pre-screen (no LLM latency)
    2. LLM structured classification for intent + risk
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        import openai
        self.client = openai.OpenAI(api_key=api_key)

    def classify(
        self,
        question: str,
        user_profile: Optional[dict] = None,
        role_key: str = "patient",
        patient_history=None,
        recent_turns: Optional[List[dict]] = None,
        conversation_summary: str = "",
    ) -> IntentClassification:
        """
        Full classification pipeline. Run this concurrently with query expansion
        inside the orchestrator to minimise latency.
        """
        # Stage 1: Instant crisis check
        if self._crisis_prescreen(question, role_key=role_key):
            return IntentClassification(
                intent_category="crisis",
                risk_level="crisis",
                escalation_required=True,
                escalation_reason="Potential emergency symptoms detected -- please seek immediate help.",
                crisis_detected=True,
                pathway_hint="general_triage",
                confidence=0.95,
            )

        # A personal request to choose antibiotics for stated pneumonia needs
        # prompt in-person severity assessment; it is neither routine dosing
        # advice nor an automatic emergency. Keep clinician education out of
        # this rule so professional guideline questions still reach retrieval.
        if self._acute_treatment_prescreen(question, role_key=role_key):
            return IntentClassification(
                intent_category="medication_query",
                risk_level="urgent",
                escalation_required=True,
                escalation_reason="Pneumonia treatment requires prompt clinical assessment.",
                pathway_hint="medications",
                confidence=0.95,
            )

        # Stage 2: LLM classification
        try:
            result = self._llm_classify(
                question,
                user_profile or {},
                role_key,
                patient_history,
                recent_turns,
                conversation_summary,
            )
            return self._calibrate_llm_crisis_label(question, role_key, result)
        except Exception as exc:
            print(f"IntentRiskClassifier LLM call failed, using safe defaults: {exc}")
            return self._safe_default()

    def _crisis_prescreen(self, question: str, role_key: str = "patient") -> bool:
        """Detect an active emergency without treating topic mentions as events.

        Clinical education and guideline questions must proceed to retrieval.
        An explicit active emergency still wins for every role.
        """
        text = (question or "").strip()
        if not any(pattern.search(text) for pattern in _CRISIS_PATTERNS):
            return False
        if _ACTIVE_EMERGENCY_PATTERN.search(text):
            return True
        if (
            role_key in _CLINICAL_ROLES
            and _EXPLICIT_STABILITY_PATTERN.search(text)
            and _NEGATED_CARDIORESPIRATORY_PATTERN.search(text)
        ):
            return False

        educational = bool(_GENERAL_EDUCATION_PATTERN.search(text))
        if role_key in _CLINICAL_ROLES:
            educational = educational or bool(_CLINICAL_EDUCATION_PATTERN.search(text))
        return not educational

    def _calibrate_llm_crisis_label(
        self,
        question: str,
        role_key: str,
        result: IntentClassification,
    ) -> IntentClassification:
        """Require case-level evidence before accepting an LLM crisis label.

        A clinician may quote red-flag terms inside a stable ward note. The
        model must not turn that into an active emergency when the note itself
        explicitly says the patient is stable and gives no current emergency
        evidence. This guard only downgrades to urgent review; an explicit
        active-emergency phrase always wins.
        """
        text = (question or "").strip()
        if (
            role_key not in _CLINICAL_ROLES
            or result.risk_level != "crisis"
            or not _EXPLICIT_STABILITY_PATTERN.search(text)
            or not _NEGATED_CARDIORESPIRATORY_PATTERN.search(text)
            or _ACTIVE_EMERGENCY_PATTERN.search(text)
        ):
            return result

        result.intent_category = "symptom_triage"
        result.risk_level = "urgent"
        result.crisis_detected = False
        result.escalation_required = True
        result.escalation_reason = (
            "The reported change needs prompt in-person assessment, but the supplied "
            "status does not establish an active emergency."
        )
        result.confidence = min(result.confidence, 0.8)
        return result

    def _acute_treatment_prescreen(self, question: str, role_key: str = "patient") -> bool:
        text = (question or "").strip()
        if role_key in _CLINICAL_ROLES and _CLINICAL_EDUCATION_PATTERN.search(text):
            return False
        return bool(_PERSONAL_PNEUMONIA_TREATMENT_PATTERN.search(text))

    def _llm_classify(
        self,
        question: str,
        user_profile: dict,
        role_key: str,
        patient_history=None,
        recent_turns: Optional[List[dict]] = None,
        conversation_summary: str = "",
    ) -> IntentClassification:
        role_hint = f"The user's clinical role is: {role_key}." if role_key else ""
        pregnancy_hint = ""
        if "pregnan" in question.lower() or role_key == "midwife":
            pregnancy_hint = " Note: pregnancy context may be present -- apply maternity flags carefully."

        history_block = ""
        if patient_history is not None:
            history_text = patient_history.as_prompt_block()
            if history_text:
                history_block = (
                    "\n\nPatient's known medical history:\n"
                    + history_text
                    + "\n\nUse a history item only when it has a validated connection to the current "
                    "presentation. A diagnosis, medicine, vulnerability label, or prior escalation alone "
                    "must not raise urgency without compatible current facts. Ignore unrelated history."
                )

        continuation_block = ""
        if recent_turns:
            turns = [
                m for m in recent_turns
                if m.get("role") in ("user", "assistant") and m.get("content", "").strip()
            ][-5:]
            if turns:
                rendered = "\n".join(f"{m['role'].title()}: {m['content'].strip()}" for m in turns)
                continuation_block = (
                    "\n\nRecent conversation (most recent last):\n"
                    + rendered
                    + "\n\nIf the last assistant turn already asked a clarifying question about an "
                    "ambiguous term, and the patient's current message answers it, resolve the "
                    "ambiguity using that reply -- set ambiguous_term_detected to false (it is "
                    "already resolved) and classify normally using both messages together as the "
                    "intended question. Treat details supplied in recent turns as known and never "
                    "ask for the same information again. An assistant's earlier diagnosis, treatment "
                    "indication, or patient-history statement is not a confirmed fact unless the user "
                    "subsequently confirms it or it appears in the structured patient history."
                )

        summary_block = ""
        if conversation_summary and conversation_summary != "No earlier conversation.":
            summary_block = (
                "\n\nSummary of the whole earlier conversation (role-labelled; prior assistant "
                "statements are not confirmed patient facts):\n"
                + conversation_summary
            )

        prompt = (
            "You are a clinical intent classifier for a health information system.\n"
            f"{role_hint}{pregnancy_hint}{history_block}{summary_block}{continuation_block}\n\n"
            "First distinguish an active patient event from professional education, guideline review, "
            "research, teaching, audit, or hypothetical discussion. Emergency terminology in an educational "
            "request is not evidence that an emergency is occurring. If a clinician describes an active patient, "
            "classify the patient's acuity; otherwise keep educational requests routine. Apply this distinction "
            "in any language.\n\n"
            "Classify the following health question and return a JSON object with these exact keys:\n"
            "- intent_category: one of [symptom_triage, medication_query, chronic_condition, "
            "maternity, msk, mental_health, general_info, crisis, administrative]\n"
            "- risk_level: one of [routine, elevated, urgent, crisis]\n"
            "- vulnerable_flags: array of applicable flags from "
            "[pregnancy, paediatric, elderly, renal_impairment, immunocompromised, postpartum, newborn]\n"
            "- escalation_required: boolean -- true if the question suggests urgent clinical need\n"
            "- escalation_reason: short string (≤60 chars) explaining why escalation is needed, "
            "or empty string if not required\n"
            "- pathway_hint: one of [general_triage, maternity, msk, medications, chronic_conditions]\n"
            "- confidence: float 0.0–1.0\n"
            "- presentation_hint: one of the following -- set ONLY if the description clearly matches, "
            "otherwise use 'none':\n"
            "  'thunderclap_headache' -- sudden-onset severe headache described as the worst ever, "
            "coming on in seconds; the patient does not need to use those exact words.\n"
            "  'possible_sepsis' -- ALL THREE present: altered mental state/confusion AND "
            "fever/high temperature AND reduced urine output.\n"
            "  'recurrent_blackout' -- multiple episodes of transient loss of consciousness, "
            "near-fainting, or vision going black; must be recurrent (more than once).\n"
            "  'chronic_cough_red_flags' -- cough lasting 8+ weeks WITH any of: coughing blood, "
            "unexplained weight loss, or drenching night sweats.\n"
            "  'chronic_cough_no_red_flags' -- cough lasting 8+ weeks WITHOUT those red flags.\n"
            "  'none' -- none of the above clearly apply.\n"
            "- ambiguous_term_detected: boolean -- true ONLY if the question uses a specific "
            "clinical term/measurement/lab/symptom name that has genuinely different meanings "
            "across medical specialties, where the different meanings would lead to MATERIALLY "
            "DIFFERENT clinical guidance, AND the patient's known history above does not already "
            "make clear which meaning applies. Example: 'peak flow' alone could be peak "
            "EXPIRATORY flow (respiratory/asthma, L/min) or peak URINARY flow rate / Qmax "
            "(urology, mL/s) -- if the patient's history doesn't already show which one, and the "
            "question doesn't say, flag this. Do NOT flag ordinary vague wording -- only genuine "
            "cross-specialty ambiguity. Default to false; most questions are not ambiguous in "
            "this sense. NEVER set this true if risk_level is urgent or crisis -- resolve as best "
            "judgement and proceed instead of delaying care with a question.\n"
            "- ambiguous_term: the specific ambiguous term from the question, or empty string.\n"
            "- ambiguity_clarifying_question: if ambiguous_term_detected is true, ONE short, "
            "natural, role-appropriate question asking the patient which meaning applies. Empty "
            "string otherwise.\n"
            "- ambiguity_reply_options: if ambiguous_term_detected is true, an array of 2-3 "
            "objects {\"display\": short label, \"prompt\": a FULL, SELF-CONTAINED restatement of "
            "the patient's original question with the ambiguity resolved} -- each prompt must "
            "stand alone as a complete question. Example for 'What is my peak flow level and what "
            "does it mean?': [{\"display\": \"It was a breathing test\", \"prompt\": \"My peak "
            "flow was measured with a breathing/asthma peak flow meter -- what does my reading "
            "mean?\"}, {\"display\": \"It was a urine flow test\", \"prompt\": \"My peak flow was "
            "measured during a urology urine flow test (uroflowmetry) -- what does my reading "
            "mean?\"}]. Empty array otherwise.\n"
            "- clarification_required: boolean -- true only when this is a PERSONALIZED health "
            "or active-patient decision and missing facts are essential to give a useful, specific "
            "answer. Examples include a patient asking whether a prescription is 'for me' without "
            "saying what it was prescribed to treat, a clinician asking for patient-specific treatment "
            "without the indication or a decision-changing contraindication, or a non-urgent symptom "
            "description that is too incomplete to assess. False for definitions, general education, "
            "record lookups, administrative requests, and questions that can be answered directly. "
            "Never use this for urgent or crisis presentations and never delay an immediate safety action.\n"
            "- clarification_reason: when clarification_required is true, a concrete phrase naming "
            "the missing decision, such as 'the prescription indication is unknown'; empty otherwise.\n"
            "- clarifying_questions: when clarification_required is true, an array of 1-3 concise, "
            "role-appropriate questions. Ask only for facts that could change the answer. Use the known "
            "patient history and recent conversation; do not ask for anything already recorded or answered. "
            "For a medication prescribed to a patient, normally establish the indication, whether any "
            "doses were taken and what happened, and whether the prescriber knew about a relevant recorded "
            "allergy. For clinicians, ask in concise clinical terms. Empty otherwise.\n\n"
            f"Question: {question}\n\n"
            "Return only valid JSON, no other text."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)

        valid_presentations = {
            "none", "thunderclap_headache", "possible_sepsis", "recurrent_blackout",
            "chronic_cough_red_flags", "chronic_cough_no_red_flags",
        }
        raw_presentation = data.get("presentation_hint", "none")
        presentation_hint = raw_presentation if raw_presentation in valid_presentations else "none"

        risk_level = data.get("risk_level", "routine")
        ambiguity_clarifying_question = str(data.get("ambiguity_clarifying_question", "")).strip()
        ambiguity_reply_options = [
            {"display": str(o.get("display", "")).strip(), "prompt": str(o.get("prompt", "")).strip()}
            for o in (data.get("ambiguity_reply_options", []) or [])
            if isinstance(o, dict)
            and str(o.get("display", "")).strip()
            and str(o.get("prompt", "")).strip()
        ][:3]
        # A broken/empty clarification (missing question or options) must never surface as an
        # interrupt -- only ask when we have something usable to show the patient.
        ambiguous_term_detected = bool(
            data.get("ambiguous_term_detected", False)
            and ambiguity_clarifying_question
            and ambiguity_reply_options
            and risk_level not in ("urgent", "crisis")
        )
        clarifying_questions = [
            str(item).strip()
            for item in (data.get("clarifying_questions", []) or [])
            if str(item).strip()
        ][:3]
        clarification_required = bool(
            data.get("clarification_required", False)
            and clarifying_questions
            and risk_level not in ("urgent", "crisis")
            and not ambiguous_term_detected
        )

        intent_category = data.get("intent_category", "general_info")
        raw_pathway = data.get("pathway_hint", "")
        valid_pathways = {
            "general_triage", "maternity", "msk", "medications", "chronic_conditions"
        }
        pathway_hint = (
            raw_pathway
            if raw_pathway in valid_pathways
            else _INTENT_TO_PATHWAY.get(intent_category, "general_triage")
        )

        return IntentClassification(
            intent_category=intent_category,
            risk_level=risk_level,
            vulnerable_flags=data.get("vulnerable_flags", []),
            escalation_required=bool(data.get("escalation_required", False)),
            escalation_reason=data.get("escalation_reason", ""),
            crisis_detected=data.get("risk_level", "") == "crisis",
            pathway_hint=pathway_hint,
            confidence=float(data.get("confidence", 0.8)),
            presentation_hint=presentation_hint,
            ambiguous_term_detected=ambiguous_term_detected,
            ambiguous_term=str(data.get("ambiguous_term", "")).strip() if ambiguous_term_detected else "",
            ambiguity_clarifying_question=ambiguity_clarifying_question if ambiguous_term_detected else "",
            ambiguity_reply_options=ambiguity_reply_options if ambiguous_term_detected else [],
            clarification_required=clarification_required,
            clarification_reason=(
                str(data.get("clarification_reason", "")).strip()
                if clarification_required
                else ""
            ),
            clarifying_questions=(
                clarifying_questions if clarification_required else []
            ),
        )

    @staticmethod
    def _safe_default() -> IntentClassification:
        """
        Fallback when LLM classification fails.
        Returns the most conservative safe default without any keyword or symptom matching --
        downstream systems (policy gate, clinical decision support) apply their own logic.
        """
        return IntentClassification(
            intent_category="symptom_triage",
            risk_level="elevated",
            escalation_required=False,
            pathway_hint="general_triage",
            confidence=0.3,
        )
