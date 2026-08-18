"""
Clinical policy engine: rule-based gate that applies hard safety constraints,
escalation requirements, and vulnerable-population logic.
Pure Python -- no LLM calls.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List

from backend.audit_models import PolicyGateRecord
from backend.intent_risk_classifier import IntentClassification
from backend.role_router import RoleConfig
from backend.response_templates import (
    build_crisis_response,
    build_escalation_banner,
    build_no_diagnosis_disclaimer,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Medication risk patterns ───────────────────────────────────────────────────
_MED_DOSAGE_PATIENT_PATTERN = re.compile(
    r"\b(how much|what dose|dosage|how many (mg|milligrams?|tablets?|pills?))\b",
    re.IGNORECASE,
)
_PREGNANCY_MED_PATTERN = re.compile(
    r"(safe (in|during|for) pregnan|can i take .{0,30} pregnant|"
    r"(medication|drug|tablet) .{0,30} pregnan)",
    re.IGNORECASE,
)
_DIAGNOSIS_SEEKING_PATTERN = re.compile(
    r"\b(do i have|is it|could it be|am i|is this|could this be|"
    r"diagnos|what (?:is|are) (?:wrong|my condition))\b",
    re.IGNORECASE,
)


@dataclass
class PolicyDecision:
    action: str = "allow"                        # "allow" | "escalate_only" | "block"
    gates_applied: List[PolicyGateRecord] = field(default_factory=list)
    context_notes: List[str] = field(default_factory=list)   # injected into LLM context
    escalation_banner: str = ""                  # prepended to answer if non-empty
    vulnerability_notice: str = ""               # appended near top of answer
    disclaimer: str = ""                         # appended at bottom of answer
    crisis_response: str = ""                    # returned verbatim without LLM if non-empty

    def add_gate(self, gate: PolicyGateRecord) -> None:
        self.gates_applied.append(gate)

    def gates_as_dicts(self) -> List[dict]:
        return [g.as_dict() for g in self.gates_applied]


class PolicyEngine:
    """
    Applies hard clinical safety gates based on intent classification and role.
    Call gate() once per request, before LLM generation.
    """

    def gate(
        self,
        intent: IntentClassification,
        role_config: RoleConfig,
        question: str,
        patient_history=None,
    ) -> PolicyDecision:
        decision = PolicyDecision()

        # Accumulate vulnerability flags from both intent and role config
        all_flags = list(set(intent.vulnerable_flags + role_config.vulnerable_population_flags))

        # Apply gates in priority order
        self._gate_crisis(intent, role_config, decision)
        if decision.action == "escalate_only":
            return decision

        # Known-condition gate runs before other gates so it can raise risk_level
        # in time for _gate_urgent_escalation to act on it
        if patient_history is not None and not patient_history.is_empty():
            self._gate_known_condition_risk(intent, role_config, question, patient_history, decision)
            if decision.action == "escalate_only":
                return decision

        self._gate_urgent_escalation(intent, role_config, decision)
        self._gate_pregnancy(intent, role_config, question, decision)
        self._gate_postpartum_newborn(intent, decision)
        self._gate_paediatric(intent, all_flags, decision)
        self._gate_allergy_contraindication(intent, role_config, patient_history, decision)
        self._gate_medication_dosage(intent, role_config, question, decision)
        self._gate_diagnosis_request(intent, role_config, question, decision)
        self._gate_elderly_polypharmacy(intent, all_flags, decision)
        self._gate_mental_health(intent, role_config, decision)

        # A vulnerability label alone is not an escalation trigger and should not
        # add generic safety padding.  Relevant gates already contribute focused
        # instructions when the current request engages the context.

        # Disclaimers help on personal clinical questions, but are noise on
        # administrative and purely definitional requests.
        if intent.intent_category not in ("administrative", "general_info"):
            decision.disclaimer = build_no_diagnosis_disclaimer(role_config.role_key)

        return decision

    # ── Individual gates ────────────────────────────────────────────────────────

    def _gate_crisis(
        self,
        intent: IntentClassification,
        role_config: RoleConfig,
        decision: PolicyDecision,
    ) -> None:
        if not intent.crisis_detected:
            return

        gate = PolicyGateRecord(
            gate_name="crisis",
            applied=True,
            reason="Crisis-level risk detected -- returning emergency guidance without LLM generation.",
        )
        decision.add_gate(gate)
        decision.action = "escalate_only"
        decision.crisis_response = build_crisis_response(role_config.role_key)

    def _gate_urgent_escalation(
        self,
        intent: IntentClassification,
        role_config: RoleConfig,
        decision: PolicyDecision,
    ) -> None:
        if intent.risk_level not in ("urgent", "crisis"):
            return
        if role_config.escalation_threshold == "high":
            # Clinical roles -- add context note but don't force banner
            gate = PolicyGateRecord(
                gate_name="urgent_clinical",
                applied=True,
                reason=f"Urgent intent detected for clinical user ({role_config.role_key}).",
            )
            decision.add_gate(gate)
            decision.context_notes.append(
                "POLICY NOTE: This question has been classified as urgent. "
                "Prioritise red flag information, disposition, and immediate action guidance in your response."
            )
        else:
            # Non-clinical roles -- force escalation banner
            reason = intent.escalation_reason or "Urgent clinical concern detected."
            gate = PolicyGateRecord(
                gate_name="urgent_escalation",
                applied=True,
                reason=reason,
            )
            decision.add_gate(gate)
            decision.escalation_banner = build_escalation_banner(reason, role_config.role_key)
            decision.context_notes.append(
                f"POLICY NOTE: Urgent risk level. Always lead with escalation guidance. "
                f"Then give the clearest safe next step and timeframe. Reason: {reason}"
            )

    def _gate_pregnancy(
        self,
        intent: IntentClassification,
        role_config: RoleConfig,
        question: str,
        decision: PolicyDecision,
    ) -> None:
        # The role describes the user, not the patient in the current question.
        # Only a question-specific pregnancy flag or an explicit pregnancy and
        # medicine phrase should activate pregnancy-only guidance.
        has_pregnancy_flag = "pregnancy" in intent.vulnerable_flags
        has_pregnancy_med = bool(_PREGNANCY_MED_PATTERN.search(question))

        if not (has_pregnancy_flag or has_pregnancy_med):
            return

        gate = PolicyGateRecord(
            gate_name="pregnancy_safety",
            applied=True,
            reason="Pregnancy context detected -- applying heightened medication and escalation safety.",
        )
        decision.add_gate(gate)
        decision.context_notes.append(
            "POLICY NOTE: Pregnancy context is present. "
            "Apply heightened caution for all medication, dosage, and intervention advice. "
            "Reference NICE/RCOG guidelines specifically. "
            "Never recommend stopping or starting prescription medication without explicit NICE guidance. "
            "Include only obstetric warning signs connected to the presentation and state a proportionate care route."
        )
        if role_config.role_key not in ("midwife", "doctor"):
            decision.escalation_banner = build_escalation_banner(
                "Pregnancy-related question -- always verify medication safety with your midwife or GP.",
                role_config.role_key,
            )

    def _gate_paediatric(
        self,
        intent: IntentClassification,
        all_flags: List[str],
        decision: PolicyDecision,
    ) -> None:
        if "paediatric" not in all_flags:
            return

        gate = PolicyGateRecord(
            gate_name="paediatric_safety",
            applied=True,
            reason="Paediatric population flag -- applying child-specific safety thresholds.",
        )
        decision.add_gate(gate)
        decision.context_notes.append(
            "POLICY NOTE: Paediatric context. "
            "All dosing, weight-based recommendations, and growth milestones must be explicitly age-qualified. "
            "Never extrapolate adult guidance to children without stating it. "
            "Refer to BNFC and NICE paediatric pathways."
        )

    def _gate_postpartum_newborn(
        self,
        intent: IntentClassification,
        decision: PolicyDecision,
    ) -> None:
        """Apply precise postnatal context without reusing pregnancy wording."""
        flags = set(intent.vulnerable_flags)
        if "postpartum" in flags:
            decision.add_gate(
                PolicyGateRecord(
                    gate_name="postpartum_safety",
                    applied=True,
                    reason="Postpartum context detected; applying postnatal guidance.",
                )
            )
            decision.context_notes.append(
                "POLICY NOTE: Postpartum context is present. Use postnatal guidance and "
                "do not describe the patient as pregnant unless pregnancy is separately "
                "established. Consider breastfeeding-specific medicine advice only when "
                "breastfeeding is confirmed."
            )

        if "newborn" in flags:
            decision.add_gate(
                PolicyGateRecord(
                    gate_name="newborn_safety",
                    applied=True,
                    reason="Newborn context detected; applying neonatal guidance.",
                )
            )
            decision.context_notes.append(
                "POLICY NOTE: Newborn context is present. Use neonatal guidance and "
                "age- and weight-specific medicine thresholds. Do not substitute "
                "pregnancy or general adult advice."
            )

        if intent.intent_category != "medication_query":
            return
        if "newborn" in flags:
            decision.vulnerability_notice = (
                "> **Newborn medicine check:** Newborn medicines require age- and "
                "weight-specific review. Confirm the exact product and dose with the "
                "child's clinician or a pharmacist before giving or changing it.\n\n"
            )
        elif "postpartum" in flags:
            decision.vulnerability_notice = (
                "> **Postpartum medicine check:** Medicine safety can differ after birth, "
                "particularly when breastfeeding. Confirm the exact medicine with a "
                "midwife, GP or pharmacist before changing it.\n\n"
            )

    def _gate_allergy_contraindication(
        self,
        intent: IntentClassification,
        role_config: RoleConfig,
        patient_history,
        decision: PolicyDecision,
    ) -> None:
        if patient_history is None or not patient_history.known_allergies:
            return
        if intent.intent_category not in ("medication_query", "symptom_triage", "chronic_condition"):
            return

        allergy_list = "; ".join(patient_history.known_allergies[:10])
        gate = PolicyGateRecord(
            gate_name="allergy_contraindication",
            applied=True,
            reason=f"Patient has recorded allergies/contraindications: {allergy_list}",
        )
        decision.add_gate(gate)
        decision.context_notes.append(
            f"SAFETY FLAG -- PATIENT ALLERGIES / CONTRAINDICATIONS: {allergy_list}\n"
            "Before recommending, mentioning, or discussing any medication, treatment, or substance:\n"
            "1. Check whether it or any cross-reactive agent appears in the allergy list above.\n"
            "2. If there is any match or plausible cross-reaction, flag it prominently at the top of your answer.\n"
            "3. Recommend verification with the prescribing clinician or pharmacist before use.\n"
            "4. If no conflict is identified, state explicitly that the recommendations do not appear to conflict "
            "with the recorded allergies -- but remind the patient that the list reflects only what has been entered."
        )

    def _gate_medication_dosage(
        self,
        intent: IntentClassification,
        role_config: RoleConfig,
        question: str,
        decision: PolicyDecision,
    ) -> None:
        if intent.intent_category != "medication_query":
            if not _MED_DOSAGE_PATIENT_PATTERN.search(question):
                return

        if role_config.role_key in ("doctor", "nurse", "midwife", "physiotherapist", "healthcare_professional"):
            # Clinical roles get BNF context note
            gate = PolicyGateRecord(
                gate_name="medication_clinical",
                applied=True,
                reason="Medication query for clinical user -- BNF/NICE reference context added.",
            )
            decision.add_gate(gate)
            decision.context_notes.append(
                "POLICY NOTE: Medication query for a clinician. "
                "Reference BNF, NICE, or MHRA guidance. "
                "Show evidence uncertainty for off-label or non-guideline uses explicitly. "
                "When the evidence supports it, state concrete monitoring parameters, contraindications, and escalation thresholds."
            )
        else:
            # Lay roles get pharmacist/GP referral note
            gate = PolicyGateRecord(
                gate_name="medication_lay",
                applied=True,
                reason="Medication dosage question for lay user -- pharmacist/GP referral required.",
            )
            decision.add_gate(gate)
            decision.context_notes.append(
                "POLICY NOTE: Medication question from a patient or caregiver. "
                "Do not provide specific dosage advice for prescription medicines. "
                "Always recommend verification with a pharmacist or GP. "
                "Use BNF/NICE as the source basis and give the clearest safe care route."
            )

    def _gate_diagnosis_request(
        self,
        intent: IntentClassification,
        role_config: RoleConfig,
        question: str,
        decision: PolicyDecision,
    ) -> None:
        is_diagnosis_seeking = bool(_DIAGNOSIS_SEEKING_PATTERN.search(question))
        if not is_diagnosis_seeking:
            return
        if role_config.role_key in ("doctor",):
            # Doctor receives differential discussion context
            gate = PolicyGateRecord(
                gate_name="diagnosis_clinical",
                applied=True,
                reason="Diagnostic question for clinician -- differential discussion permitted with uncertainty labelling.",
            )
            decision.add_gate(gate)
            decision.context_notes.append(
                "POLICY NOTE: Diagnostic question from a clinician. "
                "Differential discussion is appropriate but must be clearly framed as hypothesis, "
                "not definitive diagnosis. Label evidence quality explicitly and link the differential to immediate management priorities."
            )
        else:
            gate = PolicyGateRecord(
                gate_name="no_diagnosis",
                applied=True,
                reason="Diagnosis-seeking language from non-clinician -- no-diagnosis policy enforced.",
            )
            decision.add_gate(gate)
            decision.context_notes.append(
                "POLICY NOTE: This user appears to be seeking a diagnosis. "
                "You must NOT provide a definitive diagnosis. "
                "Explain possible causes and what the symptoms may suggest, "
                "but always direct them to the clearest appropriate care route for diagnosis."
            )

    def _gate_elderly_polypharmacy(
        self,
        intent: IntentClassification,
        all_flags: List[str],
        decision: PolicyDecision,
    ) -> None:
        if "elderly" not in all_flags:
            return
        if intent.intent_category != "medication_query":
            return

        gate = PolicyGateRecord(
            gate_name="elderly_polypharmacy",
            applied=True,
            reason="Elderly patient + medication query -- polypharmacy and renal function considerations added.",
        )
        decision.add_gate(gate)
        decision.context_notes.append(
            "POLICY NOTE: Elderly population + medication context. "
            "Highlight polypharmacy risks, renal/hepatic dosing adjustments, and fall risk. "
            "Reference STOPP/START criteria and NICE guidance for older people."
        )

    def _gate_mental_health(
        self,
        intent: IntentClassification,
        role_config: RoleConfig,
        decision: PolicyDecision,
    ) -> None:
        if intent.intent_category != "mental_health":
            return

        gate = PolicyGateRecord(
            gate_name="mental_health",
            applied=True,
            reason="Mental health topic detected -- crisis resources and empathetic framing required.",
        )
        decision.add_gate(gate)
        decision.context_notes.append(
            "POLICY NOTE: Mental health topic. "
            "Include locally verified crisis support only when self-harm, suicide, or immediate safety concerns are present. "
            "Use empathetic, non-stigmatising language. "
            "Never minimise distress. "
            "If any self-harm or suicidal ideation is implied, apply crisis response."
        )

    def _gate_known_condition_risk(
        self,
        intent: IntentClassification,
        role_config: RoleConfig,
        question: str,
        patient_history,
        decision: PolicyDecision,
    ) -> None:
        """
        Applies condition-aware safety rules based on the patient's known history.

        Critically: this gate does NOT inspect the raw question text for symptoms.
        The LLM intent classifier already received the full patient history context
        and produced intent.risk_level / intent.intent_category / intent.escalation_required
        accordingly. This gate trusts that output and applies three types of action:

        REINFORCE   -- when the LLM already flagged elevated/urgent risk and the patient
                      has a known condition that makes those symptoms more dangerous,
                      confirm escalation_required and add condition-specific context.

        ELEVATE     -- for high-risk condition + intent category combinations where ANY
                      relevant clinical question carries a higher baseline risk (e.g.
                      immunocompromised + any symptom question), bump routine → elevated
                      even if the LLM did not.

        CONTEXT     -- inject condition-specific clinical guidance into the LLM prompt
                      for every relevant combination regardless of risk level, so the
                      response covers the right red flags, recommends BP checks, lists
                      monitoring advice, etc. -- without us hardcoding what those flags are.
        """
        is_clinical = role_config.role_key in ("doctor", "nurse", "midwife", "physiotherapist", "healthcare_professional")

        escalate_notes: List[str] = []
        context_notes: List[str] = []

        # Shorthand for intent categories
        is_symptom   = intent.intent_category in ("symptom_triage", "chronic_condition", "maternity", "msk")
        is_med_query = intent.intent_category == "medication_query"
        is_mental    = intent.intent_category == "mental_health"
        llm_risk     = intent.risk_level  # what the LLM decided (already saw history)

        # Historical conditions and vulnerability labels do not independently
        # prove deterioration in the current presentation.  Reinforce symptom
        # escalation only after compatible urgent facts were identified.  Drug
        # questions continue so relevant contraindications can still be checked.
        if is_symptom and not (
            intent.escalation_required
            or intent.crisis_detected
            or llm_risk in ("urgent", "crisis")
        ):
            return

        # ── CARDIAC ──────────────────────────────────────────────────────────────
        # The LLM already had cardiac history context. If it still flagged elevated+
        # risk on a symptom question, reinforce escalation. Otherwise add guidance.
        if patient_history.has_cardiac_history and is_symptom:
            if llm_risk in ("urgent", "crisis"):
                intent.escalation_required = True
                escalate_notes.append(
                    "Known cardiac history: LLM classified this as urgent -- escalation confirmed."
                )
            elif llm_risk == "elevated":
                if is_clinical:
                    context_notes.append(
                        "CLINICAL CONTEXT (cardiac history): Elevated-risk symptom in a patient with known "
                        "cardiac disease. Consider atypical ACS, arrhythmia, and heart failure exacerbation. "
                        "Do not dismiss without clinical assessment."
                    )
                else:
                    context_notes.append(
                        "CLINICAL CONTEXT (cardiac history): Patient has known heart disease. "
                        "Advise clinical review for any new or worsening symptoms. "
                        "Instruct the patient on when to seek emergency care."
                    )
            else:
                # routine -- add a general awareness note; do not escalate
                context_notes.append(
                    "CLINICAL CONTEXT (cardiac history): Patient has a known cardiac condition. "
                    "Ensure the response covers relevant red flags for this patient group and "
                    "advises when to seek urgent review."
                )

        # ── HYPERTENSION ─────────────────────────────────────────────────────────
        # Do NOT escalate based on risk level alone. The LLM should ask about red flags.
        # We inject guidance so the response covers: BP check, what warrants emergency,
        # what warrants GP review, and what is self-manageable.
        if patient_history.has_hypertension and is_symptom:
            if llm_risk in ("urgent", "crisis"):
                intent.escalation_required = True
                escalate_notes.append(
                    "Known hypertension + urgent classification: confirm escalation and cover "
                    "hypertensive emergency red flags in response."
                )
            else:
                if is_clinical:
                    context_notes.append(
                        "CLINICAL CONTEXT (hypertension): Patient has known hypertension. "
                        "For any symptom question: consider whether BP is controlled and whether "
                        "red flags for hypertensive emergency are present (end-organ symptoms: "
                        "visual disturbance, confusion, focal neurology, chest pain, oliguria). "
                        "If absent and BP controlled: reassurance + routine GP follow-up. "
                        "If absent but BP uncontrolled: same-day GP review. "
                        "Do not classify a headache as a hypertensive emergency without red flags."
                    )
                else:
                    context_notes.append(
                        "CLINICAL CONTEXT (hypertension): Patient has known high blood pressure. "
                        "The response should: recommend checking BP if possible, explain which "
                        "symptoms in this patient warrant same-day GP review versus emergency care, "
                        "and clearly state that not every headache or dizzy spell is an emergency. "
                        "Include red flags that should prompt calling 999."
                    )

        # ── DIABETES ─────────────────────────────────────────────────────────────
        # If the LLM flagged elevated+ risk, reinforce and add diabetes-specific context.
        # For routine questions still add monitoring guidance.
        if patient_history.has_diabetes and is_symptom:
            if llm_risk in ("urgent", "crisis"):
                intent.escalation_required = True
                escalate_notes.append(
                    "Known diabetes + urgent classification: confirm escalation. Cover DKA / "
                    "hypoglycaemia / diabetic emergency signs in the response."
                )
            elif llm_risk == "elevated":
                intent.risk_level = _escalate_risk(intent.risk_level)
                if is_clinical:
                    context_notes.append(
                        "CLINICAL CONTEXT (diabetes): Elevated-risk symptom in a diabetic patient. "
                        "Ensure the response covers relevant diabetic complications for this symptom "
                        "(e.g. hypoglycaemia, DKA, diabetic foot, retinopathy, nephropathy). "
                        "Recommend HbA1c and organ complication review where appropriate."
                    )
                else:
                    context_notes.append(
                        "CLINICAL CONTEXT (diabetes): Patient has diabetes. Ensure the response "
                        "explains how this symptom may relate to diabetes, when to seek prompt "
                        "GP review, and what constitutes an emergency."
                    )
            else:
                context_notes.append(
                    "CLINICAL CONTEXT (diabetes): Patient has known diabetes. The response should "
                    "note any relevant diabetes-specific considerations for this symptom and advise "
                    "appropriate monitoring or clinical review."
                )

        # ── ANTICOAGULANTS ───────────────────────────────────────────────────────
        # Any symptom question OR medication question in an anticoagulated patient
        # carries additional risk. If the LLM flagged elevated+, confirm escalation.
        # Always add anticoagulant-specific guidance regardless of risk level.
        if patient_history.on_anticoagulants and (is_symptom or is_med_query):
            if llm_risk in ("elevated", "urgent", "crisis"):
                intent.escalation_required = True
                escalate_notes.append(
                    "On anticoagulants: elevated-risk question -- over-anticoagulation and bleeding "
                    "risk must be considered and addressed in the response."
                )
            context_notes.append(
                "CLINICAL CONTEXT (anticoagulants): Patient is on anticoagulant therapy. "
                "The response must cover anticoagulant-specific considerations: bleeding risk, "
                "drug interactions, signs of over-anticoagulation, and when same-day clinical "
                "review is required. Do not recommend medications that increase bleeding risk "
                "without appropriate caveats."
            )

        # ── IMMUNOCOMPROMISED ────────────────────────────────────────────────────
        # Any symptom question in an immunocompromised patient carries a higher baseline.
        # Bump routine → elevated regardless of LLM output; confirm escalation if higher.
        if patient_history.immunocompromised and is_symptom:
            if llm_risk in ("urgent", "crisis"):
                intent.escalation_required = True
                escalate_notes.append(
                    "Immunocompromised + urgent classification: infections deteriorate rapidly -- "
                    "escalation confirmed."
                )
            else:
                if intent.risk_level == "routine":
                    intent.risk_level = "elevated"
                if is_clinical:
                    context_notes.append(
                        "CLINICAL CONTEXT (immunocompromised): Patient is immunocompromised. "
                        "Apply a lower threshold for escalation. Cover neutropenic sepsis risk, "
                        "atypical infection presentations, and need for same-day review for "
                        "any new infection symptoms."
                    )
                else:
                    context_notes.append(
                        "CLINICAL CONTEXT (immunocompromised): Patient has a weakened immune system. "
                        "The response should advise that infections can become serious more quickly "
                        "and recommend prompt clinical review for any infection-related symptoms."
                    )

        # ── RENAL DISEASE ────────────────────────────────────────────────────────
        # Context only. No risk escalation -- medication safety and monitoring guidance.
        if patient_history.has_renal_disease and (is_symptom or is_med_query):
            context_notes.append(
                "CLINICAL CONTEXT (renal disease): Patient has known CKD/renal impairment. "
                "The response must flag any nephrotoxic medications (NSAIDs, aminoglycosides, "
                "contrast agents) as contraindicated or requiring dose adjustment. "
                "Cover eGFR and electrolyte monitoring. Note that new fluid retention, "
                "reduced urine output, or acute breathlessness in a renal patient warrants "
                "urgent review."
            )

        # ── RESPIRATORY DISEASE ──────────────────────────────────────────────────
        # Elevate if the LLM flagged elevated on a symptom question; add exacerbation context.
        if patient_history.has_respiratory_disease and is_symptom:
            if llm_risk in ("urgent", "crisis"):
                intent.escalation_required = True
                escalate_notes.append(
                    "Known respiratory disease + urgent classification: possible acute exacerbation -- "
                    "escalation confirmed."
                )
            elif llm_risk == "elevated":
                context_notes.append(
                    "CLINICAL CONTEXT (respiratory disease): Elevated-risk symptom in a patient "
                    "with known respiratory disease. Cover acute exacerbation features, "
                    "rescue inhaler use, when to seek emergency care, and escalation thresholds "
                    "(peak flow, sentence completion, cyanosis)."
                )
            else:
                context_notes.append(
                    "CLINICAL CONTEXT (respiratory disease): Patient has known respiratory disease. "
                    "Ensure the response notes relevant respiratory considerations and advises "
                    "when worsening symptoms require clinical review."
                )

        # ── STROKE / TIA ─────────────────────────────────────────────────────────
        # Any elevated+ symptom question → confirm escalation and mandate FAST guidance.
        # Routine symptom questions → always include FAST red flag instruction.
        if patient_history.has_stroke_history and is_symptom:
            if llm_risk in ("elevated", "urgent", "crisis"):
                intent.escalation_required = True
                escalate_notes.append(
                    "Previous stroke/TIA + elevated-risk symptom: stroke recurrence must be excluded -- "
                    "FAST assessment required."
                )
            context_notes.append(
                "CLINICAL CONTEXT (stroke/TIA history): Patient has a previous stroke or TIA. "
                "The response MUST include FAST assessment guidance (Face, Arms, Speech, Time to call 999) "
                "for any neurological, cardiovascular, or unexplained symptom. "
                "Be explicit about which symptoms require emergency response versus GP review."
            )

        # ── LIVER DISEASE ────────────────────────────────────────────────────────
        # Context-driven: medication safety + decompensation awareness.
        if patient_history.has_liver_disease and (is_symptom or is_med_query):
            if llm_risk in ("urgent", "crisis"):
                intent.escalation_required = True
                escalate_notes.append(
                    "Known liver disease + urgent classification: possible decompensation -- "
                    "escalation confirmed."
                )
            context_notes.append(
                "CLINICAL CONTEXT (liver disease): Patient has known liver disease. "
                "Cover hepatotoxicity risk for any medication discussed. "
                "Note reduced paracetamol dose ceiling. Flag signs of decompensation "
                "(jaundice, encephalopathy, ascites, haematemesis) as requiring emergency review."
            )

        # ── MENTAL HEALTH HISTORY ────────────────────────────────────────────────
        # If the LLM already flagged mental_health or crisis intent, reinforce.
        # For all mental health questions with history, add safety-net context.
        if patient_history.has_mental_health_history and (is_symptom or is_mental):
            if llm_risk == "crisis" or intent.crisis_detected:
                intent.escalation_required = True
                escalate_notes.append(
                    "Known mental health history + crisis-level classification: applying crisis response."
                )
            elif is_mental:
                context_notes.append(
                    "CLINICAL CONTEXT (mental health history): Patient has a known mental health "
                    "history. The response must include crisis support resources, use non-stigmatising "
                    "language, and clearly state when to seek urgent mental health support."
                )

        # ── Repeated previous escalations ────────────────────────────────────────
        # If this patient has been escalated urgently before, raise routine → elevated
        # for symptom questions. Do not over-ride elevated or higher.
        if patient_history.previous_escalation_count >= 2 and is_symptom:
            if intent.risk_level == "routine":
                intent.risk_level = "elevated"
                context_notes.append(
                    f"CLINICAL CONTEXT: Patient has {patient_history.previous_escalation_count} previous "
                    "urgent clinical escalations. Applying cautious baseline -- the response should "
                    "include clear guidance on when to seek review and not underplay symptoms."
                )

        # ── Write to decision ────────────────────────────────────────────────────
        if not escalate_notes and not context_notes:
            return

        for note in escalate_notes:
            decision.add_gate(PolicyGateRecord(
                gate_name="known_condition_risk",
                applied=True,
                reason=note,
            ))

        if escalate_notes:
            decision.context_notes.append(
                "POLICY NOTE (patient history): " + " | ".join(escalate_notes)
            )
        for note in context_notes:
            decision.context_notes.append(note)

        # Escalation banner for non-clinical users only, and only when required
        if intent.escalation_required and not is_clinical and escalate_notes:
            decision.escalation_banner = build_escalation_banner(
                escalate_notes[0], role_config.role_key
            )


def _escalate_risk(current: str) -> str:
    """Bump risk level up one step, capped at 'urgent'."""
    order = ["routine", "elevated", "urgent", "crisis"]
    try:
        idx = order.index(current)
        return order[min(idx + 1, 2)]  # cap at urgent; crisis only via crisis gate
    except ValueError:
        return "elevated"
