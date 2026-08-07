"""
Clinical note generation and management.

Produces standard SOAP notes (Subjective / Objective / Assessment / Plan)
from a FlynnMed conversation and the patient's stored profile.
Notes are stored per-user and can be edited by clinicians before sharing.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

from backend.response_templates import get_persona_block
from backend.user_store import UserStore, compute_current_age
from backend.utils import render_vital_for_prompt


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_section(value) -> str:
    """Convert LLM output to a clean markdown string regardless of type returned."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for i, item in enumerate(value, 1):
            if isinstance(item, str):
                line = item.strip()
                # already numbered?
                parts.append(line if (line[:2].rstrip('.').isdigit()) else f"{i}. {line}")
            elif isinstance(item, dict):
                parts.append(f"{i}. " + "; ".join(f"**{k}**: {v}" for k, v in item.items()))
        return "\n".join(parts)
    if isinstance(value, dict):
        lines = []
        for k, v in value.items():
            label = k.replace("_", " ").title()
            if isinstance(v, list):
                lines.append(f"**{label}:**")
                for item in v:
                    lines.append(f"- {item}")
            elif isinstance(v, dict):
                lines.append(f"**{label}:**")
                for sk, sv in v.items():
                    lines.append(f"- {sk.replace('_', ' ').title()}: {sv}")
            else:
                lines.append(f"**{label}:** {v}")
        return "\n".join(lines)
    return str(value).strip()


def _build_objective_section(
    user_profile: dict,
    vitals: List[Dict],
    medications: List[Dict],
    conditions: List[Dict],
    allergies: List[Dict],
) -> str:
    """Build the Objective section from structured patient data."""
    lines: List[str] = []

    age = compute_current_age(user_profile.get("date_of_birth", ""))
    sex = user_profile.get("biological_sex", "Not stated")
    dob = user_profile.get("date_of_birth", "")
    demo = " | ".join(filter(None, [f"Age {age}" if age else "", sex, dob]))
    if demo:
        lines.append(f"Demographics: {demo}")

    active_conditions = [c["name"] for c in conditions if c.get("status") == "active"]
    if active_conditions:
        lines.append(f"Active conditions: {', '.join(active_conditions)}")

    past_conditions = [c["name"] for c in conditions if c.get("status") in ("past", "resolved")]
    if past_conditions:
        lines.append(f"Past conditions: {', '.join(past_conditions[:4])}")

    if medications:
        med_list = [
            f"{m['name']}{' ' + m['dose'] if m.get('dose') else ''}"
            f"{' (' + m['schedule'] + ')' if m.get('schedule') else ''}"
            for m in medications[:10]
        ]
        lines.append(f"Current medications: {'; '.join(med_list)}")

    if allergies:
        allergy_list = [
            f"{a['name']} ({a.get('severity', 'unknown severity')}, {a.get('allergy_type', 'unknown type')})"
            for a in allergies[:6]
        ]
        lines.append(f"Allergies / ADRs: {'; '.join(allergy_list)}")

    if vitals:
        seen_types: set = set()
        vital_lines: List[str] = []
        for v in vitals:
            vtype = v.get("type", "")
            if vtype and vtype not in seen_types:
                seen_types.add(vtype)
                truncated = dict(v, recorded_on=v.get("recorded_on", "")[:10])
                vital_lines.append(render_vital_for_prompt(truncated, date_prefix="recorded "))
        if vital_lines:
            lines.append("Recent vitals / labs:\n  " + "\n  ".join(vital_lines[:10]))

    return "\n".join(lines) if lines else "No objective data recorded in this account."


def generate_soap_note(
    username: str,
    conversation_summary: str,
    question: str,
    triage_summary: Optional[Dict],
    llm,
    vitals: Optional[List[Dict]] = None,
    medications: Optional[List[Dict]] = None,
    conditions: Optional[List[Dict]] = None,
    allergies: Optional[List[Dict]] = None,
    trace_id: Optional[str] = None,
) -> Dict:
    """
    Generate a SOAP note from the conversation context and patient profile.
    Returns a note dict ready for storage. Does NOT save -- caller decides.
    """
    from backend.summarizer import LLMHelper

    user_profile = UserStore.get_user_profile(username) or {}
    resolved_vitals = vitals if vitals is not None else UserStore.get_vitals(username, limit=20)
    resolved_meds = medications if medications is not None else UserStore.get_medications(username)
    resolved_conditions = conditions if conditions is not None else UserStore.get_conditions(username)
    resolved_allergies = allergies if allergies is not None else UserStore.get_allergies(username)

    objective_section = _build_objective_section(
        user_profile, resolved_vitals, resolved_meds, resolved_conditions, resolved_allergies
    )

    urgency = "routine"
    requires_gp = False
    gp_reason = ""
    if triage_summary:
        urgency = triage_summary.get("urgency_level", "routine").lower()
        next_step = triage_summary.get("next_step", "")
        requires_gp = urgency in ("high", "urgent", "crisis") or "gp" in next_step.lower()
        gp_reason = next_step

    role_key = (UserStore.get_user_profile(username) or {}).get("clinical_role", "doctor")
    role_guidance = {
        "doctor": (
            "Use full UK GP/hospital SOAP format. "
            "Assessment: differential diagnosis, clinical impression, risk stratification. "
            "Plan: numbered investigations, referrals, medications, follow-up, safety-netting."
        ),
        "nurse": (
            "Use a nursing SOAP format. "
            "Objective: observations, NEWS2 score if relevant, pressure area/falls risk. "
            "Assessment: nursing diagnosis, risk scores. "
            "Plan: nursing interventions, care tasks, patient education, escalation criteria, handover notes."
        ),
        "midwife": (
            "Use a midwifery SOAP format. "
            "Objective: maternal observations, fetal assessment (movement, CTG if relevant), gestation. "
            "Assessment: maternal and fetal risk assessment. "
            "Plan: maternity care pathway, referrals, birth plan considerations."
        ),
        "physiotherapist": (
            "Use a physiotherapy SOAP format. "
            "Objective: range of motion, strength grades, special orthopaedic tests, functional assessment. "
            "Assessment: clinical impression, problem list, functional diagnosis. "
            "Plan: treatment goals, exercise programme, manual therapy, home exercise plan, review date."
        ),
    }.get(str(role_key).lower(), (
        "Use a standard UK clinical SOAP format appropriate for this clinician's role."
    ))

    prompt = (
        "You are a clinical note writer for FlynnMed. "
        f"Generate a SOAP note for a {role_key} in UK clinical format.\n"
        f"Role guidance: {role_guidance}\n\n"
        f"CONSULTATION SUMMARY:\n{conversation_summary}\n\n"
        f"PATIENT QUESTION: {question}\n\n"
        f"OBJECTIVE DATA (use exactly this data, formatted cleanly):\n{objective_section}\n\n"
        f"URGENCY LEVEL: {urgency}\n\n"
        "IMPORTANT: ALL field values MUST be plain strings (no lists, no dicts, no JSON objects inside).\n"
        "Return ONLY a JSON object with these exact string fields:\n"
        "{\n"
        '  "subjective": "2-4 sentences of patient narrative in clinical language.",\n'
        '  "objective": "Formatted text of objective findings -- demographics, conditions, meds, vitals each on a new line.',
        ' Use plain text, not nested JSON.",\n'
        '  "assessment": "2-4 sentences: clinical impression, key findings, risk level, differentials.",\n'
        '  "plan": "Numbered steps as a single string, each step on a new line:\\n1. Step one\\n2. Step two\\n3. Step three"\n'
        "}\n\n"
        "Do not wrap values in lists or dicts. Each value must be a plain text string."
    )

    try:
        response = llm.client.chat.completions.create(
            model=LLMHelper.AUX_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_completion_tokens=900,
        )
        raw = response.choices[0].message.content or "{}"
        sections = json.loads(raw)
    except Exception as exc:
        print(f"[ClinicalNotes] SOAP generation failed: {exc}")
        sections = {
            "subjective": f"Patient enquired about: {question}. Conversation context: {conversation_summary[:200]}",
            "objective": objective_section,
            "assessment": "Unable to auto-generate assessment -- please complete manually.",
            "plan": "Please complete this section manually based on clinical judgement.",
        }

    note_id = f"note-{uuid4().hex[:12]}"
    now = _utc_now()
    display_name = user_profile.get("display_name", username)

    return {
        "note_id": note_id,
        "created_at": now,
        "updated_at": now,
        "username": username,
        "display_name": display_name,
        "trace_id": trace_id or "",
        "question": question[:300],
        "subjective": _coerce_section(sections.get("subjective", "")),
        "objective": _coerce_section(sections.get("objective", objective_section)),
        "assessment": _coerce_section(sections.get("assessment", "")),
        "plan": _coerce_section(sections.get("plan", "")),
        "role_key": role_key,
        "urgency_level": urgency,
        "requires_gp_visit": requires_gp,
        "gp_visit_reason": gp_reason,
        "generated_by": "flynnmed_ai",
        "edited_by": None,
        "email_sent": False,
        "email_sent_at": None,
    }


def _format_history_section(patient_chart: Dict) -> str:
    """
    The longitudinal/history material _build_objective_section doesn't
    cover -- recent symptom logs, triage history, and prior care-plan/
    clinical-note summaries -- formatted for a chart-review briefing rather
    than a single-visit SOAP note.
    """
    lines: List[str] = []

    symptom_logs = patient_chart.get("symptom_logs") or []
    if symptom_logs:
        recent = [
            f"{s.get('symptom', '')} (severity {s.get('severity', '?')}, {s.get('logged_for', 'date unknown')})"
            for s in symptom_logs[:8]
        ]
        lines.append("Recent symptom logs: " + "; ".join(recent))

    triage_summaries = patient_chart.get("triage_summaries") or []
    if triage_summaries:
        recent_triage = [
            f"{t.get('urgency_level', '?')} -- {t.get('next_step', '')} ({t.get('question', '')[:80]})"
            for t in triage_summaries[:5]
        ]
        lines.append("Recent triage history: " + "; ".join(recent_triage))

    care_plans = patient_chart.get("care_plans") or []
    if care_plans:
        plan_lines = [
            f"{cp.get('title', cp.get('condition', 'Care plan'))} ({cp.get('status', 'unknown')})"
            for cp in care_plans[:5]
        ]
        lines.append("Active care plans: " + "; ".join(plan_lines))

    clinical_notes = patient_chart.get("clinical_notes") or []
    if clinical_notes:
        note_lines = [
            f"{cn.get('created_at', '')[:10]}: {cn.get('assessment', '')[:150]}"
            for cn in clinical_notes[:5]
        ]
        lines.append("Prior clinical notes (assessment excerpts):\n  " + "\n  ".join(note_lines))

    return "\n".join(lines) if lines else "No additional symptom/triage/care-plan history recorded."


def generate_previsit_chart_summary(
    patient_chart: Dict,
    chat_transcript: List[Dict],
    clinician_role_key: str,
    llm,
    previous_draft: Optional[str] = None,
) -> str:
    """
    Synthesizes a patient's full chart into one flowing pre-visit briefing --
    contrast with generate_soap_note above, which is tightly coupled to
    summarizing a single live conversation into 4 fixed SOAP fields tied to
    one triage event. This instead reviews the whole longitudinal record
    (conditions, meds, allergies, vitals trend, symptom logs, triage
    history, prior care plans/notes) into a chart-review narrative, and --
    when previous_draft/chat_transcript are supplied, i.e. the "regenerate"
    path after the clinician has asked follow-up questions -- builds on that
    prior material rather than starting fresh. Returns a single markdown
    string (not a JSON object of fixed fields). Never touches the
    retrieval/evidence pipeline -- this is structured-data synthesis, not an
    evidence-grounded clinical question; the inline chat is what gets that.
    """
    from backend.summarizer import LLMHelper

    user_profile = patient_chart.get("user_profile") or {}
    objective_section = _build_objective_section(
        user_profile,
        patient_chart.get("vitals") or [],
        patient_chart.get("medications") or [],
        patient_chart.get("conditions") or [],
        patient_chart.get("allergies") or [],
    )
    history_section = _format_history_section(patient_chart)

    transcript_block = ""
    if chat_transcript:
        turns = [
            f"{'Clinician' if t.get('role') == 'clinician' else 'Assistant'}: {t.get('content', '')[:400]}"
            for t in chat_transcript[-12:]
        ]
        transcript_block = (
            "\n\nEXPLORATORY CHAT WITH THE CLINICIAN (incorporate what was learned here):\n"
            + "\n".join(turns)
        )

    previous_draft_block = (
        f"\n\nPREVIOUS DRAFT (revise and improve, don't discard prior clinical reasoning "
        f"unless the chart or chat above contradicts it):\n{previous_draft}"
        if previous_draft
        else ""
    )

    persona = get_persona_block(clinician_role_key)
    prompt = (
        f"{persona}\n\n"
        "You are drafting a pre-visit chart-review briefing for this clinician, to read "
        "before a call or visit with this patient. Synthesize the record below into one "
        "flowing markdown summary -- not a SOAP note, not a fixed-field form. Cover: "
        "overview of active issues, anything notable that changed recently, and 1-3 "
        "specific points worth raising or confirming at this visit. Be concise and "
        "clinically precise; do not invent facts not present in the record below.\n\n"
        f"CHART -- OBJECTIVE DATA:\n{objective_section}\n\n"
        f"CHART -- HISTORY:\n{history_section}"
        f"{transcript_block}"
        f"{previous_draft_block}\n\n"
        "Return only the markdown briefing text, nothing else."
    )

    try:
        response = llm.client.chat.completions.create(
            model=LLMHelper.AUX_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[ClinicalNotes] Pre-visit summary generation failed: {exc}")
        return (
            "_AI drafting failed -- showing raw chart data instead. Please write the "
            "summary manually or try regenerating._\n\n"
            f"{objective_section}\n\n{history_section}"
        )
