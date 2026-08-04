"""
FlynnMed MCP Server.

Exposes clinical tools via the Model Context Protocol.

Deployed (Railway):
  The FastAPI app mounts this at /mcp automatically via streamable HTTP.
  Set MCP_API_KEY in Railway environment variables to restrict access.
  Claude Desktop config:
    {
      "mcpServers": {
        "flynnmed": {
          "url": "https://<your-app>.railway.app/mcp",
          "headers": { "Authorization": "Bearer <MCP_API_KEY>" }
        }
      }
    }

Local (stdio -- Claude Desktop direct):
  python -m backend.mcp_server

Tools:
  get_patient_context           -- full patient profile, vitals, meds, conditions
  scrutinize_patient_context    -- resolve specialty/meaning before an agent acts
  validate_clinical_output      -- post-generation answer safety gate
  validate_care_plan_output     -- post-generation care-plan safety gate
  extract_article_evidence      -- structured evidence extraction from a medical article
  generate_clinical_note        -- generate a SOAP note from a consultation
  send_health_email             -- send clinical note or urgent alert by email
  search_trials_for_patient     -- search ClinicalTrials.gov for this patient
"""
from __future__ import annotations

import json

from dotenv import load_dotenv

load_dotenv()

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    message = (
        "MCP integration is unavailable. Install the compatible dependency with "
        "`pip install 'mcp>=1.28,<2'`. "
        f"Original import error: {exc}"
    )
    if __name__ == "__main__":
        raise SystemExit(message) from exc
    raise RuntimeError(message) from exc

from backend.clinical_notes import generate_soap_note  # noqa: E402
from backend.clinical_context_guard import (  # noqa: E402
    adjudicate_patient_context,
    decision_from_dict,
    validate_care_plan,
    validate_generated_answer,
)  # noqa: E402
from backend.email_service import send_clinical_note_email, send_urgent_care_alert  # noqa: E402
from backend.product_config import PRODUCT_NAME  # noqa: E402
from backend.summarizer import LLMHelper  # noqa: E402
from backend.user_store import UserStore  # noqa: E402

mcp = FastMCP(f"{PRODUCT_NAME} Clinical Tools")
_llm = LLMHelper()


# ─── Tool: get_patient_context ────────────────────────────────────────────────

@mcp.tool()
def get_patient_context(username: str) -> str:
    """
    Get the complete health context for a patient.

    Returns structured JSON with:
    - profile (demographics, role, care context)
    - conditions (active and past)
    - medications (with dose and schedule)
    - allergies
    - vitals and lab readings (last 20)
    - recent symptom logs
    - latest triage summary
    - longitudinal clinical memory
    """
    profile = UserStore.get_user_profile(username)
    if not profile:
        return json.dumps({"error": f"User '{username}' not found"})

    triage_list = UserStore.get_triage_summaries(username, limit=1)
    conditions = UserStore.get_conditions(username)
    medications = UserStore.get_medications(username)
    vitals = UserStore.get_vitals(username, limit=20)
    allergies = UserStore.get_allergies(username)
    triage_all = UserStore.get_triage_summaries(username, limit=None)
    context_decision = adjudicate_patient_context(
        conditions=conditions,
        medications=medications,
        vitals=vitals,
        allergies=allergies,
        triage_summaries=triage_all,
        document_summaries=UserStore.get_document_summaries(username),
        longitudinal_memory=(UserStore.get_longitudinal_memory(username) or {}).get("summary", ""),
    )
    return json.dumps(
        {
            "username": username,
            "profile": profile,
            "conditions": conditions,
            "medications": medications,
            "allergies": allergies,
            "vitals": vitals,
            "symptom_logs": UserStore.get_symptom_logs(username, limit=10),
            "latest_triage": triage_list[0] if triage_list else {},
            "longitudinal_memory": UserStore.get_longitudinal_memory(username),
            "clinical_context": context_decision.as_dict(),
        },
        default=str,
    )


@mcp.tool()
def scrutinize_patient_context(username: str, question: str = "", requested_topic: str = "") -> str:
    """Adjudicate a patient's structured record before another agent acts on it.

    This is the MCP equivalent of the in-process context gate. It returns the
    confirmed specialty/meaning, direct facts used, blocked interpretations,
    and whether a clarification is required. It never makes a diagnosis.
    """
    profile = UserStore.get_user_profile(username)
    if not profile:
        return json.dumps({"error": f"User '{username}' not found"})
    decision = adjudicate_patient_context(
        question=question,
        requested_topic=requested_topic,
        conditions=UserStore.get_conditions(username),
        medications=UserStore.get_medications(username),
        vitals=UserStore.get_vitals(username, limit=None),
        allergies=UserStore.get_allergies(username),
        triage_summaries=UserStore.get_triage_summaries(username, limit=None),
        document_summaries=UserStore.get_document_summaries(username),
        longitudinal_memory=(UserStore.get_longitudinal_memory(username) or {}).get("summary", ""),
    )
    return json.dumps({"clinical_context": decision.as_dict(), "prompt_block": decision.as_prompt_block()})


@mcp.tool()
def validate_clinical_output(
    username: str,
    question: str,
    answer: str,
    requested_topic: str = "",
) -> str:
    """Run the post-generation specialty check on an answer before delivery."""
    context_json = scrutinize_patient_context(username, question, requested_topic)
    context = json.loads(context_json)
    if context.get("error"):
        return context_json
    decision = decision_from_dict(context.get("clinical_context"))
    result = validate_generated_answer(answer, decision)
    if not result["valid"] and decision:
        result["safe_replacement"] = decision.correction_message()
    return json.dumps({"clinical_context": context.get("clinical_context", {}), "validation": result})


@mcp.tool()
def validate_care_plan_output(username: str, plan_json: str, requested_topic: str = "") -> str:
    """Run the post-generation specialty check on a care-plan JSON payload."""
    context_json = scrutinize_patient_context(username, requested_topic, requested_topic)
    context = json.loads(context_json)
    if context.get("error"):
        return context_json
    decision = decision_from_dict(context.get("clinical_context"))
    try:
        plan = json.loads(plan_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "plan_json must be valid JSON"})
    result = validate_care_plan(plan, decision)
    return json.dumps({"clinical_context": context.get("clinical_context", {}), "validation": result})


# ─── Tool: extract_article_evidence ──────────────────────────────────────────

@mcp.tool()
def extract_article_evidence(
    article_title: str,
    article_text: str,
    patient_question: str,
    patient_conditions: str,
    patient_medications: str,
    patient_age: str = "",
    evidence_tier: int = 3,
) -> str:
    """
    Extract structured, patient-specific evidence from a medical article.

    Parameters:
    - article_title: title of the article
    - article_text: the article body or abstract (first 1200 chars used)
    - patient_question: the patient's health question
    - patient_conditions: comma-separated list of conditions
    - patient_medications: comma-separated list of medications
    - patient_age: age string (optional)
    - evidence_tier: 1=NHS/NICE, 2=systematic review, 3=primary research

    Returns JSON ArticleEvidence object with:
    - question_facts: facts that answer the question
    - patient_aligned_facts: facts matched to patient profile
    - contraindications: relevant warnings
    - patient_relevant_summary: concise patient-specific summary
    - alignment_confidence: 0-1 quality score
    """
    from backend.evidence_extractor import _extract_one_article

    source = {
        "source_id": "mcp-extract",
        "title": article_title,
        "snippet": article_text[:1200],
        "evidence_tier": evidence_tier,
        "tier_label": {1: "NHS/NICE", 2: "Systematic Review", 3: "Research"}.get(evidence_tier, "Research"),
    }
    conditions_list = [c.strip() for c in patient_conditions.split(",") if c.strip()]
    meds_list = [m.strip() for m in patient_medications.split(",") if m.strip()]
    patient_summary = (
        f"Age: {patient_age}; "
        f"Conditions: {patient_conditions}; "
        f"Medications: {patient_medications}"
    )

    result = _extract_one_article(
        _llm, source, patient_question, patient_summary, meds_list, conditions_list
    )
    return result.model_dump_json(indent=2)


# ─── Tool: generate_clinical_note ────────────────────────────────────────────

@mcp.tool()
def generate_clinical_note(
    username: str,
    patient_question: str,
    conversation_summary: str,
    urgency_level: str = "routine",
    next_step: str = "",
) -> str:
    """
    Generate a standard SOAP clinical note from a FlynnMed consultation.

    Parameters:
    - username: the patient's account username
    - patient_question: the question they asked
    - conversation_summary: summary of the conversation context
    - urgency_level: routine | elevated | high | urgent | crisis
    - next_step: recommended action (from triage)

    Returns the generated note as JSON and saves it to the patient's record.
    Note fields: note_id, subjective, objective, assessment, plan,
                 urgency_level, requires_gp_visit, gp_visit_reason.
    """
    triage = (
        {"urgency_level": urgency_level, "next_step": next_step}
        if urgency_level != "routine"
        else None
    )

    note = generate_soap_note(
        username=username,
        conversation_summary=conversation_summary,
        question=patient_question,
        triage_summary=triage,
        llm=_llm,
    )
    UserStore.save_clinical_note(username, note)
    return json.dumps(note, default=str, indent=2)


# ─── Tool: send_health_email ─────────────────────────────────────────────────

@mcp.tool()
def send_health_email(
    username: str,
    email_type: str,
    note_id: str = "",
    urgency_level: str = "high",
    reason: str = "",
) -> str:
    """
    Send a health email to the user.

    email_type:
    - "clinical_note" -- sends a saved SOAP note (requires note_id)
    - "urgent_alert"  -- sends an urgent care alert (uses urgency_level + reason)

    Returns {"ok": true, "sent_to": "email"} or {"error": "message"}.
    """
    profile = UserStore.get_user_profile(username)
    if not profile:
        return json.dumps({"error": f"User '{username}' not found"})

    email_address = profile.get("email", "")
    display_name = profile.get("display_name", username)

    if not email_address:
        return json.dumps({"error": "User has no email address on file"})

    try:
        if email_type == "clinical_note":
            notes = UserStore.get_clinical_notes(username)
            note = next((n for n in notes if n["note_id"] == note_id), None)
            if not note:
                return json.dumps({"error": f"Note '{note_id}' not found"})
            send_clinical_note_email(email_address, display_name, note)
            UserStore.mark_note_email_sent(username, note_id)
            return json.dumps({"ok": True, "sent_to": email_address})

        if email_type == "urgent_alert":
            if not reason:
                return json.dumps({"error": "Provide a reason for the urgent alert"})
            send_urgent_care_alert(email_address, display_name, reason, urgency_level)
            return json.dumps({"ok": True, "sent_to": email_address})

        return json.dumps({"error": f"Unknown email_type '{email_type}'. Use 'clinical_note' or 'urgent_alert'"})

    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ─── Tool: search_trials_for_patient ─────────────────────────────────────────

@mcp.tool()
def search_trials_for_patient(
    username: str, location: str = "United Kingdom", max_results: int = 10
) -> str:
    """
    Search ClinicalTrials.gov for recruiting trials matched to a patient's conditions.

    Builds a structured search profile from the patient's saved conditions,
    medications, and symptom logs, then ranks returning trials by profile match.

    Returns JSON array of ranked trial results.
    """
    from backend.clinical_trials import build_trial_search_profile, find_matching_trials

    profile = UserStore.get_user_profile(username)
    if not profile:
        return json.dumps({"error": f"User '{username}' not found"})

    conditions = UserStore.get_conditions(username)
    medications = UserStore.get_medications(username)
    symptom_logs = UserStore.get_symptom_logs(username, limit=20)

    try:
        search_profile = build_trial_search_profile(
            profile=profile,
            memory=UserStore.get_longitudinal_memory(username),
            symptom_logs=symptom_logs,
            conditions=conditions,
            medications=medications,
            allergies=UserStore.get_allergies(username),
            vitals=UserStore.get_vitals(username, limit=None),
            triage_summaries=UserStore.get_triage_summaries(username, limit=None),
            document_summaries=UserStore.get_document_summaries(username),
        )
        results = find_matching_trials(
            search_profile, location_query=location, max_results=max_results
        )
        return json.dumps(results, default=str, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
