"""
Generates a clinician-facing medication proposal: a specific candidate drug
+ dose/frequency, grounded in real retrieved evidence (never the LLM's own
general knowledge), and deterministically cross-checked against the
patient's actual recorded allergies and current medications.

This deliberately reuses the full evidence pipeline (mandatory retrieval,
HyDE expansion, evidence ranking, claim-source-alignment/citation gate) via
RAGEngine.stream_user_question_events' target_patient_data seam, the same
mechanism backend/api.py's previsit_chat already uses -- a specific drug/dose
recommendation needs the same citation-grounding guarantee as any other
clinical claim, not a cheaper unretrieved LLM call.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from backend.clinician_chat_data import load_patient_data_bundle
from backend.medication_checker import MedicationInteractionChecker, check_allergy_conflicts


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _consume_final_payload(rag_engine, question: str, username: str, bundle: Dict) -> Optional[Dict]:
    """Runs the question through the real retrieval+answer pipeline and
    returns the terminal payload -- same consumption pattern as
    backend/api.py's previsit_chat (iterate the generator, discard status/
    token events, keep the `final` event's payload)."""
    payload_final = None
    for event in rag_engine.stream_user_question_events(
        question=question,
        user=username,
        target_patient_data=bundle,
    ):
        if event.get("type") == "final":
            payload_final = event.get("payload")
    return payload_final


def _run_safety_checks(candidate_name: str, medications: List[Dict], allergies: List[Dict]) -> Dict:
    """
    Deterministic checks only -- no LLM involved. Cross-checks the candidate
    against the patient's ACTUAL recorded current medications (openFDA
    pairwise interaction check) and ACTUAL recorded allergies (name/drug-
    class heuristic). Always attached to the draft; never a hard gate here --
    the release endpoint is what decides what to do with unresolved flags.
    """
    checker = MedicationInteractionChecker()
    stored_names = [m.get("name", "") for m in medications if m.get("name")]

    result = checker.check_interactions([candidate_name, *stored_names])
    candidate_lower = candidate_name.strip().lower()
    interaction_flags = [
        alert
        for alert in result.get("alerts", [])
        if candidate_lower in alert.get("pair", "").lower()
    ]

    resolved_candidate = checker.resolve_medication(candidate_name)
    allergy_flags = check_allergy_conflicts(resolved_candidate or {}, allergies) if resolved_candidate else []

    return {
        "allergy_flags": allergy_flags,
        "interaction_flags": interaction_flags,
        "unresolved_medications": result.get("unresolved_medications", []),
        "checked_at": _utc_now_iso(),
    }


def has_unresolved_safety_flags(safety_check: Dict) -> bool:
    """
    Shared derivation used by both the draft-save and release endpoints --
    safety_check is never cached as a separate boolean (see
    ProposedMedication's docstring), always recomputed from this. An allergy
    flag of any kind is always unresolved; an interaction flag only counts at
    high/monitor severity, not a bare `mentioned` label co-occurrence.
    """
    if safety_check.get("allergy_flags"):
        return True
    return any(
        flag.get("severity") in ("high", "monitor")
        for flag in safety_check.get("interaction_flags", [])
    )


def recheck_candidate_safety(db, patient, candidate_medication_name: str) -> Dict:
    """
    Re-runs the deterministic safety checks against the patient's current
    real Medication/Allergy rows for a candidate name -- used whenever a
    clinician edits candidate_medication_name on a draft, or on release,
    since a changed drug name must never be checked against a stale
    safety_check computed for a different drug.
    """
    bundle = load_patient_data_bundle(db, patient)
    return _run_safety_checks(candidate_medication_name, bundle["medications"], bundle["allergies"])


def generate_medication_proposal(
    db,
    patient,
    clinical_situation: str,
    rag_engine,
    username: str,
    clinician_role_label: str,
) -> Dict:
    """
    Returns one of:
      {"status": "insufficient_evidence"} -- retrieval found nothing, or the
        grounded answer didn't actually name a specific drug + dose. Callers
        must not persist a draft in this case, not guess or retry silently.
      {"status": "ok", "clinical_situation_text", "candidate_medication_name",
       "candidate_dose_frequency", "rationale_text", "citations", "trace_id",
       "safety_check"} -- ready to persist as a new draft row.
    """
    bundle = load_patient_data_bundle(db, patient)
    # The target_patient_data seam sources persona/role resolution from
    # user_profile.clinical_role, which is the PATIENT's own account role by
    # default (see backend/rag_system.py's _prepare_answer_bundle docstring)
    # -- appropriate for the patient-facing previsit chat this seam was built
    # for, but wrong here: a medication proposal is drafted for and reviewed
    # by the clinician, so it should use the CLINICIAN's role for persona/
    # citation-format resolution (e.g. the professional decision-format
    # rules), not the patient's. Override locally; load_patient_data_bundle
    # and previsit_chat's own behavior are untouched.
    bundle = dict(bundle)
    bundle["user_profile"] = {**bundle["user_profile"], "clinical_role": clinician_role_label}

    question = (
        "A clinician is considering medication options for this patient given the "
        f"following clinical situation:\n{clinical_situation}\n\n"
        "Research current evidence-based guidance and, if the evidence supports it, "
        "propose ONE specific candidate medication with a specific dose/frequency, "
        "citing the source(s) that support it. If non-pharmacological management is "
        "more appropriate, or no specific medication is clearly indicated by the "
        "evidence, say so explicitly rather than naming a drug."
    )

    payload_final = _consume_final_payload(rag_engine, question, username, bundle)
    if not payload_final or not payload_final.get("sources"):
        return {"status": "insufficient_evidence"}

    answer_markdown = payload_final.get("answer_markdown", "")
    sources = payload_final.get("sources", [])

    extraction = rag_engine.llm.extract_medication_proposal(answer_markdown, sources)
    if not extraction:
        return {"status": "insufficient_evidence"}

    safety_check = _run_safety_checks(
        extraction["medication_name"], bundle["medications"], bundle["allergies"]
    )

    return {
        "status": "ok",
        "clinical_situation_text": clinical_situation,
        "candidate_medication_name": extraction["medication_name"],
        "candidate_dose_frequency": extraction["dose_frequency"],
        "rationale_text": answer_markdown,
        "citations": sources,
        "trace_id": payload_final.get("trace", {}).get("trace_id", ""),
        "safety_check": safety_check,
    }
