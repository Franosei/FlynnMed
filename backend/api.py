from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Generator, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from backend.clinical_notes import generate_previsit_chart_summary, generate_soap_note
from backend.care_plan_agent import CarePlanAgent
from backend.care_plan_store import CarePlanStore
from backend.clinician_access import (
    AccessWorkflowError,
    access_overview,
    authorized_patient_summary,
    decide_access_request,
    request_patient_access,
    require_active_previsit_access,
    revoke_access,
)
from backend.clinician_chat_data import load_patient_data_bundle
from backend.medication_proposal import (
    generate_medication_proposal,
    has_unresolved_safety_flags,
    recheck_candidate_safety,
)
from backend.models.account import Account, AccountKind
from backend.models.audit import AuditAction, AuditLogEntry, AuditOutcome
from backend.models.patient import Patient, PreVisitChatMessage, PreVisitSummary, ProposedMedication
from backend.clinical_context_guard import (
    adjudicate_patient_context,
    build_review_required_plan,
    validate_care_plan,
    validate_generated_answer,
)
from backend.clinical_trials import build_trial_search_profile, find_matching_trials
from backend.query_expander import QueryExpander
from backend.config import DatabaseConfigurationError
from backend.db import get_db, get_session_factory
from backend.email_service import send_clinical_note_email, send_urgent_care_alert
from backend.feedback_store import save_feedback
from backend.fhir.stub_client import fhir_integration_status
from backend.document_analysis_agent import (
    DocumentAnalysisError,
    MAX_DOCUMENT_BYTES,
    SUPPORTED_DOCUMENT_MIME_TYPES,
    normalize_document_mime_type,
)
from backend.image_analysis_agent import (
    ImageAnalysisError,
    MAX_IMAGE_BYTES,
    SUPPORTED_IMAGE_MIME_TYPES,
    normalize_image_mime_type,
)
from backend.intent_risk_classifier import IntentRiskClassifier
from backend.patient_history import build_patient_history_context
from backend.role_router import RoleRouter
from backend.safety_review import build_safety_reviews, update_review_state
from backend.product_config import (
    FOUNDER_NAME,
    PRIVACY_NOTICE_POINTS,
    PRODUCT_NAME,
    PRODUCT_SUBTITLE,
    PRODUCT_TAGLINE,
    ROLE_OPTIONS,
    ROLE_TERMS,
    SUPPORT_EMAIL,
    TERMS_VERSION,
    default_care_context_for_role,
    get_terms_for_role,
    is_clinician_role,
)
from backend.rag_system import RAGEngine
from backend.upload_verification import verify_saved_pdf
from backend.user_store import UserStore
from backend.voice_transcriber import VoiceTranscriber

load_dotenv()

logger = logging.getLogger(__name__)

app = FastAPI(title=f"{PRODUCT_NAME} API")


@app.exception_handler(DatabaseConfigurationError)
async def database_configuration_error(
    request: Request,
    exc: DatabaseConfigurationError,
) -> JSONResponse:
    del request, exc
    return JSONResponse(
        status_code=503,
        content={
            "detail": (
                "Clinician access is unavailable because the secure patient database is not connected. "
                "Set DATABASE_URL, run the database migrations, then restart FlynnMed."
            )
        },
    )


@app.exception_handler(SQLAlchemyError)
async def relational_database_error(
    request: Request,
    exc: SQLAlchemyError,
) -> JSONResponse:
    del request
    logger.error(
        "Relational database request failed",
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return JSONResponse(
        status_code=503,
        content={
            "detail": (
                "Clinician access is unavailable because the secure patient database is not ready. "
                "Check the database connection, apply the current migrations, then restart FlynnMed."
            )
        },
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_RAG_ENGINE: Optional[RAGEngine] = None
_TOKEN_TTL_SECONDS = 60 * 60 * 24 * 7


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _token_secret() -> bytes:
    secret = os.getenv("APP_SECRET") or os.getenv("SECRET_KEY") or "flynnmed-local-dev-secret"
    return secret.encode("utf-8")


def _b64_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def _create_token(username: str) -> str:
    payload = {
        "sub": username.strip().lower(),
        "exp": int(time.time()) + _TOKEN_TTL_SECONDS,
    }
    payload_part = _b64_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(_token_secret(), payload_part.encode("ascii"), hashlib.sha256).digest()
    return f"{payload_part}.{_b64_encode(signature)}"


def _read_token(token: str) -> str:
    try:
        payload_part, signature_part = token.split(".", 1)
        expected_signature = hmac.new(
            _token_secret(),
            payload_part.encode("ascii"),
            hashlib.sha256,
        ).digest()
        actual_signature = _b64_decode(signature_part)
        if not hmac.compare_digest(expected_signature, actual_signature):
            raise ValueError("bad signature")
        payload = json.loads(_b64_decode(payload_part))
        if int(payload.get("exp", 0)) < int(time.time()):
            raise ValueError("expired")
        username = str(payload.get("sub", "")).strip().lower()
        if not username or not UserStore.get_user_profile(username):
            raise ValueError("unknown user")
        return username
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Sign in again to continue.") from exc


def current_user(authorization: str = Header(default="")) -> str:
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Missing access token.")
    return _read_token(token)


def _get_rag_engine() -> RAGEngine:
    global _RAG_ENGINE
    if _RAG_ENGINE is None:
        _RAG_ENGINE = RAGEngine()
    return _RAG_ENGINE


def _json_line(payload: Dict) -> bytes:
    return (json.dumps(payload, default=str) + "\n").encode("utf-8")


def _safe_filename(filename: str) -> str:
    name = Path(filename or "upload.pdf").name
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip()
    return cleaned or "upload.pdf"


def _public_profile(username: str) -> Dict:
    profile = UserStore.get_user_profile(username)
    return {
        "username": username,
        **profile,
    }


def _reconcile_chat_history(username: str, chat_history: List[Dict]) -> List[Dict]:
    """Replace persisted answers that are contradicted by a confirmed test type.

    This keeps a previously generated wrong-specialty answer from continuing to
    appear as if it were trusted clinical guidance after the guard is deployed.
    The original interaction remains in the audit/trace records; the patient
    conversation shows the safe correction instead.
    """
    if not chat_history:
        return chat_history
    conditions = UserStore.get_conditions(username)
    medications = UserStore.get_medications(username)
    vitals = UserStore.get_vitals(username, limit=None)
    allergies = UserStore.get_allergies(username)
    triage = UserStore.get_triage_summaries(username, limit=None)
    documents = UserStore.get_document_summaries(username)
    memory = UserStore.get_longitudinal_memory(username) or {}
    changed = False

    for message in chat_history:
        content = str(message.get("content", ""))
        if message.get("role") != "assistant" or not re.search(
            r"\b(?:peak\s+flow|qmax|uroflow(?:metry)?)\b", content, re.I
        ):
            continue
        decision = adjudicate_patient_context(
            question=content,
            conditions=conditions,
            medications=medications,
            vitals=vitals,
            allergies=allergies,
            triage_summaries=triage,
            document_summaries=documents,
            longitudinal_memory=memory.get("summary", ""),
        )
        validation = validate_generated_answer(content, decision)
        if decision.domain and not validation["valid"]:
            message["content"] = decision.correction_message()
            metadata = dict(message.get("metadata") or {})
            metadata["context_guard_replaced"] = True
            metadata["context_guard_violations"] = validation["violations"]
            message["metadata"] = metadata
            changed = True

    if changed:
        UserStore.set_chat_history(username, chat_history)
    return chat_history


def _snapshot(username: str) -> Dict:
    uploads = UserStore.get_uploads(username)
    symptom_logs = UserStore.get_symptom_logs(username, limit=None)
    medications = UserStore.get_medications(username)
    allergies = UserStore.get_allergies(username)
    conditions = UserStore.get_conditions(username)
    vitals = UserStore.get_vitals(username, limit=None)
    triage_summaries = UserStore.get_triage_summaries(username, limit=None)
    latest_triage = triage_summaries[0] if triage_summaries else {}
    chat_history = _reconcile_chat_history(username, UserStore.get_chat_history(username))
    safety_reviews = build_safety_reviews(
        vitals=vitals,
        symptoms=symptom_logs,
        medications=medications,
        allergies=allergies,
        saved_states=UserStore.get_safety_review_states(username),
    )

    return {
        "product": {
            "name": PRODUCT_NAME,
            "tagline": PRODUCT_TAGLINE,
            "subtitle": PRODUCT_SUBTITLE,
            "support_email": SUPPORT_EMAIL,
        },
        "user": username,
        "profile": _public_profile(username),
        "metrics": {
            "messages": len(chat_history),
            "documents": len(uploads),
            "symptoms": len(symptom_logs),
            "conditions": len(conditions),
            "medications": len(medications),
            "allergies": len(allergies),
            "vitals": len(vitals),
            "triage_records": len(triage_summaries),
        },
        "latest_triage": latest_triage,
        "chat_history": chat_history,
        "uploads": uploads,
        "document_summaries": UserStore.get_document_summaries(username),
        "symptom_logs": symptom_logs,
        "medications": medications,
        "allergies": allergies,
        "conditions": conditions,
        "vitals": vitals,
        "triage_summaries": triage_summaries,
        "traces": UserStore.get_interaction_traces(username, limit=10),
        "audit": UserStore.get_audit(username, limit=20),
        "memory": UserStore.get_longitudinal_memory(username),
        "trial_search_result": UserStore.get_trial_search_result(username),
        "clinical_notes": UserStore.get_clinical_notes(username),
        "safety_reviews": safety_reviews,
    }


class LoginPayload(BaseModel):
    identifier: str
    password: str


class SignupPayload(BaseModel):
    full_name: str
    email: str
    username: str
    role: str
    password: str
    confirm_password: str
    organization: str = ""
    date_of_birth: str = ""
    biological_sex: str = ""
    accept_role_terms: bool = False
    accept_privacy: bool = False


class ProfilePayload(BaseModel):
    display_name: Optional[str] = None
    email: Optional[str] = None
    care_context: Optional[str] = None
    role: Optional[str] = None
    clinical_role: Optional[str] = None
    organization: Optional[str] = None
    follow_up_preferences: Optional[str] = None
    date_of_birth: Optional[str] = None
    biological_sex: Optional[str] = None


class ChatPayload(BaseModel):
    message: str


class FeedbackPayload(BaseModel):
    trace_id: str
    rating: str
    message_id: str = ""


class SymptomPayload(BaseModel):
    symptom: str
    logged_for: str
    severity: int
    triggers: str = ""
    notes: str = ""


class ConditionPayload(BaseModel):
    name: str
    status: str = "active"
    recorded_on: str = ""
    notes: str = ""
    condition_id: str = ""


class MedicationPayload(BaseModel):
    name: str
    dose: str = ""
    schedule: str = ""
    reason: str = ""
    started_on: str = ""
    notes: str = ""
    medication_id: str = ""


class AllergyPayload(BaseModel):
    name: str
    reaction: str = ""
    severity: str = "unknown"
    allergy_type: str = "other"
    confirmed: bool = True
    notes: str = ""
    allergy_id: str = ""


class VitalsPayload(BaseModel):
    type: str
    value: str
    unit: str = ""
    recorded_on: str = ""
    notes: str = ""


class SafetyReviewUpdatePayload(BaseModel):
    status: str
    action_happened: Optional[bool] = None
    patient_improved: Optional[bool] = None
    note: str = ""


class TrialSearchPayload(BaseModel):
    location: str
    max_results: int = 10


class NoteGeneratePayload(BaseModel):
    conversation_summary: str = ""
    question: str = ""
    trace_id: str = ""


class NoteUpdatePayload(BaseModel):
    subjective: Optional[str] = None
    objective: Optional[str] = None
    assessment: Optional[str] = None
    plan: Optional[str] = None
    urgency_level: Optional[str] = None
    requires_gp_visit: Optional[bool] = None
    gp_visit_reason: Optional[str] = None


class EmailNotePayload(BaseModel):
    note_id: str


class UrgentAlertPayload(BaseModel):
    reason: str
    urgency_level: str = "high"


class AccessRequestPayload(BaseModel):
    patient_id: str
    reason: str = ""
    include_chat_history: bool = False


class AccessDecisionPayload(BaseModel):
    approve: bool
    decision_note: str = ""


class PrevisitSummaryDraftPayload(BaseModel):
    summary_text: str


class PrevisitChatPayload(BaseModel):
    message: str


class MedicationProposalGeneratePayload(BaseModel):
    clinical_situation: str


class MedicationProposalDraftPayload(BaseModel):
    clinical_situation_text: str
    candidate_medication_name: str
    candidate_dose_frequency: str
    rationale_text: str
    citations: List[Dict] = []


class MedicationProposalReleasePayload(BaseModel):
    clinical_situation_text: str
    candidate_medication_name: str
    candidate_dose_frequency: str
    rationale_text: str
    citations: List[Dict] = []
    override_reason: str = ""


@app.get("/api/health")
def health() -> Dict:
    return {"ok": True, "product": PRODUCT_NAME}


@app.get("/api/integrations/fhir/status")
def fhir_status() -> Dict:
    """Reports whether a real EHR connection is configured. Always
    {"connected": false} in this phase -- see backend/fhir/ for the
    interface-only SMART-on-FHIR adapter layer this will wire into later."""
    return fhir_integration_status()


@app.get("/api/config")
def config() -> Dict:
    return {
        "product_name": PRODUCT_NAME,
        "product_tagline": PRODUCT_TAGLINE,
        "product_subtitle": PRODUCT_SUBTITLE,
        "founder_name": FOUNDER_NAME,
        "support_email": SUPPORT_EMAIL,
        "terms_version": TERMS_VERSION,
        "role_options": ROLE_OPTIONS,
        "role_terms": ROLE_TERMS,
        "privacy_notice_points": PRIVACY_NOTICE_POINTS,
    }


@app.post("/api/auth/login")
def login(payload: LoginPayload) -> Dict:
    identifier = payload.identifier.strip().lower()
    if not identifier or not payload.password:
        raise HTTPException(status_code=400, detail="Enter your email or username and password.")
    if not UserStore.authenticate(identifier, payload.password):
        raise HTTPException(status_code=401, detail="The email, username, or password is incorrect.")
    username = UserStore.resolve_login_username(identifier)
    if not username:
        raise HTTPException(status_code=401, detail="We could not open your account.")
    UserStore.update_last_login(username)
    return {
        "token": _create_token(username),
        "profile": _public_profile(username),
        "snapshot": _snapshot(username),
    }


@app.post("/api/auth/signup")
def signup(payload: SignupPayload) -> Dict:
    full_name = payload.full_name.strip()
    username = payload.username.strip().lower()
    email = payload.email.strip().lower()
    name_tokens = re.findall(r"[A-Za-z]{2,}", full_name)

    if not full_name or not email or not username or not payload.password or not payload.confirm_password:
        raise HTTPException(status_code=400, detail="Full name, email, username, and password are required.")
    if len(name_tokens) < 2:
        raise HTTPException(status_code=400, detail="Enter your full name with at least first and last name.")
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    if payload.password != payload.confirm_password:
        raise HTTPException(status_code=400, detail="The password and confirmation fields must match.")
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Use a password with at least 8 characters.")
    if payload.role not in ROLE_OPTIONS:
        raise HTTPException(status_code=400, detail="Choose a valid account role.")
    if not payload.accept_role_terms or not payload.accept_privacy:
        raise HTTPException(status_code=400, detail="Accept the role terms and privacy notice before creating the account.")
    if UserStore.resolve_login_username(username):
        raise HTTPException(status_code=409, detail="That username is already taken.")
    if UserStore.resolve_login_username(email):
        raise HTTPException(status_code=409, detail="That email address is already registered.")

    accepted_at = _utc_now()
    created = UserStore.create_user(
        username,
        payload.password,
        display_name=full_name,
        email=email,
        care_context=default_care_context_for_role(payload.role),
        role=payload.role,
        clinical_role=payload.role,
        organization=payload.organization if is_clinician_role(payload.role) else "",
        terms_version=TERMS_VERSION,
        terms_role=payload.role,
        terms_accepted_at=accepted_at,
        privacy_accepted_at=accepted_at,
        date_of_birth=payload.date_of_birth,
        biological_sex=payload.biological_sex,
    )
    if not created:
        raise HTTPException(status_code=400, detail="Account creation failed. Try another username or email.")

    UserStore.update_last_login(username)
    return {
        "token": _create_token(username),
        "profile": _public_profile(username),
        "snapshot": _snapshot(username),
    }


# ---------------------------------------------------------------------------
# Care Plans
# ---------------------------------------------------------------------------

class GeneratePlanPayload(BaseModel):
    condition: str
    chat_summary: str = ""
    # Set when resubmitting after the user answered a "clarify" event -- lets the
    # ambiguity gate resolve via the classifier's continuation-turn logic instead
    # of re-flagging the same ambiguous term forever.
    clarification_question: str = ""
    clarification_answer: str = ""


class TaskTogglePayload(BaseModel):
    done: bool


class AfterVisitPayload(BaseModel):
    note: str


@app.get("/api/care-plans")
def list_care_plans(username: str = Depends(current_user)) -> List[Dict]:
    return _reconcile_care_plans(username)


def _reconcile_care_plans(username: str) -> List[Dict]:
    """Fail closed for plans saved before the context gate existed."""
    plans = CarePlanStore.list_plans(username)
    if not plans:
        return []
    snap = _snapshot(username)
    reconciled: List[Dict] = []
    for plan in plans:
        plan_text = json.dumps(plan, ensure_ascii=False)
        decision = adjudicate_patient_context(
            question=plan_text,
            requested_topic=str(plan.get("condition", "")),
            conditions=snap.get("conditions", []),
            medications=snap.get("medications", []),
            vitals=snap.get("vitals", []),
            allergies=snap.get("allergies", []),
            triage_summaries=snap.get("triage_summaries", []),
            document_summaries=snap.get("document_summaries", []),
            longitudinal_memory=snap.get("memory", {}).get("summary", ""),
        )
        validation = validate_care_plan(plan, decision)
        if not validation["valid"]:
            replacement = build_review_required_plan(decision)
            replacement["id"] = plan.get("id", replacement["id"])
            replacement["created_at"] = plan.get("created_at", replacement["created_at"])
            replacement["validation"]["violations"] = validation["violations"]
            plan = CarePlanStore.save_plan(username, replacement)
        reconciled.append(plan)
    return reconciled


@app.post("/api/care-plans/generate")
def generate_care_plan(
    payload: GeneratePlanPayload,
    username: str = Depends(current_user),
) -> StreamingResponse:
    """Streams NDJSON progress events, then emits either a 'clarify' event (ambiguous
    patient history -- generation withheld pending clarification) or a final 'done'
    event with the plan."""
    profile = UserStore.get_user_profile(username) or {}
    snap = _snapshot(username)
    longitudinal_memory = _get_rag_engine().get_combined_longitudinal_memory(username)

    patient_history = build_patient_history_context(
        longitudinal_memory=longitudinal_memory,
        medications=snap.get("medications", []),
        triage_summaries=snap.get("triage_summaries", []),
        user_profile=profile,
        allergies=snap.get("allergies", []),
        conditions=snap.get("conditions", []),
        vitals=snap.get("vitals", []),
    )
    clinical_context = adjudicate_patient_context(
        question=f"Create a care plan for: {payload.condition}",
        requested_topic=payload.condition,
        conditions=snap.get("conditions", []),
        medications=snap.get("medications", []),
        vitals=snap.get("vitals", []),
        allergies=snap.get("allergies", []),
        triage_summaries=snap.get("triage_summaries", []),
        document_summaries=snap.get("document_summaries", []),
        longitudinal_memory=longitudinal_memory,
        chat_summary=payload.chat_summary,
    )

    user_context = {
        "profile": profile,
        "medications": snap.get("medications", []),
        "conditions": snap.get("conditions", []),
        "allergies": snap.get("allergies", []),
        "vitals": snap.get("vitals", []),
        "triage_summaries": snap.get("triage_summaries", []),
        "longitudinal_memory": longitudinal_memory,
        "chat_summary": payload.chat_summary,
        "patient_history": patient_history,
        "clinical_context": clinical_context,
    }

    def stream() -> Generator[str, None, None]:
        if clinical_context.requires_clarification and not (
            payload.clarification_question and payload.clarification_answer
        ):
            yield json.dumps({
                "type": "clarify",
                "question": clinical_context.clarifying_question,
                "options": clinical_context.clarification_options,
            }) + "\n"
            return

        # Ambiguity gate: stop and ask before generating a plan around a guessed
        # meaning of a cross-specialty term (e.g. "peak flow") the patient's own
        # history doesn't already resolve.
        recent_turns = None
        if payload.clarification_question and payload.clarification_answer:
            # Matches the exact shape IntentRiskClassifier._llm_classify's
            # continuation_block looks for: the last assistant turn asking the
            # clarifying question, the last user turn answering it. Without this,
            # the classifier has no signal the ambiguity was already resolved and
            # just re-detects the same ambiguous term every time.
            recent_turns = [
                {"role": "assistant", "content": payload.clarification_question},
                {"role": "user", "content": payload.clarification_answer},
            ]

        try:
            role_config = RoleRouter().resolve(profile.get("clinical_role") or profile.get("role", ""))
            intent = IntentRiskClassifier().classify(
                question=(
                    f"Create a care plan for: {payload.condition}\n\n"
                    f"Recent context:\n{payload.chat_summary[-600:]}"
                ),
                user_profile=profile,
                role_key=role_config.role_key,
                patient_history=patient_history,
                recent_turns=recent_turns,
            )
        except Exception as exc:
            print(f"[CarePlan] Ambiguity check failed (non-fatal): {exc}")
            intent = None

        if intent and intent.ambiguous_term_detected:
            yield json.dumps({
                "type": "clarify",
                "question": intent.ambiguity_clarifying_question,
                "options": intent.ambiguity_reply_options,
            }) + "\n"
            return

        agent = CarePlanAgent()
        plan: Dict = {}
        error_msg = ""

        # We can't yield from a nested callback directly in a generator, so
        # we collect events via a list and flush them in the loop.
        progress_events: List[str] = []

        def collect(msg: str) -> None:
            progress_events.append(msg)

        try:
            plan = agent.generate(payload.condition, user_context, on_progress=collect)
        except Exception as exc:
            error_msg = str(exc)

        # First flush any collected progress messages
        for msg in progress_events:
            yield json.dumps({"type": "progress", "message": msg}) + "\n"

        if error_msg:
            yield json.dumps({"type": "error", "message": error_msg}) + "\n"
            return

        # Persist and return
        saved = CarePlanStore.save_plan(username, plan)
        yield json.dumps({"type": "done", "plan": saved}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/api/care-plans/{plan_id}")
def get_care_plan(plan_id: str, username: str = Depends(current_user)) -> Dict:
    plan = next((item for item in _reconcile_care_plans(username) if item.get("id") == plan_id), None)
    if not plan:
        raise HTTPException(status_code=404, detail="Care plan not found.")
    return plan


@app.delete("/api/care-plans/{plan_id}")
def delete_care_plan(plan_id: str, username: str = Depends(current_user)) -> Dict:
    if not CarePlanStore.delete_plan(username, plan_id):
        raise HTTPException(status_code=404, detail="Care plan not found.")
    return {"ok": True}


@app.patch("/api/care-plans/{plan_id}/tasks/{task_id}")
def toggle_task(
    plan_id: str,
    task_id: str,
    payload: TaskTogglePayload,
    username: str = Depends(current_user),
) -> Dict:
    plan = CarePlanStore.toggle_task(username, plan_id, task_id, payload.done)
    if not plan:
        raise HTTPException(status_code=404, detail="Plan or task not found.")
    return plan


@app.post("/api/care-plans/{plan_id}/after-visit")
def after_visit_note(
    plan_id: str,
    payload: AfterVisitPayload,
    username: str = Depends(current_user),
) -> Dict:
    plan = CarePlanStore.add_after_visit_note(username, plan_id, payload.note)
    if not plan:
        raise HTTPException(status_code=404, detail="Care plan not found.")
    return plan


@app.post("/api/care-plans/{plan_id}/gp-prep")
def gp_prep(plan_id: str, username: str = Depends(current_user)) -> Dict:
    plan = next((item for item in _reconcile_care_plans(username) if item.get("id") == plan_id), None)
    if not plan:
        raise HTTPException(status_code=404, detail="Care plan not found.")
    profile = UserStore.get_user_profile(username) or {}
    snap = _snapshot(username)
    agent = CarePlanAgent()
    user_context = {
        "profile": profile,
        "medications": snap.get("medications", []),
        "conditions": snap.get("conditions", []),
        "allergies": snap.get("allergies", []),
        "vitals": snap.get("vitals", []),
        "triage_summaries": snap.get("triage_summaries", []),
        "longitudinal_memory": _get_rag_engine().get_combined_longitudinal_memory(username),
    }
    summary = agent.generate_gp_prep(plan, user_context)
    updated = CarePlanStore.set_gp_prep(username, plan_id, summary)
    return {"gp_prep_summary": summary, "plan": updated}


@app.get("/api/me")
def me(username: str = Depends(current_user)) -> Dict:
    return {"profile": _public_profile(username), "snapshot": _snapshot(username)}


@app.get("/api/snapshot")
def snapshot(username: str = Depends(current_user)) -> Dict:
    return _snapshot(username)


@app.patch("/api/safety-reviews/{review_id}")
def update_safety_review(
    review_id: str,
    payload: SafetyReviewUpdatePayload,
    username: str = Depends(current_user),
) -> Dict:
    reviews = {item["review_id"]: item for item in _snapshot(username)["safety_reviews"]}
    if review_id not in reviews:
        raise HTTPException(status_code=404, detail="This safety review is no longer active.")
    current = UserStore.get_safety_review_states(username).get(review_id)
    try:
        state = update_review_state(current, payload.dict())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    UserStore.save_safety_review_state(username, review_id, state)
    updated_snapshot = _snapshot(username)
    updated_review = next(
        item for item in updated_snapshot["safety_reviews"]
        if item["review_id"] == review_id
    )
    return {"review": updated_review, "snapshot": updated_snapshot}


# ---------------------------------------------------------------------------
# Consent-based clinician access
# ---------------------------------------------------------------------------


def _access_error(db: Session, exc: AccessWorkflowError) -> HTTPException:
    # Access denials are intentionally audit-logged by the service. Commit the
    # denial record before FastAPI's database dependency rolls back on the
    # HTTPException.
    db.commit()
    return HTTPException(status_code=403, detail=str(exc))


@app.get("/api/access")
def get_access_overview(
    username: str = Depends(current_user),
    db: Session = Depends(get_db),
) -> Dict:
    try:
        return access_overview(db, username)
    except AccessWorkflowError as exc:
        raise _access_error(db, exc) from exc


@app.post("/api/access/requests")
def create_access_request(
    payload: AccessRequestPayload,
    username: str = Depends(current_user),
    db: Session = Depends(get_db),
) -> Dict:
    try:
        return request_patient_access(
            db,
            username,
            payload.patient_id,
            payload.reason,
            payload.include_chat_history,
        )
    except AccessWorkflowError as exc:
        raise _access_error(db, exc) from exc


@app.post("/api/access/requests/{grant_id}/decision")
def decide_access(
    grant_id: str,
    payload: AccessDecisionPayload,
    username: str = Depends(current_user),
    db: Session = Depends(get_db),
) -> Dict:
    try:
        return decide_access_request(
            db,
            username,
            grant_id,
            payload.approve,
            payload.decision_note,
        )
    except AccessWorkflowError as exc:
        raise _access_error(db, exc) from exc


@app.delete("/api/access/requests/{grant_id}")
def delete_access(
    grant_id: str,
    username: str = Depends(current_user),
    db: Session = Depends(get_db),
) -> Dict:
    try:
        return revoke_access(db, username, grant_id)
    except AccessWorkflowError as exc:
        raise _access_error(db, exc) from exc


@app.get("/api/clinician/patients/{patient_id}")
def clinician_patient_summary(
    patient_id: str,
    username: str = Depends(current_user),
    db: Session = Depends(get_db),
) -> Dict:
    try:
        return authorized_patient_summary(db, username, patient_id)
    except AccessWorkflowError as exc:
        raise _access_error(db, exc) from exc


# ---------------------------------------------------------------------------
# Clinician pre-visit summary + inline patient-scoped chat
# ---------------------------------------------------------------------------


def _previsit_summary_dict(row: PreVisitSummary) -> Dict:
    return {
        "id": str(row.id),
        "status": row.status,
        "generation_trigger": row.generation_trigger,
        "summary_text": row.summary_text,
        "authored_by_display_name": row.authored_by_display_name,
        "authored_by_clinical_role": row.authored_by_clinical_role,
        "authored_by_organization": row.authored_by_organization,
        "released_at": row.released_at.isoformat() if row.released_at else "",
        "released_by_display_name": row.released_by_display_name,
        "released_by_clinical_role": row.released_by_clinical_role,
        "created_at": row.created_at.isoformat(),
    }


def _previsit_chat_message_dict(row: PreVisitChatMessage) -> Dict:
    return {
        "id": str(row.id),
        "role": row.role,
        "content": row.content,
        "authored_by_display_name": row.authored_by_display_name,
        "authored_by_clinical_role": row.authored_by_clinical_role,
        "sources": row.sources,
        "created_at": row.created_at.isoformat(),
    }


def _clinician_role_label(clinician: Account) -> str:
    return clinician.clinical_role or clinician.role_label or "healthcare_professional"


@app.post("/api/clinician/patients/{patient_id}/summary")
def generate_previsit_summary(
    patient_id: str,
    username: str = Depends(current_user),
    db: Session = Depends(get_db),
) -> Dict:
    """Generates or regenerates the AI-suggested pre-visit summary draft.
    Same endpoint serves both -- "regenerate" is just calling this again
    once a chat transcript / prior draft exists, so the model builds on
    what's already there rather than starting fresh."""
    try:
        clinician, patient, grant = require_active_previsit_access(
            db, username, patient_id, action=AuditAction.clinician_generate_previsit_summary
        )
    except AccessWorkflowError as exc:
        raise _access_error(db, exc) from exc

    bundle = load_patient_data_bundle(db, patient)

    latest_draft = db.execute(
        select(PreVisitSummary)
        .where(PreVisitSummary.patient_id == patient.id, PreVisitSummary.status == "draft")
        .order_by(PreVisitSummary.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()

    transcript_rows = db.execute(
        select(PreVisitChatMessage)
        .where(PreVisitChatMessage.patient_id == patient.id)
        .order_by(PreVisitChatMessage.timestamp.asc())
    ).scalars().all()

    role_key = _clinician_role_label(clinician)
    summary_text = generate_previsit_chart_summary(
        patient_chart=bundle,
        chat_transcript=[{"role": t.role, "content": t.content} for t in transcript_rows],
        clinician_role_key=role_key,
        llm=_get_rag_engine().llm,
        previous_draft=latest_draft.summary_text if latest_draft else None,
    )

    row = PreVisitSummary(
        id=uuid.uuid4(),
        patient_id=patient.id,
        status="draft",
        generation_trigger="ai_generated",
        summary_text=summary_text,
        authored_by_account_id=clinician.id,
        authored_by_display_name=clinician.display_name,
        authored_by_clinical_role=role_key,
        authored_by_organization=clinician.organization,
        consent_grant_id=grant.id,
    )
    db.add(row)
    # Flush (not commit) -- populates the Python-side TimestampMixin
    # defaults/generated id on `row` within the current transaction; the
    # actual commit happens once at the end of the request via get_db()'s
    # dependency (commit-on-success, rollback-on-error), matching how every
    # other DB-backed route in this codebase already works.
    db.flush()
    return _previsit_summary_dict(row)


@app.patch("/api/clinician/patients/{patient_id}/summary/draft")
def save_previsit_summary_draft(
    patient_id: str,
    payload: PrevisitSummaryDraftPayload,
    username: str = Depends(current_user),
    db: Session = Depends(get_db),
) -> Dict:
    """Persists the clinician's hand-edited draft as a new append-only row.
    Frontend should call this only on an explicit "Save draft" click, not
    per-keystroke, to avoid flooding the append-only table."""
    try:
        clinician, patient, grant = require_active_previsit_access(
            db, username, patient_id, action=AuditAction.clinician_edit_previsit_summary_draft
        )
    except AccessWorkflowError as exc:
        raise _access_error(db, exc) from exc

    role_key = _clinician_role_label(clinician)
    row = PreVisitSummary(
        id=uuid.uuid4(),
        patient_id=patient.id,
        status="draft",
        generation_trigger="clinician_edited",
        summary_text=payload.summary_text,
        authored_by_account_id=clinician.id,
        authored_by_display_name=clinician.display_name,
        authored_by_clinical_role=role_key,
        authored_by_organization=clinician.organization,
        consent_grant_id=grant.id,
    )
    db.add(row)
    # Flush (not commit) -- populates the Python-side TimestampMixin
    # defaults/generated id on `row` within the current transaction; the
    # actual commit happens once at the end of the request via get_db()'s
    # dependency (commit-on-success, rollback-on-error), matching how every
    # other DB-backed route in this codebase already works.
    db.flush()
    return _previsit_summary_dict(row)


@app.post("/api/clinician/patients/{patient_id}/summary/release")
def release_previsit_summary(
    patient_id: str,
    payload: PrevisitSummaryDraftPayload,
    username: str = Depends(current_user),
    db: Session = Depends(get_db),
) -> Dict:
    """The only code path anywhere that writes a status="released" row --
    nothing reaches the patient's portal until this explicit, separate
    action is called. Body carries the summary text so any last-second edit
    not yet saved as a draft row is captured at release time too."""
    try:
        clinician, patient, grant = require_active_previsit_access(
            db, username, patient_id, action=AuditAction.clinician_release_previsit_summary
        )
    except AccessWorkflowError as exc:
        raise _access_error(db, exc) from exc

    role_key = _clinician_role_label(clinician)
    now = datetime.now(timezone.utc)
    row = PreVisitSummary(
        id=uuid.uuid4(),
        patient_id=patient.id,
        status="released",
        generation_trigger="released",
        summary_text=payload.summary_text,
        authored_by_account_id=clinician.id,
        authored_by_display_name=clinician.display_name,
        authored_by_clinical_role=role_key,
        authored_by_organization=clinician.organization,
        consent_grant_id=grant.id,
        released_at=now,
        released_by_account_id=clinician.id,
        released_by_display_name=clinician.display_name,
        released_by_clinical_role=role_key,
    )
    db.add(row)
    # Flush (not commit) -- populates the Python-side TimestampMixin
    # defaults/generated id on `row` within the current transaction; the
    # actual commit happens once at the end of the request via get_db()'s
    # dependency (commit-on-success, rollback-on-error), matching how every
    # other DB-backed route in this codebase already works.
    db.flush()
    return _previsit_summary_dict(row)


@app.get("/api/clinician/patients/{patient_id}/chat")
def get_previsit_chat(
    patient_id: str,
    username: str = Depends(current_user),
    db: Session = Depends(get_db),
) -> Dict:
    try:
        _clinician, patient, _grant = require_active_previsit_access(
            db, username, patient_id, action=AuditAction.clinician_chat_previsit
        )
    except AccessWorkflowError as exc:
        raise _access_error(db, exc) from exc

    rows = db.execute(
        select(PreVisitChatMessage)
        .where(PreVisitChatMessage.patient_id == patient.id)
        .order_by(PreVisitChatMessage.timestamp.asc())
    ).scalars().all()
    return {"messages": [_previsit_chat_message_dict(r) for r in rows]}


def _save_previsit_chat_message(
    *,
    patient_id: uuid.UUID,
    role: str,
    content: str,
    authored_by_account_id: Optional[uuid.UUID],
    authored_by_display_name: str,
    authored_by_clinical_role: str,
    consent_grant_id: Optional[uuid.UUID],
    trace_id: str = "",
    sources: Optional[List[Dict]] = None,
) -> None:
    # Opens its own session rather than reusing the request-scoped `db` from
    # the outer endpoint -- this runs from inside a StreamingResponse
    # generator, and a fresh, short-lived session here is simpler to reason
    # about than relying on FastAPI's yield-dependency lifetime for the
    # duration of a stream, matching the per-call-session convention
    # UserStore/CarePlanStore already use elsewhere in this codebase.
    session_factory = get_session_factory()
    with session_factory() as write_db:
        write_db.add(
            PreVisitChatMessage(
                id=uuid.uuid4(),
                patient_id=patient_id,
                role=role,
                content=content,
                authored_by_account_id=authored_by_account_id,
                authored_by_display_name=authored_by_display_name,
                authored_by_clinical_role=authored_by_clinical_role,
                consent_grant_id=consent_grant_id,
                trace_id=trace_id or None,
                sources=sources or [],
                timestamp=datetime.now(timezone.utc),
            )
        )
        write_db.commit()


@app.post("/api/clinician/patients/{patient_id}/chat")
def previsit_chat(
    patient_id: str,
    payload: PrevisitChatPayload,
    username: str = Depends(current_user),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """
    Streaming, patient-scoped clinician chat -- reuses the full evidence
    pipeline (mandatory retrieval, HyDE expansion, evidence ranking,
    claim-verification) via target_patient_data rather than a cheaper
    bypass, same quality bar as the patient-facing chat.
    """
    question = payload.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Enter a message before sending.")

    try:
        clinician, patient, grant = require_active_previsit_access(
            db, username, patient_id, action=AuditAction.clinician_chat_previsit
        )
    except AccessWorkflowError as exc:
        raise _access_error(db, exc) from exc

    bundle = load_patient_data_bundle(db, patient)
    transcript_rows = db.execute(
        select(PreVisitChatMessage)
        .where(PreVisitChatMessage.patient_id == patient.id)
        .order_by(PreVisitChatMessage.timestamp.asc())
    ).scalars().all()
    chat_history = [{"role": t.role, "content": t.content} for t in transcript_rows]

    role_key = _clinician_role_label(clinician)
    # The target_patient_data seam sources persona/role resolution from
    # user_profile.clinical_role, which is the PATIENT's own account role by
    # default (see rag_system.py's _prepare_answer_bundle docstring) -- wrong
    # here, this chat is asked by and answered for the clinician. Same fix
    # already applied in medication_proposal.py's generate_medication_proposal;
    # load_patient_data_bundle itself is untouched.
    bundle = dict(bundle)
    bundle["user_profile"] = {**bundle["user_profile"], "clinical_role": role_key}
    now = _utc_now()

    _save_previsit_chat_message(
        patient_id=patient.id,
        role="clinician",
        content=question,
        authored_by_account_id=clinician.id,
        authored_by_display_name=clinician.display_name,
        authored_by_clinical_role=role_key,
        consent_grant_id=grant.id,
    )

    def generate() -> Generator[bytes, None, None]:
        yield _json_line(
            {
                "type": "user_message",
                "message": {"role": "clinician", "content": question, "timestamp": now},
            }
        )
        try:
            rag_engine = _get_rag_engine()
            payload_final = None
            for event in rag_engine.stream_user_question_events(
                question=question,
                chat_history=chat_history,
                user=username,
                target_patient_data=bundle,
            ):
                event_type = event.get("type")
                if event_type == "status":
                    yield _json_line({"type": "status", "message": event.get("message", "Working...")})
                elif event_type == "token":
                    yield _json_line({"type": "token", "delta": event.get("delta", "")})
                elif event_type == "final":
                    payload_final = event.get("payload")

            if not payload_final:
                raise RuntimeError("The answer pipeline did not return a payload.")

            assistant_content = payload_final["answer_markdown"]
            _save_previsit_chat_message(
                patient_id=patient.id,
                role="assistant",
                content=assistant_content,
                authored_by_account_id=None,
                authored_by_display_name="FlynnMed",
                authored_by_clinical_role="",
                consent_grant_id=grant.id,
                trace_id=payload_final.get("trace", {}).get("trace_id", ""),
                sources=payload_final.get("sources", []),
            )
            yield _json_line(
                {
                    "type": "assistant_message",
                    "message": {
                        "role": "assistant",
                        "content": assistant_content,
                        "sources": payload_final.get("sources", []),
                        "follow_up_questions": payload_final.get("follow_up_questions", []),
                    },
                }
            )
            yield _json_line({"type": "done"})
        except Exception as exc:
            yield _json_line({"type": "error", "message": str(exc)})

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# ---------------------------------------------------------------------------
# Clinician medication proposals
# ---------------------------------------------------------------------------


def _proposed_medication_dict(row: ProposedMedication) -> Dict:
    return {
        "id": str(row.id),
        "status": row.status,
        "generation_trigger": row.generation_trigger,
        "clinical_situation_text": row.clinical_situation_text,
        "candidate_medication_name": row.candidate_medication_name,
        "candidate_dose_frequency": row.candidate_dose_frequency,
        "rationale_text": row.rationale_text,
        "citations": row.citations,
        "safety_check": row.safety_check,
        "override_reason": row.override_reason or "",
        "authored_by_display_name": row.authored_by_display_name,
        "authored_by_clinical_role": row.authored_by_clinical_role,
        "authored_by_organization": row.authored_by_organization,
        "released_at": row.released_at.isoformat() if row.released_at else "",
        "released_by_display_name": row.released_by_display_name,
        "released_by_clinical_role": row.released_by_clinical_role,
        "created_at": row.created_at.isoformat(),
    }


@app.post("/api/clinician/patients/{patient_id}/medication-proposals")
def generate_medication_proposal_endpoint(
    patient_id: str,
    payload: MedicationProposalGeneratePayload,
    username: str = Depends(current_user),
    db: Session = Depends(get_db),
) -> Dict:
    """
    Generates a new AI-drafted medication proposal: real retrieval-grounded
    candidate + dose, deterministically safety-checked against the patient's
    actual current medications/allergies. Draft only -- nothing here is ever
    visible to the patient until an explicit release (see release below).
    """
    situation = payload.clinical_situation.strip()
    if not situation:
        raise HTTPException(status_code=400, detail="Describe the clinical situation.")

    try:
        clinician, patient, grant = require_active_previsit_access(
            db, username, patient_id, action=AuditAction.clinician_generate_medication_proposal
        )
    except AccessWorkflowError as exc:
        raise _access_error(db, exc) from exc

    role_key = _clinician_role_label(clinician)
    result = generate_medication_proposal(
        db=db,
        patient=patient,
        clinical_situation=situation,
        rag_engine=_get_rag_engine(),
        username=username,
        clinician_role_label=role_key,
    )
    if result.get("status") != "ok":
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not find evidence supporting a specific medication for this "
                "situation. Try describing it more specifically, or this may not be "
                "a case with a clear pharmacological option."
            ),
        )

    row = ProposedMedication(
        id=uuid.uuid4(),
        patient_id=patient.id,
        status="draft",
        generation_trigger="ai_generated",
        clinical_situation_text=result["clinical_situation_text"],
        candidate_medication_name=result["candidate_medication_name"],
        candidate_dose_frequency=result["candidate_dose_frequency"],
        rationale_text=result["rationale_text"],
        citations=result["citations"],
        safety_check=result["safety_check"],
        trace_id=result.get("trace_id") or None,
        authored_by_account_id=clinician.id,
        authored_by_display_name=clinician.display_name,
        authored_by_clinical_role=role_key,
        authored_by_organization=clinician.organization,
        consent_grant_id=grant.id,
    )
    db.add(row)
    db.flush()
    return _proposed_medication_dict(row)


@app.patch("/api/clinician/patients/{patient_id}/medication-proposals/draft")
def save_medication_proposal_draft(
    patient_id: str,
    payload: MedicationProposalDraftPayload,
    username: str = Depends(current_user),
    db: Session = Depends(get_db),
) -> Dict:
    """
    Persists the clinician's hand-edited draft as a new append-only row.
    Unlike a plain-text summary edit, an edited candidate_medication_name
    must be re-checked against the patient's real medications/allergies
    server-side here -- a stale safety_check computed for the AI's original
    suggestion must never carry over to a clinician-changed drug name.
    """
    try:
        clinician, patient, grant = require_active_previsit_access(
            db, username, patient_id, action=AuditAction.clinician_edit_medication_proposal_draft
        )
    except AccessWorkflowError as exc:
        raise _access_error(db, exc) from exc

    role_key = _clinician_role_label(clinician)
    safety_check = recheck_candidate_safety(db, patient, payload.candidate_medication_name.strip())

    row = ProposedMedication(
        id=uuid.uuid4(),
        patient_id=patient.id,
        status="draft",
        generation_trigger="clinician_edited",
        clinical_situation_text=payload.clinical_situation_text,
        candidate_medication_name=payload.candidate_medication_name.strip(),
        candidate_dose_frequency=payload.candidate_dose_frequency.strip(),
        rationale_text=payload.rationale_text,
        citations=payload.citations,
        safety_check=safety_check,
        authored_by_account_id=clinician.id,
        authored_by_display_name=clinician.display_name,
        authored_by_clinical_role=role_key,
        authored_by_organization=clinician.organization,
        consent_grant_id=grant.id,
    )
    db.add(row)
    db.flush()
    return _proposed_medication_dict(row)


@app.post("/api/clinician/patients/{patient_id}/medication-proposals/release")
def release_medication_proposal(
    patient_id: str,
    payload: MedicationProposalReleasePayload,
    username: str = Depends(current_user),
    db: Session = Depends(get_db),
) -> Dict:
    """
    The only code path anywhere that writes a status="released" row --
    nothing reaches the patient's portal until this explicit, separate
    action is called. Recomputes safety_check fresh from the submitted
    candidate against the patient's actual current records server-side
    (never trusts a client-supplied safety_check blob) and requires a
    documented override_reason if unresolved flags remain.
    """
    try:
        clinician, patient, grant = require_active_previsit_access(
            db, username, patient_id, action=AuditAction.clinician_release_medication_proposal
        )
    except AccessWorkflowError as exc:
        raise _access_error(db, exc) from exc

    candidate_name = payload.candidate_medication_name.strip()
    safety_check = recheck_candidate_safety(db, patient, candidate_name)
    override_reason = payload.override_reason.strip()
    if has_unresolved_safety_flags(safety_check) and not override_reason:
        raise HTTPException(
            status_code=400,
            detail=(
                "This candidate has an unresolved allergy or interaction flag. "
                "Enter an override reason to release it anyway."
            ),
        )

    role_key = _clinician_role_label(clinician)
    now = datetime.now(timezone.utc)
    row = ProposedMedication(
        id=uuid.uuid4(),
        patient_id=patient.id,
        status="released",
        generation_trigger="released",
        clinical_situation_text=payload.clinical_situation_text,
        candidate_medication_name=candidate_name,
        candidate_dose_frequency=payload.candidate_dose_frequency.strip(),
        rationale_text=payload.rationale_text,
        citations=payload.citations,
        safety_check=safety_check,
        override_reason=override_reason or None,
        authored_by_account_id=clinician.id,
        authored_by_display_name=clinician.display_name,
        authored_by_clinical_role=role_key,
        authored_by_organization=clinician.organization,
        consent_grant_id=grant.id,
        released_at=now,
        released_by_account_id=clinician.id,
        released_by_display_name=clinician.display_name,
        released_by_clinical_role=role_key,
    )
    db.add(row)
    db.flush()
    return _proposed_medication_dict(row)


@app.get("/api/my-medication-proposals")
def list_my_medication_proposals(
    username: str = Depends(current_user),
    db: Session = Depends(get_db),
) -> Dict:
    """Patient-side read -- only ever returns status="released" rows for the
    caller's own patient record, newest first."""
    account = db.execute(
        select(Account).where(Account.username == username.strip().lower())
    ).scalar_one_or_none()
    if account is None or account.account_kind != AccountKind.patient:
        raise HTTPException(status_code=403, detail="Patient account required.")
    patient = db.execute(
        select(Patient).where(Patient.account_id == account.id)
    ).scalar_one_or_none()
    if patient is None:
        return {"proposals": []}

    db.add(
        AuditLogEntry(
            actor_account_id=account.id,
            actor_role_at_time=account.account_kind.value,
            patient_id=patient.id,
            action=AuditAction.patient_read_self,
            resource_type="patient",
            resource_id=patient.patient_id,
            outcome=AuditOutcome.success,
        )
    )
    db.flush()

    rows = db.execute(
        select(ProposedMedication)
        .where(ProposedMedication.patient_id == patient.id, ProposedMedication.status == "released")
        .order_by(ProposedMedication.created_at.desc())
    ).scalars().all()
    return {"proposals": [_proposed_medication_dict(r) for r in rows]}


@app.get("/api/previsit-summaries")
def list_my_previsit_summaries(
    username: str = Depends(current_user),
    db: Session = Depends(get_db),
) -> Dict:
    """Patient-side read -- only ever returns status="released" rows for the
    caller's own patient record, newest first. Kept separate from the hot
    _snapshot() payload (refetched after nearly every action app-wide)
    rather than folded in, mirroring the existing list_notes/get_note
    precedent for ClinicalNote."""
    account = db.execute(
        select(Account).where(Account.username == username.strip().lower())
    ).scalar_one_or_none()
    if account is None or account.account_kind != AccountKind.patient:
        raise HTTPException(status_code=403, detail="Patient account required.")
    patient = db.execute(
        select(Patient).where(Patient.account_id == account.id)
    ).scalar_one_or_none()
    if patient is None:
        return {"summaries": []}

    db.add(
        AuditLogEntry(
            actor_account_id=account.id,
            actor_role_at_time=account.account_kind.value,
            patient_id=patient.id,
            action=AuditAction.patient_read_self,
            resource_type="patient",
            resource_id=patient.patient_id,
            outcome=AuditOutcome.success,
        )
    )
    db.flush()

    rows = db.execute(
        select(PreVisitSummary)
        .where(PreVisitSummary.patient_id == patient.id, PreVisitSummary.status == "released")
        .order_by(PreVisitSummary.created_at.desc())
    ).scalars().all()
    return {"summaries": [_previsit_summary_dict(r) for r in rows]}


@app.put("/api/profile")
def update_profile(payload: ProfilePayload, username: str = Depends(current_user)) -> Dict:
    updates = {key: value for key, value in payload.dict().items() if value is not None}
    if not UserStore.update_profile(username, updates):
        raise HTTPException(status_code=400, detail="Profile update failed.")
    _get_rag_engine().restore_user_context(username)
    return _snapshot(username)


@app.delete("/api/chat")
def clear_chat(username: str = Depends(current_user)) -> Dict:
    UserStore.clear_chat_history(username)
    return _snapshot(username)


@app.post("/api/chat/stream")
def stream_chat(payload: ChatPayload, username: str = Depends(current_user)) -> StreamingResponse:
    question = payload.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Enter a message before sending.")

    def generate() -> Generator[bytes, None, None]:
        try:
            rag_engine = _get_rag_engine()
            chat_history = _reconcile_chat_history(username, UserStore.get_chat_history(username))
        except Exception as exc:
            yield _json_line({"type": "error", "message": f"Failed to start the answer pipeline: {exc}"})
            yield _json_line({"type": "done"})
            return

        now = _utc_now()
        user_entry = {
            "role": "user",
            "content": question,
            "timestamp": now,
            "sources": [],
            "metadata": {},
        }
        UserStore.append_chat(username, user_entry)
        chat_history.append(user_entry)
        yield _json_line({"type": "user_message", "message": user_entry})

        try:
            payload_final = None
            streamed_answer_parts: List[str] = []
            for event in rag_engine.stream_user_question_events(
                question=question,
                chat_history=chat_history,
                user=username,
            ):
                event_type = event.get("type")
                if event_type == "status":
                    yield _json_line({"type": "status", "message": event.get("message", "Working...")})
                elif event_type == "token":
                    delta = event.get("delta", "")
                    streamed_answer_parts.append(delta)
                    yield _json_line({"type": "token", "delta": delta})
                elif event_type == "final":
                    payload_final = event.get("payload")

            if not payload_final:
                raise RuntimeError("The answer pipeline did not return a payload.")

            image_b64 = ""
            if payload_final.get("image_bytes"):
                image_b64 = base64.b64encode(payload_final["image_bytes"]).decode("ascii")

            assistant_entry = {
                "role": "assistant",
                "content": payload_final["answer_markdown"],
                "timestamp": _utc_now(),
                "sources": payload_final.get("sources", []),
                "trace_id": payload_final.get("trace", {}).get("trace_id"),
                "metadata": {
                    "personal_context": payload_final.get("personal_context", []),
                    "longitudinal_memory": payload_final.get("longitudinal_memory", ""),
                    "triage_summary": payload_final.get("triage_summary", {}),
                    "medication_alerts": payload_final.get("medication_alerts", []),
                    "resolved_medications": payload_final.get("resolved_medications", []),
                    "trace": payload_final.get("trace", {}),
                    "image_url": payload_final.get("image_url", ""),
                    "image_b64": image_b64,
                    "image_caption": payload_final.get("image_caption", ""),
                    "video_url": payload_final.get("video_url", ""),
                    "video_caption": payload_final.get("video_caption", ""),
                    "video_rate_limit_msg": payload_final.get("video_rate_limit_msg", ""),
                    "follow_up_questions": payload_final.get("follow_up_questions", []),
                },
            }

            try:
                refreshed_memory = rag_engine.refresh_longitudinal_memory_from_turn(
                    user=username,
                    user_message=question,
                    personal_context=payload_final.get("personal_context", []),
                )
                if refreshed_memory:
                    assistant_entry["metadata"]["longitudinal_memory"] = refreshed_memory
            except Exception as exc:
                print(f"Longitudinal memory refresh failed: {exc}")

            UserStore.append_chat(username, assistant_entry)
            yield _json_line({"type": "assistant_message", "message": assistant_entry})
            yield _json_line({"type": "snapshot", "snapshot": _snapshot(username)})
            yield _json_line({"type": "done"})
        except Exception as exc:
            error_message = (
                "## Response unavailable\n"
                f"I ran into an issue while building the answer: `{exc}`.\n\n"
                "Please try again, or narrow the question if the request is very broad."
            )
            assistant_entry = {
                "role": "assistant",
                "content": error_message,
                "timestamp": _utc_now(),
                "sources": [],
                "metadata": {},
            }
            UserStore.append_chat(username, assistant_entry)
            yield _json_line({"type": "error", "message": str(exc), "assistant_message": assistant_entry})
            yield _json_line({"type": "snapshot", "snapshot": _snapshot(username)})

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/api/chat/document/stream")
async def stream_document_chat(
    message: str = Form(""),
    document: UploadFile = File(...),
    username: str = Depends(current_user),
) -> StreamingResponse:
    user_note = message.strip()
    filename = _safe_filename(document.filename or "document.pdf")
    mime_type = normalize_document_mime_type(document.content_type or "", filename)
    if mime_type not in SUPPORTED_DOCUMENT_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Upload a PDF document.")

    document_bytes = await document.read()
    if not document_bytes:
        raise HTTPException(status_code=400, detail="The uploaded document was empty.")
    if len(document_bytes) > MAX_DOCUMENT_BYTES:
        max_mb = max(1, MAX_DOCUMENT_BYTES // (1024 * 1024))
        raise HTTPException(status_code=400, detail=f"Document uploads must be {max_mb} MB or smaller.")

    display_message = user_note or f"Please analyse uploaded document: {filename}"

    def generate() -> Generator[bytes, None, None]:
        try:
            rag_engine = _get_rag_engine()
            chat_history = UserStore.get_chat_history(username)
        except Exception as exc:
            yield _json_line({"type": "error", "message": f"Failed to start the document pipeline: {exc}"})
            yield _json_line({"type": "done"})
            return

        now = _utc_now()
        user_entry = {
            "role": "user",
            "content": display_message,
            "timestamp": now,
            "sources": [],
            "metadata": {
                "document_analysis_request": True,
                "uploaded_document_name": filename,
                "uploaded_document_mime": mime_type,
            },
        }
        UserStore.append_chat(username, user_entry)
        chat_history.append(user_entry)
        yield _json_line({"type": "user_message", "message": user_entry})

        try:
            payload_final = None
            streamed_answer_parts: List[str] = []
            for event in rag_engine.stream_document_analysis_events(
                document_bytes=document_bytes,
                mime_type=mime_type,
                filename=filename,
                user_note=user_note,
                chat_history=chat_history,
                user=username,
            ):
                event_type = event.get("type")
                if event_type == "status":
                    yield _json_line({"type": "status", "message": event.get("message", "Working...")})
                elif event_type == "token":
                    delta = event.get("delta", "")
                    streamed_answer_parts.append(delta)
                    yield _json_line({"type": "token", "delta": delta})
                elif event_type == "final":
                    payload_final = event.get("payload")

            if not payload_final:
                raise RuntimeError("The document analysis pipeline did not return a payload.")

            assistant_entry = {
                "role": "assistant",
                "content": payload_final["answer_markdown"],
                "timestamp": _utc_now(),
                "sources": payload_final.get("sources", []),
                "trace_id": payload_final.get("trace", {}).get("trace_id"),
                "metadata": {
                    "personal_context": payload_final.get("personal_context", []),
                    "longitudinal_memory": payload_final.get("longitudinal_memory", ""),
                    "triage_summary": payload_final.get("triage_summary", {}),
                    "medication_alerts": payload_final.get("medication_alerts", []),
                    "resolved_medications": payload_final.get("resolved_medications", []),
                    "trace": payload_final.get("trace", {}),
                    "follow_up_questions": payload_final.get("follow_up_questions", []),
                    "document_analysis": payload_final.get("document_analysis", {}),
                    "document_original_question": payload_final.get("document_original_question", user_note),
                    "uploaded_document_name": filename,
                    "uploaded_document_mime": mime_type,
                },
            }

            UserStore.append_chat(username, assistant_entry)
            yield _json_line({"type": "assistant_message", "message": assistant_entry})
            yield _json_line({"type": "snapshot", "snapshot": _snapshot(username)})
            yield _json_line({"type": "done"})
        except DocumentAnalysisError as exc:
            error_message = (
                "## Document Not Analysed\n\n"
                f"{exc}\n\n"
                "Please upload a clear PDF medical document."
            )
            assistant_entry = {
                "role": "assistant",
                "content": error_message,
                "timestamp": _utc_now(),
                "sources": [],
                "metadata": {},
            }
            UserStore.append_chat(username, assistant_entry)
            yield _json_line({"type": "error", "message": str(exc), "assistant_message": assistant_entry})
            yield _json_line({"type": "snapshot", "snapshot": _snapshot(username)})
        except Exception as exc:
            error_message = (
                "## Document Analysis Unavailable\n\n"
                f"I ran into an issue while analysing the document: `{exc}`.\n\n"
                "Please try again with a text-based PDF, or describe the content in text."
            )
            assistant_entry = {
                "role": "assistant",
                "content": error_message,
                "timestamp": _utc_now(),
                "sources": [],
                "metadata": {},
            }
            UserStore.append_chat(username, assistant_entry)
            yield _json_line({"type": "error", "message": str(exc), "assistant_message": assistant_entry})
            yield _json_line({"type": "snapshot", "snapshot": _snapshot(username)})

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/api/chat/image/stream")
async def stream_image_chat(
    message: str = Form(""),
    image: UploadFile = File(...),
    username: str = Depends(current_user),
) -> StreamingResponse:
    user_note = message.strip()
    filename = _safe_filename(image.filename or "medical-image")
    mime_type = normalize_image_mime_type(image.content_type or "", filename)
    if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Upload a JPG, PNG, or WebP medical image.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image was empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        max_mb = max(1, MAX_IMAGE_BYTES // (1024 * 1024))
        raise HTTPException(status_code=400, detail=f"Image uploads must be {max_mb} MB or smaller.")

    display_message = user_note or f"Please analyse uploaded image: {filename}"

    def generate() -> Generator[bytes, None, None]:
        try:
            rag_engine = _get_rag_engine()
            chat_history = UserStore.get_chat_history(username)
        except Exception as exc:
            yield _json_line({"type": "error", "message": f"Failed to start the image pipeline: {exc}"})
            yield _json_line({"type": "done"})
            return

        now = _utc_now()
        user_entry = {
            "role": "user",
            "content": display_message,
            "timestamp": now,
            "sources": [],
            "metadata": {
                "image_analysis_request": True,
                "uploaded_image_name": filename,
                "uploaded_image_mime": mime_type,
            },
        }
        UserStore.append_chat(username, user_entry)
        chat_history.append(user_entry)

        event_user_entry = dict(user_entry)
        event_user_entry["metadata"] = dict(user_entry["metadata"])
        if len(image_bytes) <= 1_500_000:
            event_user_entry["metadata"]["uploaded_image_b64"] = base64.b64encode(image_bytes).decode("ascii")
        yield _json_line({"type": "user_message", "message": event_user_entry})

        try:
            payload_final = None
            streamed_answer_parts: List[str] = []
            for event in rag_engine.stream_image_analysis_events(
                image_bytes=image_bytes,
                mime_type=mime_type,
                filename=filename,
                user_note=user_note,
                chat_history=chat_history,
                user=username,
            ):
                event_type = event.get("type")
                if event_type == "status":
                    yield _json_line({"type": "status", "message": event.get("message", "Working...")})
                elif event_type == "token":
                    delta = event.get("delta", "")
                    streamed_answer_parts.append(delta)
                    yield _json_line({"type": "token", "delta": delta})
                elif event_type == "final":
                    payload_final = event.get("payload")

            if not payload_final:
                raise RuntimeError("The image analysis pipeline did not return a payload.")

            assistant_entry = {
                "role": "assistant",
                "content": payload_final["answer_markdown"],
                "timestamp": _utc_now(),
                "sources": payload_final.get("sources", []),
                "trace_id": payload_final.get("trace", {}).get("trace_id"),
                "metadata": {
                    "personal_context": payload_final.get("personal_context", []),
                    "longitudinal_memory": payload_final.get("longitudinal_memory", ""),
                    "triage_summary": payload_final.get("triage_summary", {}),
                    "medication_alerts": payload_final.get("medication_alerts", []),
                    "resolved_medications": payload_final.get("resolved_medications", []),
                    "trace": payload_final.get("trace", {}),
                    "follow_up_questions": payload_final.get("follow_up_questions", []),
                    "image_analysis": payload_final.get("image_analysis", {}),
                    "image_original_question": payload_final.get("image_original_question", user_note),
                    "uploaded_image_name": filename,
                    "uploaded_image_mime": mime_type,
                },
            }

            UserStore.append_chat(username, assistant_entry)
            yield _json_line({"type": "assistant_message", "message": assistant_entry})
            yield _json_line({"type": "snapshot", "snapshot": _snapshot(username)})
            yield _json_line({"type": "done"})
        except ImageAnalysisError as exc:
            error_message = (
                "## Image Not Analysed\n\n"
                f"{exc}\n\n"
                "Please upload a clear JPG, PNG, or WebP medical image."
            )
            assistant_entry = {
                "role": "assistant",
                "content": error_message,
                "timestamp": _utc_now(),
                "sources": [],
                "metadata": {},
            }
            UserStore.append_chat(username, assistant_entry)
            yield _json_line({"type": "error", "message": str(exc), "assistant_message": assistant_entry})
            yield _json_line({"type": "snapshot", "snapshot": _snapshot(username)})
        except Exception as exc:
            error_message = (
                "## Image Analysis Unavailable\n\n"
                f"I ran into an issue while analysing the image: `{exc}`.\n\n"
                "Please try again with a clearer image, or describe the concern in text."
            )
            assistant_entry = {
                "role": "assistant",
                "content": error_message,
                "timestamp": _utc_now(),
                "sources": [],
                "metadata": {},
            }
            UserStore.append_chat(username, assistant_entry)
            yield _json_line({"type": "error", "message": str(exc), "assistant_message": assistant_entry})
            yield _json_line({"type": "snapshot", "snapshot": _snapshot(username)})

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/api/feedback")
def submit_feedback(payload: FeedbackPayload, username: str = Depends(current_user)) -> Dict:
    trace_id = payload.trace_id.strip()
    rating = payload.rating.strip().lower()
    if rating not in {"thumbs_up", "thumbs_down"}:
        raise HTTPException(status_code=400, detail="Choose thumbs up or thumbs down.")
    if not trace_id:
        raise HTTPException(status_code=400, detail="Feedback needs a response trace.")

    existing = UserStore.get_response_feedback(username, trace_id, payload.message_id)
    if existing:
        return {
            "ok": True,
            "already_rated": True,
            "rating": existing.get("rating", rating),
            "saved": bool(existing.get("saved_to_feedback_store")),
            "snapshot": _snapshot(username),
        }

    trace = UserStore.get_response_trace(username, trace_id, payload.message_id) or next(
        (
            item
            for item in UserStore.get_interaction_traces(username, limit=None)
            if item.get("trace_id") == trace_id
        ),
        None,
    )
    if not trace:
        raise HTTPException(status_code=404, detail="Could not find the response trace for feedback.")

    saved = save_feedback(rating, trace)
    marked = UserStore.mark_response_feedback(
        username,
        trace_id=trace_id,
        message_id=payload.message_id,
        rating=rating,
        saved_to_feedback_store=saved,
    )
    if not marked:
        raise HTTPException(status_code=404, detail="Could not find the response message for feedback.")

    return {
        "ok": True,
        "already_rated": False,
        "rating": rating,
        "saved": saved,
        "snapshot": _snapshot(username),
    }


@app.post("/api/uploads")
async def upload_documents(
    files: List[UploadFile] = File(...),
    process_unverified: bool = Form(False),
    username: str = Depends(current_user),
) -> Dict:
    profile = UserStore.get_user_profile(username)
    expected_name = profile.get("display_name", username)
    save_dir = UserStore.get_upload_dir(username)
    ready_paths: List[Path] = []
    content_hashes: Dict[str, str] = {}
    pending: List[Dict] = []
    duplicates: List[Dict] = []

    # Content-hash based dedup: catches the same document re-uploaded under a
    # different filename (browsers often rename on re-pick), and avoids
    # burning two LLM calls (summarize + structured extraction) reprocessing
    # a file we already have.
    existing_hashes = {
        item.get("content_hash"): item
        for item in UserStore.get_uploads(username)
        if item.get("content_hash")
    }

    for upload in files:
        filename = _safe_filename(upload.filename or "upload.pdf")
        if not filename.lower().endswith(".pdf"):
            pending.append(
                {
                    "file": filename,
                    "status": "unsupported",
                    "message": "Only PDF uploads are supported.",
                    "detected_names": [],
                }
            )
            continue

        content = await upload.read()
        content_hash = hashlib.sha256(content).hexdigest()
        prior = existing_hashes.get(content_hash)
        if prior:
            duplicates.append(
                {
                    "file": filename,
                    "status": "duplicate",
                    "message": "This document has already been uploaded.",
                    "previous_file": prior.get("file"),
                    "previous_uploaded_at": prior.get("uploaded_at", ""),
                }
            )
            continue

        path = save_dir / filename
        path.write_bytes(content)
        verification = verify_saved_pdf(path, expected_name)
        if verification.get("status") == "matched" or process_unverified:
            ready_paths.append(path)
            content_hashes[path.name] = content_hash
        else:
            pending.append(verification)

    processed: List[Dict] = []
    rejected: List[Dict] = []
    if ready_paths:
        indexed = _get_rag_engine().ingest_documents(
            user=username, file_paths=ready_paths, content_hashes=content_hashes
        )
        for item in indexed:
            if not item.get("rejected"):
                processed.append(item)
            elif item.get("rejection_type") == "duplicate":
                duplicates.append(
                    {
                        "file": item["file"],
                        "status": "duplicate",
                        "message": "This document has already been uploaded.",
                        "previous_file": item.get("duplicate_of", ""),
                    }
                )
            else:
                rejected.append(
                    {
                        "file": item["file"],
                        "status": "unrelated",
                        "message": item.get("rejection_reason")
                        or "This does not appear to be a personal health record.",
                    }
                )

    return {
        "processed": processed,
        "pending": pending,
        "duplicates": duplicates,
        "rejected": rejected,
        "snapshot": _snapshot(username),
    }


@app.post("/api/voice/transcribe")
async def transcribe_voice(audio: UploadFile = File(...), username: str = Depends(current_user)) -> Dict:
    del username
    data = await audio.read()
    try:
        transcriber = VoiceTranscriber()
        return {"text": transcriber.transcribe(data, filename=audio.filename or "recording.webm")}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Voice transcription is unavailable: {exc}") from exc


@app.post("/api/symptoms")
def add_symptom(payload: SymptomPayload, username: str = Depends(current_user)) -> Dict:
    saved = UserStore.add_symptom_log(
        username,
        symptom=payload.symptom,
        logged_for=payload.logged_for,
        severity=payload.severity,
        triggers=payload.triggers,
        notes=payload.notes,
    )
    if not saved:
        raise HTTPException(status_code=400, detail="Enter a symptom and date.")
    _get_rag_engine().restore_user_context(username)
    return _snapshot(username)


@app.delete("/api/symptoms/{log_id}")
def delete_symptom(log_id: str, username: str = Depends(current_user)) -> Dict:
    UserStore.delete_symptom_log(username, log_id)
    _get_rag_engine().restore_user_context(username)
    return _snapshot(username)


@app.post("/api/conditions")
def save_condition(payload: ConditionPayload, username: str = Depends(current_user)) -> Dict:
    saved = UserStore.save_condition(username, payload.dict(exclude_none=True))
    if not saved:
        raise HTTPException(status_code=400, detail="Enter a condition name.")
    _get_rag_engine().restore_user_context(username)
    return _snapshot(username)


@app.delete("/api/conditions/{condition_id}")
def delete_condition(condition_id: str, username: str = Depends(current_user)) -> Dict:
    UserStore.delete_condition(username, condition_id)
    _get_rag_engine().restore_user_context(username)
    return _snapshot(username)


@app.post("/api/medications")
def save_medication(payload: MedicationPayload, username: str = Depends(current_user)) -> Dict:
    saved = UserStore.save_medication(username, payload.dict(exclude_none=True))
    if not saved:
        raise HTTPException(status_code=400, detail="Enter a medication name.")
    _get_rag_engine().restore_user_context(username)
    return _snapshot(username)


@app.delete("/api/medications/{medication_id}")
def delete_medication(medication_id: str, username: str = Depends(current_user)) -> Dict:
    UserStore.delete_medication(username, medication_id)
    _get_rag_engine().restore_user_context(username)
    return _snapshot(username)


@app.post("/api/allergies")
def save_allergy(payload: AllergyPayload, username: str = Depends(current_user)) -> Dict:
    saved = UserStore.save_allergy(username, payload.dict(exclude_none=True))
    if not saved:
        raise HTTPException(status_code=400, detail="Enter an allergy name.")
    _get_rag_engine().restore_user_context(username)
    return _snapshot(username)


@app.delete("/api/allergies/{allergy_id}")
def delete_allergy(allergy_id: str, username: str = Depends(current_user)) -> Dict:
    UserStore.delete_allergy(username, allergy_id)
    _get_rag_engine().restore_user_context(username)
    return _snapshot(username)


@app.post("/api/vitals")
def save_vitals(payload: VitalsPayload, username: str = Depends(current_user)) -> Dict:
    saved = UserStore.save_vitals_entry(username, payload.dict(exclude_none=True))
    if not saved:
        raise HTTPException(status_code=400, detail="Enter a measurement type and value.")
    _get_rag_engine().restore_user_context(username)
    return _snapshot(username)


@app.delete("/api/vitals/{vitals_id}")
def delete_vitals(vitals_id: str, username: str = Depends(current_user)) -> Dict:
    UserStore.delete_vitals_entry(username, vitals_id)
    _get_rag_engine().restore_user_context(username)
    return _snapshot(username)


@app.get("/api/export/account")
def export_account(username: str = Depends(current_user)) -> JSONResponse:
    return JSONResponse(UserStore.export_user_snapshot(username))


@app.get("/api/export/summary.pdf")
def export_summary(username: str = Depends(current_user)) -> Response:
    UserStore.add_audit(username, "summary_generated", "Health summary generated")
    pdf = _get_rag_engine().build_summary_pdf_for_user(username)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{username}-health-summary.pdf"'},
    )


@app.get("/api/terms/{role_label}")
def terms_for_role(role_label: str) -> Dict:
    return get_terms_for_role(role_label)


@app.post("/api/trials/search")
def search_trials(payload: TrialSearchPayload, username: str = Depends(current_user)) -> Dict:
    profile = UserStore.get_user_profile(username)
    trial_profile = build_trial_search_profile(
        profile=profile,
        memory=UserStore.get_longitudinal_memory(username),
        symptom_logs=UserStore.get_symptom_logs(username, limit=None),
        medications=UserStore.get_medications(username),
        allergies=UserStore.get_allergies(username),
        conditions=UserStore.get_conditions(username),
        vitals=UserStore.get_vitals(username, limit=None),
        triage_summaries=UserStore.get_triage_summaries(username, limit=None),
        document_summaries=UserStore.get_document_summaries(username),
    )
    result = find_matching_trials(
        profile=trial_profile,
        location_query=payload.location,
        max_results=max(1, min(payload.max_results, 25)),
        query_expander=QueryExpander(),
    )
    UserStore.save_trial_search_result(username, result)
    return {"result": result, "snapshot": _snapshot(username)}


@app.get("/api/trials/result")
def trial_result(username: str = Depends(current_user)) -> Dict:
    return {"result": UserStore.get_trial_search_result(username)}


# ── Clinical notes ─────────────────────────────────────────────────────────────

@app.get("/api/notes")
def list_notes(username: str = Depends(current_user)) -> Dict:
    return {"notes": UserStore.get_clinical_notes(username)}


@app.post("/api/notes")
def create_note(payload: NoteGeneratePayload, username: str = Depends(current_user)) -> Dict:
    """Generate a SOAP note from the current conversation context."""
    llm = _get_rag_engine().llm
    chat_history = UserStore.get_chat_history(username)

    question = payload.question.strip()
    conversation_summary = payload.conversation_summary.strip()

    if not question and chat_history:
        # Use last user message as the question
        for msg in reversed(chat_history):
            if msg.get("role") == "user":
                question = msg.get("content", "")[:300]
                break

    if not conversation_summary and chat_history:
        # Build a brief summary from the last 4 messages
        recent = chat_history[-4:]
        conversation_summary = " | ".join(
            f"{m.get('role','?').title()}: {m.get('content','')[:150]}"
            for m in recent
        )

    triage_list = UserStore.get_triage_summaries(username, limit=1)
    triage = triage_list[0] if triage_list else None

    note = generate_soap_note(
        username=username,
        conversation_summary=conversation_summary,
        question=question,
        triage_summary=triage,
        llm=llm,
        trace_id=payload.trace_id,
    )
    UserStore.save_clinical_note(username, note)
    return {"note": note, "snapshot": _snapshot(username)}


@app.get("/api/notes/{note_id}")
def get_note(note_id: str, username: str = Depends(current_user)) -> Dict:
    notes = UserStore.get_clinical_notes(username)
    note = next((n for n in notes if n["note_id"] == note_id), None)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found.")
    return {"note": note}


@app.put("/api/notes/{note_id}")
def update_note(
    note_id: str, payload: NoteUpdatePayload, username: str = Depends(current_user)
) -> Dict:
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    note = UserStore.update_clinical_note(username, note_id, updates)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found.")
    return {"note": note, "snapshot": _snapshot(username)}


@app.delete("/api/notes/{note_id}", status_code=204)
def delete_note(note_id: str, username: str = Depends(current_user)) -> None:
    if not UserStore.delete_clinical_note(username, note_id):
        raise HTTPException(status_code=404, detail="Note not found.")


@app.post("/api/notes/{note_id}/email")
def email_note(
    note_id: str, username: str = Depends(current_user)
) -> Dict:
    """Send a SOAP note to the user's registered email address."""
    profile = UserStore.get_user_profile(username)
    email_address = (profile or {}).get("email", "")
    if not email_address:
        raise HTTPException(status_code=400, detail="No email address saved on this account.")

    notes = UserStore.get_clinical_notes(username)
    note = next((n for n in notes if n["note_id"] == note_id), None)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found.")

    try:
        send_clinical_note_email(
            email_address,
            (profile or {}).get("display_name", username),
            note,
        )
        UserStore.mark_note_email_sent(username, note_id)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Email failed: {exc}")

    return {"ok": True, "sent_to": email_address, "snapshot": _snapshot(username)}


# ── Urgent care email alert ────────────────────────────────────────────────────

@app.post("/api/email/urgent")
def send_urgent_alert(
    payload: UrgentAlertPayload, username: str = Depends(current_user)
) -> Dict:
    """Send an urgent care alert email to the user."""
    profile = UserStore.get_user_profile(username)
    email_address = (profile or {}).get("email", "")
    if not email_address:
        raise HTTPException(status_code=400, detail="No email address saved on this account.")

    try:
        send_urgent_care_alert(
            email_address,
            (profile or {}).get("display_name", username),
            payload.reason,
            payload.urgency_level,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Email failed: {exc}")

    return {"ok": True, "sent_to": email_address}


# ── MCP server (streamable HTTP -- works locally and on Railway) ───────────────
# Mounted at /mcp so Claude Desktop / remote agents can connect to:
#   https://<your-railway-app>.railway.app/mcp
# Set MCP_API_KEY in Railway environment variables to restrict access.
#
# Two things are required for this to actually work, not just import cleanly:
#   1. FastMCP's own internal route defaults to "/mcp", so mounting its ASGI
#      app at "/mcp" would double up to "/mcp/mcp". We reset it to "/" first
#      so the mount prefix alone defines the public path.
#   2. The streamable HTTP session manager needs its `.run()` context manager
#      entered before it will handle a single request -- it starts the anyio
#      task group all sessions run in. Starlette does NOT propagate a mounted
#      sub-app's own `lifespan=` to the parent app, so without wiring it into
#      this app's startup/shutdown, every request 500s with "Task group is
#      not initialized." Confirmed live: mounting alone is not sufficient.

_MCP_KEY = os.getenv("MCP_API_KEY", "")

try:
    from contextlib import AsyncExitStack  # noqa: E402

    from backend.mcp_server import mcp as _mcp_server  # noqa: E402

    _mcp_server.settings.streamable_http_path = "/"
    _mcp_asgi = _mcp_server.streamable_http_app()

    if _MCP_KEY:
        # Wrap the ASGI app with a Bearer token gate
        _unguarded = _mcp_asgi

        async def _mcp_asgi(scope, receive, send):  # type: ignore[no-redef]
            if scope.get("type") in ("http", "websocket"):
                raw_headers = dict(scope.get("headers", []))
                auth_header = raw_headers.get(b"authorization", b"").decode()
                if auth_header != f"Bearer {_MCP_KEY}":
                    from starlette.responses import Response as _R
                    await _R("Unauthorized", status_code=401)(scope, receive, send)
                    return
            await _unguarded(scope, receive, send)

    # Starlette's Mount only matches "/mcp/..." (something after the prefix),
    # not the bare "/mcp" that every client is told to connect to. Without
    # this, the documented URL falls through to the frontend static-file
    # catch-all and returns 404/405 instead of reaching the MCP app.
    @app.api_route("/mcp", methods=["GET", "POST", "DELETE"], include_in_schema=False)
    async def _mcp_root_redirect() -> Response:
        return Response(status_code=307, headers={"Location": "/mcp/"})

    app.mount("/mcp", _mcp_asgi)

    _mcp_exit_stack: AsyncExitStack | None = None

    @app.on_event("startup")
    async def _start_mcp_session_manager() -> None:
        global _mcp_exit_stack
        _mcp_exit_stack = AsyncExitStack()
        await _mcp_exit_stack.enter_async_context(_mcp_server.session_manager.run())
        print("[API] MCP session manager started")

    @app.on_event("shutdown")
    async def _stop_mcp_session_manager() -> None:
        if _mcp_exit_stack is not None:
            await _mcp_exit_stack.aclose()

    print("[API] MCP server mounted at /mcp")
except Exception as _mcp_err:
    print(f"[API] MCP server not mounted (non-fatal): {_mcp_err}")


# ── Frontend static files ─────────────────────────────────────────────────────
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
