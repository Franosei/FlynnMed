"""Thin wrapper connecting the evaluation harness to FlynnMed's real,
production end-to-end pipeline -- nothing here reimplements clinical logic
or modifies backend/ in any way.

Every case is run through `backend.rag_system.RAGEngine.handle_user_question`,
the exact same call the chat API uses. This exercises the full pipeline
(role routing, patient-history context, crisis pre-screen, moderation,
intent/risk classification, policy gate, pathway context, agentic evidence
retrieval, and response synthesis) unmodified.

Role handling: FlynnMed's role-aware behaviour (backend/role_router.py) is
driven by the calling account's stored `clinical_role`. The harness resolves
the intended audience using explicit role statements, HealthBench audience
tags and bounded clinical-context rules. It creates an isolated, one-case
account for professional roles and passes that username into the unchanged
production pipeline. Patient cases keep `user=None`, which is FlynnMed's
patient-mode default. Evaluation account storage is configured under the run
directory before backend modules are imported, preventing real account data
or evaluation cases from entering one another's context.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from evaluations.config import EvalConfig
from evaluations.models import EvalCase, PipelineResponse
from evaluations.role_detection import eval_account_username


def configure_evaluation_storage(storage_root: Path) -> None:
    """Isolate synthetic evaluation accounts from local and hosted records."""
    storage_root.mkdir(parents=True, exist_ok=True)
    os.environ["DATA_BACKEND"] = "legacy"
    os.environ["FLYNNMED_USER_DB_PATH"] = str(storage_root / "eval_users.json")
    os.environ["FLYNNMED_DATA_DIR"] = str(storage_root / "data")


def ensure_eval_account(role: str, case_id: str) -> Optional[str]:
    """Returns the username to run this case as, creating a fresh, one-shot
    account for the detected role if it isn't "patient" (patient already IS
    role_router.py's default for an anonymous/empty profile, so there's
    nothing to gain from creating an account for it)."""
    if role == "patient":
        return None

    from backend.user_store import UserStore

    username = eval_account_username(role, case_id)
    if not UserStore.get_user_profile(username):
        UserStore.create_user(
            username=username,
            password="eval-harness-not-a-real-account-0000",
            display_name=f"Eval Harness ({role.title()})",
            email=f"{username}@example.invalid",
            care_context="Automated evaluation harness -- not a real account",
            role=role,
            clinical_role=role,
            terms_version="eval-harness",
        )
    return username


def build_rag_engine(config: EvalConfig):
    """Constructs one RAGEngine for the whole run (expensive to build -- do
    not construct per-case). Sets the response-generation model FlynnMed's
    LLMHelper actually uses for the primary answer.

    Note: `LLMHelper.ANSWER_MODEL` is a hardcoded class constant in
    `backend/summarizer.py` (not read from the `OPENAI_MODEL` environment
    variable at call time), so setting `OPENAI_MODEL` alone would not affect
    the real answer-generation call. This overrides that class attribute for
    the lifetime of this standalone evaluation process only -- it never
    touches the live app process and does not edit any source file.
    """
    from backend.rag_system import RAGEngine
    from backend.summarizer import LLMHelper

    LLMHelper.ANSWER_MODEL = config.generator_model
    LLMHelper.REQUEST_TIMEOUT_SECONDS = config.request_timeout_seconds

    return RAGEngine()


def run_case(
    rag_engine,
    case: EvalCase,
    user: Optional[str] = None,
    role: str = "patient",
    role_reason: str = "caller did not supply a role resolution reason",
    role_confidence: Optional[float] = None,
) -> PipelineResponse:
    """Runs one EvalCase through the real pipeline and captures the result.

    `user`: resolved by the caller via `resolve_case_role` and
    `ensure_eval_account`. `None` reproduces anonymous patient mode. The role,
    reason and confidence are recorded for report transparency.

    Raises whatever RAGEngine.handle_user_question raises -- the runner is
    responsible for retry/error handling (see evaluations/grading.py's
    `call_with_retry`, reused for this call too).
    """
    question_turn = case.last_user_turn()
    chat_history = [
        {"role": turn.role, "content": turn.content} for turn in case.history_turns()
    ]

    started = time.perf_counter()
    payload = rag_engine.handle_user_question(
        question=question_turn.content,
        chat_history=chat_history or None,
        stream=False,
        user=user,
    )
    duration = time.perf_counter() - started

    trace = payload.get("trace", {}) or {}
    return PipelineResponse(
        case_id=case.case_id,
        answer_markdown=payload.get("answer_markdown", ""),
        answer_text=payload.get("answer_text", payload.get("answer_markdown", "")),
        sources=payload.get("sources", []) or [],
        personal_context=payload.get("personal_context", []) or [],
        trace=trace,
        full_payload=payload,
        duration_seconds=duration,
        resolved_role=role,
        role_resolution_reason=role_reason,
        role_resolution_confidence=role_confidence,
    )
