import os

from evaluations.config import EvalConfig
from evaluations.models import ConversationTurn, EvalCase
from evaluations.pipeline import (
    build_rag_engine,
    configure_evaluation_storage,
    ensure_eval_account,
    run_case,
)


class _FakeRagEngine:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def handle_user_question(
        self, question, chat_history=None, stream=False, user=None
    ):
        self.calls.append(
            {
                "question": question,
                "chat_history": chat_history,
                "stream": stream,
                "user": user,
            }
        )
        return self._payload


def _case(conversation):
    return EvalCase(
        case_id="case-1",
        source_dataset="healthbench",
        conversation=[ConversationTurn(**turn) for turn in conversation],
        rubrics=[],
        tags=[],
    )


def test_run_case_splits_history_and_question_correctly():
    case = _case(
        [
            {"role": "user", "content": "I have a cough for 3 weeks."},
            {"role": "assistant", "content": "How severe is it?"},
            {"role": "user", "content": "It's mild but persistent."},
        ]
    )
    fake_engine = _FakeRagEngine(
        {"answer_markdown": "ans", "answer_text": "ans", "trace": {}}
    )

    run_case(fake_engine, case)

    assert len(fake_engine.calls) == 1
    call = fake_engine.calls[0]
    assert call["question"] == "It's mild but persistent."
    assert call["chat_history"] == [
        {"role": "user", "content": "I have a cough for 3 weeks."},
        {"role": "assistant", "content": "How severe is it?"},
    ]
    assert call["user"] is None  # anonymous -- never a real account


def test_run_case_passes_none_history_for_single_turn_case():
    case = _case([{"role": "user", "content": "Is ibuprofen safe with paracetamol?"}])
    fake_engine = _FakeRagEngine(
        {"answer_markdown": "ans", "answer_text": "ans", "trace": {}}
    )

    run_case(fake_engine, case)

    assert fake_engine.calls[0]["chat_history"] is None


def test_run_case_captures_trace_and_sources():
    case = _case([{"role": "user", "content": "What does my peak flow mean?"}])
    payload = {
        "answer_markdown": "Here is the answer [S1]",
        "answer_text": "Here is the answer",
        "sources": [{"source_id": "S1", "title": "NHS guidance"}],
        "personal_context": [],
        "trace": {"risk_level": "routine", "crisis_detected": False},
    }
    fake_engine = _FakeRagEngine(payload)

    response = run_case(fake_engine, case)

    assert response.case_id == "case-1"
    assert response.answer_markdown == "Here is the answer [S1]"
    assert response.sources == [{"source_id": "S1", "title": "NHS guidance"}]
    assert response.trace["risk_level"] == "routine"
    assert response.duration_seconds >= 0


def test_run_case_records_resolved_role_and_passes_user_through():
    case = _case([{"role": "user", "content": "What are the new ACLS updates?"}])
    fake_engine = _FakeRagEngine(
        {"answer_markdown": "ans", "answer_text": "ans", "trace": {}}
    )

    response = run_case(
        fake_engine,
        case,
        user="eval-harness-doctor-case-1",
        role="doctor",
        role_reason="explicit clinical role in user text",
        role_confidence=1.0,
    )

    assert fake_engine.calls[0]["user"] == "eval-harness-doctor-case-1"
    assert response.resolved_role == "doctor"
    assert response.role_resolution_reason == "explicit clinical role in user text"
    assert response.role_resolution_confidence == 1.0


def test_run_case_defaults_to_patient_role_and_none_user():
    case = _case([{"role": "user", "content": "I have a headache."}])
    fake_engine = _FakeRagEngine(
        {"answer_markdown": "ans", "answer_text": "ans", "trace": {}}
    )

    response = run_case(fake_engine, case)

    assert fake_engine.calls[0]["user"] is None
    assert response.resolved_role == "patient"


def test_ensure_eval_account_returns_none_for_patient_role():
    # patient is role_router.py's own default for an anonymous/empty profile,
    # so there's nothing to create -- must not touch UserStore at all.
    assert ensure_eval_account("patient", "case-1") is None


def test_configure_evaluation_storage_isolates_accounts(tmp_path, monkeypatch):
    monkeypatch.delenv("FLYNNMED_USER_DB_PATH", raising=False)
    monkeypatch.delenv("FLYNNMED_DATA_DIR", raising=False)

    configure_evaluation_storage(tmp_path)

    assert os.environ["DATA_BACKEND"] == "legacy"
    assert os.environ["FLYNNMED_USER_DB_PATH"] == str(tmp_path / "eval_users.json")
    assert os.environ["FLYNNMED_DATA_DIR"] == str(tmp_path / "data")


def test_ensure_eval_account_creates_account_for_non_patient_role(monkeypatch):
    import backend.user_store

    created = {}

    class _FakeUserStore:
        @staticmethod
        def get_user_profile(username):
            return {}

        @staticmethod
        def create_user(**kwargs):
            created.update(kwargs)
            return True

    monkeypatch.setattr(backend.user_store, "UserStore", _FakeUserStore)

    username = ensure_eval_account("doctor", "case-42")

    assert username is not None
    assert username.startswith("eval-harness-doctor-")
    assert created["clinical_role"] == "doctor"
    assert created["role"] == "doctor"
    assert created["username"] == username
    assert len(created["password"]) >= 8


def test_ensure_eval_account_does_not_recreate_existing_account(monkeypatch):
    import backend.user_store

    class _FakeUserStore:
        @staticmethod
        def get_user_profile(username):
            return {"username": username, "clinical_role": "doctor"}  # already exists

        @staticmethod
        def create_user(**kwargs):
            raise AssertionError(
                "create_user should not be called for an existing account"
            )

    monkeypatch.setattr(backend.user_store, "UserStore", _FakeUserStore)

    username = ensure_eval_account("doctor", "case-42")
    assert username is not None


def test_build_rag_engine_sets_generator_model_without_touching_source(monkeypatch):
    """LLMHelper.ANSWER_MODEL is a hardcoded class constant in
    backend/summarizer.py, not read from OPENAI_MODEL at call time -- this
    confirms build_rag_engine overrides it for the run (a runtime attribute
    assignment on the already-loaded class, not a source-file edit) and that
    the override is restored so it doesn't leak into other tests/processes."""
    import backend.rag_system
    import backend.summarizer

    original_answer_model = backend.summarizer.LLMHelper.ANSWER_MODEL

    class _FakeRAGEngine:
        def __init__(self):
            pass

    monkeypatch.setattr(backend.rag_system, "RAGEngine", _FakeRAGEngine)
    try:
        config = EvalConfig(generator_model="gpt-5.4-mini")
        engine = build_rag_engine(config)
        assert isinstance(engine, _FakeRAGEngine)
        assert backend.summarizer.LLMHelper.ANSWER_MODEL == "gpt-5.4-mini"
    finally:
        backend.summarizer.LLMHelper.ANSWER_MODEL = original_answer_model


def test_build_rag_engine_can_disable_query_expansion_for_ablation(monkeypatch):
    import backend.rag_system

    class _FakeOrchestrator:
        query_expander = object()

    class _FakeRAGEngine:
        def __init__(self):
            self.query_expander = object()
            self._orchestrator = _FakeOrchestrator()

    monkeypatch.setattr(backend.rag_system, "RAGEngine", _FakeRAGEngine)
    engine = build_rag_engine(EvalConfig(query_expansion_enabled=False))

    assert engine.query_expander is None
    assert engine._orchestrator.query_expander is None
