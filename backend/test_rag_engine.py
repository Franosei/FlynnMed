from unittest.mock import patch

from backend.rag_system import RAGEngine
from backend.user_store import UserStore


def test_rag_pipeline():
    print("Initializing RAG engine...")
    rag = RAGEngine(embedding_dir="data/uploads")

    question = "What does recent evidence say about hypertension treatment in older adults?"
    print(f"Asking question: {question}")

    payload = rag.handle_user_question(question=question, stream=False)

    print("\nAnswer:\n")
    print(payload["answer_markdown"])
    print("\nTrace:\n")
    print(payload["trace"])


def _minimal_target_patient_data() -> dict:
    return {
        "user_profile": {"display_name": "Test Patient"},
        "medications": [],
        "symptom_logs": [],
        "triage_summaries": [],
        "allergies": [],
        "conditions": [],
        "vitals": [],
        "document_summaries": [],
        "longitudinal_memory_base": "",
    }


def test_prepare_answer_bundle_without_target_patient_data_uses_userstore():
    """
    Backward-compat regression for the target_patient_data seam: the default,
    patient-facing path (target_patient_data=None, every pre-existing call
    site) must keep fetching via UserStore.get_*/restore_user_context exactly
    as before -- this is what every other patient-facing test already
    exercises end-to-end, this test just makes the call-count assertion
    explicit and fast (mocked, no live LLM/network calls).
    """
    rag = RAGEngine(embedding_dir="data/uploads")

    with patch.object(rag, "restore_user_context") as mock_restore, \
         patch.object(rag._orchestrator, "prepare_bundle", return_value={"kind": "final", "payload": {}}) as mock_prepare, \
         patch.object(UserStore, "get_user_profile", return_value={}) as mock_profile, \
         patch.object(UserStore, "get_medications", return_value=[]) as mock_meds:
        rag._prepare_answer_bundle(question="test question", user="someuser")

        mock_restore.assert_called_once()
        mock_profile.assert_called_once()
        # get_medications is also called a second time internally by
        # get_combined_longitudinal_memory building its own composite
        # summary (pre-existing behavior, unrelated to this seam) -- the
        # point here is just that it's called at all in the default path.
        assert mock_meds.call_count >= 1
        mock_prepare.assert_called_once()


def test_prepare_answer_bundle_with_target_patient_data_skips_userstore():
    """
    The other half of the same regression: when target_patient_data is
    supplied (the clinician-scoped patient chat), UserStore.get_*/
    restore_user_context must NOT be called at all -- the clinician's own
    account has no Patient row to fetch (UserStore/SqlUserStore only ever
    resolve "the calling account's own Patient row"), and restore_user_context
    would otherwise mutate the shared embedding store keyed by the wrong
    identity (see the docstring on _prepare_answer_bundle).
    """
    rag = RAGEngine(embedding_dir="data/uploads")

    with patch.object(rag, "restore_user_context") as mock_restore, \
         patch.object(rag._orchestrator, "prepare_bundle", return_value={"kind": "final", "payload": {}}) as mock_prepare, \
         patch.object(UserStore, "get_user_profile") as mock_profile, \
         patch.object(UserStore, "get_medications") as mock_meds:
        rag._prepare_answer_bundle(
            question="test question",
            user="clinician_account",
            target_patient_data=_minimal_target_patient_data(),
        )

        mock_restore.assert_not_called()
        mock_profile.assert_not_called()
        mock_meds.assert_not_called()
        mock_prepare.assert_called_once()
        # `user` in prepare_bundle's call is still the acting clinician, not
        # swapped for anything patient-derived -- audit/rate-limit identity
        # must never come from target_patient_data.
        assert mock_prepare.call_args.kwargs["user"] == "clinician_account"


if __name__ == "__main__":
    test_rag_pipeline()
    test_prepare_answer_bundle_without_target_patient_data_uses_userstore()
    test_prepare_answer_bundle_with_target_patient_data_skips_userstore()
