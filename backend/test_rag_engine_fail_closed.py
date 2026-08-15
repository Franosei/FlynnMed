"""
Evidence Ledger v2 (#8) tests: fail-closed verification in
backend/rag_system.py::RAGEngine._finalize_answer_payload. Previously, a
failed claim-source-alignment check or a no-op correction both silently
shipped the original, unverified answer; now both retry once and then
replace the answer with SAFE_VERIFICATION_FALLBACK_MESSAGE instead.

Constructs a real RAGEngine (matches backend/test_rag_engine.py's existing
convention -- no network/LLM calls happen at construction time) and mocks
only self.llm's two claim-check methods, calling _finalize_answer_payload
directly with a minimal hand-built bundle covering every key it reads via
direct subscript (see the bundle["..."] reads in rag_system.py).
"""
from __future__ import annotations

from unittest.mock import patch

from backend.rag_system import SAFE_VERIFICATION_FALLBACK_MESSAGE, RAGEngine


def _minimal_bundle(combined_sources):
    return {
        "combined_sources": combined_sources,
        "personal_context": [],
        "retrieval_mode": "test",
        "expanded_queries": [],
        "matches": [],
        "normalized_user": None,
        "longitudinal_memory_summary": "",
    }


def _rag() -> RAGEngine:
    return RAGEngine(embedding_dir="data/uploads")


def test_alignment_check_failing_twice_blocks_the_answer():
    rag = _rag()
    bundle = _minimal_bundle([{"source_id": "S1", "title": "Test source", "evidence_tier": 1}])

    with patch.object(rag.llm, "check_claim_source_alignment", side_effect=Exception("boom")):
        result = rag._finalize_answer_payload(
            question="What dose of ibuprofen is safe?",
            raw_answer="Take 400mg of ibuprofen [S1].",
            bundle=bundle,
        )

    assert result["answer_markdown"].strip() == SAFE_VERIFICATION_FALLBACK_MESSAGE
    assert result["trace"]["claim_alignment"] == []


def test_alignment_check_succeeding_on_retry_does_not_block():
    rag = _rag()
    bundle = _minimal_bundle([{"source_id": "S1", "title": "Test source", "evidence_tier": 1}])
    call_count = {"n": 0}

    def flaky(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise Exception("transient")
        return []

    with patch.object(rag.llm, "check_claim_source_alignment", side_effect=flaky):
        result = rag._finalize_answer_payload(
            question="What dose of ibuprofen is safe?",
            raw_answer="Take 400mg of ibuprofen [S1].",
            bundle=bundle,
        )

    assert result["answer_markdown"].strip() != SAFE_VERIFICATION_FALLBACK_MESSAGE
    assert call_count["n"] == 2


def test_correction_no_op_twice_blocks_the_answer():
    rag = _rag()
    bundle = _minimal_bundle([{"source_id": "S1", "title": "Test source", "evidence_tier": 1}])
    unsupported = [
        {
            "claim": "This herbal remedy cures the flu.",
            "status": "general_knowledge",
            "requires_evidence": True,
            "source_ids": [],
            "passage_ids": [],
        }
    ]

    with patch.object(rag.llm, "check_claim_source_alignment", return_value=unsupported), \
         patch.object(rag.llm, "apply_claim_corrections", side_effect=lambda answer_markdown, **_: answer_markdown):
        result = rag._finalize_answer_payload(
            question="Does this herbal remedy help with the flu?",
            raw_answer="This herbal remedy cures the flu.",
            bundle=bundle,
        )

    assert result["answer_markdown"].strip() == SAFE_VERIFICATION_FALLBACK_MESSAGE


def test_correction_succeeding_on_retry_does_not_block():
    rag = _rag()
    bundle = _minimal_bundle([{"source_id": "S1", "title": "Test source", "evidence_tier": 1}])
    unsupported = [
        {
            "claim": "This herbal remedy cures the flu.",
            "status": "general_knowledge",
            "requires_evidence": True,
            "source_ids": [],
            "passage_ids": [],
        }
    ]
    call_count = {"n": 0}

    def flaky_correction(answer_markdown, **_kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return answer_markdown  # no-op first attempt
        return "This remedy has not been shown to cure the flu; consult a clinician."

    with patch.object(rag.llm, "check_claim_source_alignment", return_value=unsupported), \
         patch.object(rag.llm, "apply_claim_corrections", side_effect=flaky_correction):
        result = rag._finalize_answer_payload(
            question="Does this herbal remedy help with the flu?",
            raw_answer="This herbal remedy cures the flu.",
            bundle=bundle,
        )

    assert result["answer_markdown"].strip() != SAFE_VERIFICATION_FALLBACK_MESSAGE
    assert call_count["n"] == 2
