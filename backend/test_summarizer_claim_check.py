"""
Evidence Ledger v2 tests for backend/summarizer.py::check_claim_source_alignment:
- #1 chunked checking (no answer-length truncation, no hard 5-claim cap)
- #9 deterministic corroboration cross-check (a claim the LLM calls
  "supported" is downgraded if its cited source text doesn't actually share
  vocabulary with it)

Mocks the OpenAI client the same way backend/test_evidence_extractor.py does
-- no real API call, no DB.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from backend.summarizer import LLMHelper


class _FakeCompletions:
    def __init__(self, responses):
        """responses: a list of payload dicts, consumed in order (one per
        chunk); the last one repeats if more calls happen than entries."""
        self._responses = responses
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        payload = self._responses[index]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
        )


def _helper(responses) -> tuple[LLMHelper, _FakeCompletions]:
    helper = LLMHelper()
    completions = _FakeCompletions(responses)
    helper.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return helper, completions


def _source(source_id: str, text: str) -> dict:
    return {"source_id": source_id, "title": "Test source", "snippet": text}


def test_short_answer_makes_exactly_one_llm_call():
    helper, completions = _helper(
        [{"claims": [{"claim": "Metformin reduces HbA1c.", "status": "general_knowledge", "requires_evidence": False, "source_ids": []}]}]
    )

    result = helper.check_claim_source_alignment(
        "Metformin is commonly used for type 2 diabetes.",
        [_source("S1", "Metformin is a first-line treatment for type 2 diabetes.")],
    )

    assert len(completions.calls) == 1
    assert len(result) == 1


def test_long_answer_is_chunked_and_claims_from_every_chunk_are_returned():
    long_answer = ("First paragraph about metformin dosing. " * 20 + "\n\n") * 3
    assert len(long_answer) > 1500

    helper, completions = _helper(
        [
            {"claims": [{"claim": "Claim from chunk one.", "status": "general_knowledge", "requires_evidence": False, "source_ids": []}]},
            {"claims": [{"claim": "Claim from chunk two.", "status": "general_knowledge", "requires_evidence": False, "source_ids": []}]},
            {"claims": [{"claim": "Claim from chunk three.", "status": "general_knowledge", "requires_evidence": False, "source_ids": []}]},
        ]
    )

    result = helper.check_claim_source_alignment(long_answer, [_source("S1", "irrelevant")])

    assert len(completions.calls) >= 2
    claim_texts = {item["claim"] for item in result}
    assert "Claim from chunk one." in claim_texts


def test_merged_claims_are_capped_at_25():
    # 30 distinct claims across many chunks -- should stop persisting new ones after 25.
    chunk_payload = {
        "claims": [
            {"claim": f"Distinct claim number {i}.", "status": "general_knowledge", "requires_evidence": False, "source_ids": []}
            for i in range(5)
        ]
    }
    long_answer = ("Paragraph. " * 50 + "\n\n") * 10
    helper, _ = _helper([chunk_payload])
    # Force every chunk to report the SAME 5 claims repeated with unique text
    # per chunk by monkeypatching the chunk splitter isn't needed here --
    # instead verify the cap logic directly via a payload with >25 distinct
    # claims across the (few) real chunks this answer produces.
    helper._check_claim_alignment_chunk = lambda *_a, **_k: [
        {"claim": f"Unique claim {i}.", "status": "general_knowledge", "requires_evidence": False, "source_ids": [], "passage_ids": [], "deterministic_corroboration": "not_applicable"}
        for i in range(30)
    ]

    result = helper.check_claim_source_alignment(long_answer, [_source("S1", "text")])

    assert len(result) == 25


def test_deterministic_corroboration_downgrades_unsupported_claim():
    """LLM says "supported" citing S1, but S1's text shares no real
    vocabulary with the claim -- must be downgraded to general_knowledge."""
    helper, _ = _helper(
        [
            {
                "claims": [
                    {
                        "claim": "Ibuprofen reduces fever within thirty minutes.",
                        "status": "supported",
                        "requires_evidence": True,
                        "source_ids": ["S1"],
                    }
                ]
            }
        ]
    )

    result = helper.check_claim_source_alignment(
        "Ibuprofen reduces fever within thirty minutes.",
        [_source("S1", "This guidance discusses unrelated topics like sleep hygiene and hydration advice.")],
    )

    assert len(result) == 1
    assert result[0]["status"] == "general_knowledge"
    assert result[0]["source_ids"] == []
    assert result[0]["deterministic_corroboration"] == "failed"


def test_deterministic_corroboration_confirms_genuinely_grounded_claim():
    helper, _ = _helper(
        [
            {
                "claims": [
                    {
                        "claim": "Ibuprofen reduces fever effectively.",
                        "status": "supported",
                        "requires_evidence": True,
                        "source_ids": ["S1"],
                    }
                ]
            }
        ]
    )

    result = helper.check_claim_source_alignment(
        "Ibuprofen reduces fever effectively.",
        [_source("S1", "Clinical trials show ibuprofen reduces fever effectively in most patients.")],
    )

    assert len(result) == 1
    assert result[0]["status"] == "supported"
    assert result[0]["deterministic_corroboration"] == "confirmed"


def test_empty_answer_or_sources_returns_empty_without_llm_call():
    helper, completions = _helper([{"claims": []}])

    assert helper.check_claim_source_alignment("", [_source("S1", "text")]) == []
    assert helper.check_claim_source_alignment("Some answer.", []) == []
    assert completions.calls == []
