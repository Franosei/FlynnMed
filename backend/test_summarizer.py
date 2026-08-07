from backend.summarizer import LLMHelper


def test_answer_generation():
    helper = LLMHelper()
    question = "Is dexamethasone safe for elderly patients?"
    sources = [
        {
            "source_id": "S1",
            "title": "Example corticosteroid safety study",
            "journal": "Example Journal",
            "year": "2024",
            "section": "Discussion",
            "snippet": "Older adults may require closer monitoring because adverse effects can be more frequent in frail populations.",
        }
    ]

    response = helper.answer_question(
        question=question,
        context="",
        source_briefings=sources,
        stream=False,
    )
    print(response)


_UNSUPPORTED_CLAIM_SOURCES = [
    {
        "source_id": "S1",
        "title": "Migraine self-care guidance",
        "snippet": "Rest in a dark, quiet room and stay hydrated during a migraine attack.",
    }
]
# The mechanism claim below is not present in any source and is specific
# enough that a reader would expect it to be evidence-backed.
_UNSUPPORTED_CLAIM_ANSWER = (
    "## Likely Explanation\n"
    "Migraines are caused by a 23% drop in serotonin binding at 5-HT2B receptors, "
    "which directly triggers the aura phase in 90% of cases.\n\n"
    "## What To Do Now\n"
    "Rest in a dark, quiet room and stay hydrated."
)


def test_check_claim_source_alignment_flags_unsupported_specific_claim():
    helper = LLMHelper()

    claims = helper.check_claim_source_alignment(
        answer_markdown=_UNSUPPORTED_CLAIM_ANSWER,
        source_briefings=_UNSUPPORTED_CLAIM_SOURCES,
    )

    assert isinstance(claims, list)
    assert claims, "expected at least one extracted claim"
    for claim in claims:
        assert set(claim.keys()) == {"claim", "status", "requires_evidence", "source_ids"}
        assert claim["status"] in ("supported", "general_knowledge")

    unsupported = [
        c for c in claims if c["status"] == "general_knowledge" and c["requires_evidence"]
    ]
    assert unsupported, (
        "expected the fabricated serotonin/5-HT2B statistic to be flagged as "
        f"unsupported and evidence-requiring; got {claims}"
    )


def test_rewrite_unsupported_claims_softens_flagged_claim():
    helper = LLMHelper()
    # Synthetic finding, as if check_claim_source_alignment had flagged it --
    # kept independent of that test so each test makes its own bounded set of
    # live calls rather than chaining through another test function.
    unsupported_claims = [
        {
            "claim": "Migraines are caused by a 23% drop in serotonin binding at 5-HT2B receptors, which directly triggers the aura phase in 90% of cases.",
            "status": "general_knowledge",
            "requires_evidence": True,
            "source_ids": [],
        }
    ]

    revised = helper.rewrite_unsupported_claims(
        answer_markdown=_UNSUPPORTED_CLAIM_ANSWER,
        unsupported_claims=unsupported_claims,
        source_briefings=_UNSUPPORTED_CLAIM_SOURCES,
    )

    assert revised
    assert "## What To Do Now" in revised, "unrelated sections must survive the rewrite"
    print(revised)


def test_rewrite_unsupported_claims_is_noop_without_flagged_claims():
    helper = LLMHelper()
    revised = helper.rewrite_unsupported_claims(
        answer_markdown=_UNSUPPORTED_CLAIM_ANSWER,
        unsupported_claims=[],
        source_briefings=_UNSUPPORTED_CLAIM_SOURCES,
    )
    assert revised == _UNSUPPORTED_CLAIM_ANSWER


if __name__ == "__main__":
    test_answer_generation()
    test_check_claim_source_alignment_flags_unsupported_specific_claim()
    test_rewrite_unsupported_claims_softens_flagged_claim()
    test_rewrite_unsupported_claims_is_noop_without_flagged_claims()
