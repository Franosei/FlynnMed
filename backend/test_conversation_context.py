from backend.conversation_context import build_conversation_context, render_verbatim


def test_context_excludes_current_and_keeps_previous_five_verbatim():
    history = [
        {"role": "user", "content": "First question\nwith exact spacing."},
        {"role": "assistant", "content": "First answer."},
        {"role": "user", "content": "Second question."},
        {"role": "assistant", "content": "Second answer."},
        {"role": "user", "content": "Third question."},
        {"role": "assistant", "content": "Third answer."},
        {"role": "user", "content": "Current follow-up"},
    ]

    context = build_conversation_context(history, "Current follow-up")

    assert len(context.prior_messages) == 6
    assert [item["content"] for item in context.previous_five] == [
        "First answer.",
        "Second question.",
        "Second answer.",
        "Third question.",
        "Third answer.",
    ]
    verbatim = render_verbatim(context.previous_five)
    assert "Current follow-up" not in verbatim
    assert "First answer." in verbatim
    assert "Third answer." in verbatim


def test_whole_chat_summary_covers_all_prior_roles_and_patient_only_summary_is_safe():
    history = [
        {"role": "user", "content": "I have mastitis."},
        {"role": "assistant", "content": "You might be taking Drug X."},
        {"role": "user", "content": "My prescription is flucloxacillin."},
    ]

    context = build_conversation_context(history, "A new question")

    assert "I have mastitis" in context.full_summary
    assert "You might be taking Drug X" in context.full_summary
    assert "My prescription is flucloxacillin" in context.full_summary
    assert "Assistant previously replied" in context.full_summary
    assert "Drug X" not in context.patient_statement_summary
    assert "flucloxacillin" in context.patient_statement_summary
