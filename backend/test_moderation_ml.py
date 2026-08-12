from backend import moderation_ml


def test_moderation_falls_back_to_rules_when_detoxify_is_unavailable():
    original_detoxify = moderation_ml.Detoxify
    original_import_error = moderation_ml._DETOXIFY_IMPORT_ERROR

    moderation_ml.Detoxify = None
    moderation_ml._DETOXIFY_IMPORT_ERROR = ImportError("torch/transformers stack is incompatible")

    try:
        moderation = moderation_ml.ModerationEnsemble()
        blocked, category, _, details = moderation.decide("How can I kill myself?")
    finally:
        moderation_ml.Detoxify = original_detoxify
        moderation_ml._DETOXIFY_IMPORT_ERROR = original_import_error

    assert moderation.detox is None
    assert blocked is True
    assert category == "self_harm"
    assert details["moderation_backend"] == "rules_only"
    assert "ImportError" in details["moderation_backend_error"]


def test_moderation_rules_only_mode_allows_neutral_text():
    original_detoxify = moderation_ml.Detoxify
    original_import_error = moderation_ml._DETOXIFY_IMPORT_ERROR

    moderation_ml.Detoxify = None
    moderation_ml._DETOXIFY_IMPORT_ERROR = ImportError("detoxify disabled for this test")

    try:
        moderation = moderation_ml.ModerationEnsemble()
        blocked, category, _, details = moderation.decide("What are common symptoms of dehydration?")
    finally:
        moderation_ml.Detoxify = original_detoxify
        moderation_ml._DETOXIFY_IMPORT_ERROR = original_import_error

    assert blocked is False
    assert category == "allow"
    assert details["moderation_backend"] == "rules_only"
    assert details["detoxify"]["toxicity"] == 0.0


class _FakeDetox:
    def __init__(self, scores: dict) -> None:
        self.scores = scores

    def predict(self, text: str) -> dict:
        return self.scores


def test_moderation_does_not_block_on_high_toxicity_alone():
    """
    Regression test: Detoxify's general "toxicity" aggregate is trained on
    Jigsaw/Wikipedia comment data with no clinical vocabulary and spikes on
    ordinary medical questions it wasn't trained on -- e.g. a real production
    case, "Can I breastfeed my baby while I have this mastitis", scored
    toxicity=0.84 (over the 0.60 threshold) while every specific, better-
    calibrated category stayed low. The plain toxicity score must not block
    on its own.
    """
    moderation = moderation_ml.ModerationEnsemble()
    moderation.detox = _FakeDetox(
        {
            "toxicity": 0.84,
            "severe_toxicity": 0.01,
            "obscene": 0.29,
            "threat": 0.0,
            "insult": 0.02,
            "identity_attack": 0.0,
        }
    )

    blocked, category, _, _ = moderation.decide(
        "Can I breastfeed my baby while I have this mastitis", role_key="patient"
    )

    assert blocked is False
    assert category == "allow"


def test_moderation_still_blocks_when_specific_categories_are_high():
    """
    Companion to the regression test above: genuinely abusive text spikes the
    specific categories (obscene/insult) together with toxicity, not just
    toxicity alone -- that pattern must still be blocked.
    """
    moderation = moderation_ml.ModerationEnsemble()
    moderation.detox = _FakeDetox(
        {
            "toxicity": 0.99,
            "severe_toxicity": 0.08,
            "obscene": 0.82,
            "threat": 0.0,
            "insult": 0.97,
            "identity_attack": 0.02,
        }
    )

    blocked, category, _, _ = moderation.decide(
        "You are a stupid worthless idiot and I hate you", role_key="patient"
    )

    assert blocked is True
    assert category == "toxicity"
