from backend.model_config import DEFAULT_OPENAI_MODEL, configured_openai_model


def test_default_openai_model_is_gpt_5_4_mini(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    assert DEFAULT_OPENAI_MODEL == "gpt-5.4-mini"
    assert configured_openai_model() == "gpt-5.4-mini"


def test_configured_openai_model_uses_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "custom-model")

    assert configured_openai_model() == "custom-model"


def test_blank_openai_model_uses_default(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "   ")

    assert configured_openai_model() == DEFAULT_OPENAI_MODEL
