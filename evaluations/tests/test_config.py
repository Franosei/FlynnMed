from evaluations.config import EvalConfig


def test_default_evaluation_models_are_role_specific(monkeypatch):
    for name in (
        "EVAL_GENERATOR_MODEL",
        "EVAL_PRIMARY_GRADER_MODEL",
        "EVAL_ADJUDICATOR_MODEL",
        "EVAL_RAG_METRICS_MODEL",
        "EVAL_FALLBACK_MODEL",
        "OPENAI_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    config = EvalConfig()

    assert config.generator_model == "gpt-5.4-mini"
    assert config.primary_grader_model == "gpt-5.6-luna"
    assert config.adjudicator_model == "gpt-5.6-luna"
    assert config.rag_metrics_model == "gpt-5.6-luna"
    assert config.evaluator_fallback_model == "gpt-5.6-luna"


def test_evaluation_generator_does_not_inherit_application_model(monkeypatch):
    monkeypatch.delenv("EVAL_GENERATOR_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    assert EvalConfig().generator_model == "gpt-5.4-mini"
