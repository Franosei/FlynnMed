from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.clinical_llm_gateway import ClinicalLLMPolicyError, guard_clinical_client
from backend.rate_limit import matching_rule


class _Create:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return "ok"


class _Client:
    base_url = "https://api.openai.com/v1"

    def __init__(self):
        self.operation = _Create()
        self.chat = SimpleNamespace(completions=self.operation)


def test_clinical_gateway_redacts_direct_identifiers(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    raw = _Client()
    guarded = guard_clinical_client(raw, purpose="test")

    assert guarded.chat.completions.create(
        model="test-model",
        messages=[{"role": "user", "content": "Email jane@example.com, NHS: 123 456 7890"}],
    ) == "ok"

    outbound = raw.operation.kwargs["messages"][0]["content"]
    assert "jane@example.com" not in outbound
    assert "123 456 7890" not in outbound


def test_clinical_gateway_fails_closed_without_production_approval(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("LLM_PHI_PROCESSING_ALLOWED", raising=False)
    guarded = guard_clinical_client(_Client(), purpose="test")

    with pytest.raises(ClinicalLLMPolicyError):
        guarded.chat.completions.create(model="test-model", messages=[])


@pytest.mark.parametrize(
    ("method", "path", "name"),
    [
        ("POST", "/api/auth/login", "login"),
        ("POST", "/api/chat/document/stream", "chat"),
        ("POST", "/api/notes/abc/email", "email"),
        ("POST", "/api/uploads", "uploads"),
        ("POST", "/mcp", "mcp"),
    ],
)
def test_sensitive_routes_have_distributed_rate_limit_rules(method, path, name):
    rule = matching_rule(method, path)
    assert rule is not None
    assert rule.name == name


def test_read_only_health_check_is_not_rate_limited():
    assert matching_rule("GET", "/api/health") is None
