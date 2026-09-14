from __future__ import annotations

import asyncio
import uuid

import pytest

from backend.config import mcp_auth_mode, mcp_enabled
from backend.mcp_auth import (
    PATIENT_EMAIL_SEND,
    PATIENT_READ,
    ActorContext,
    MCPAuthenticationMiddleware,
    MCPAuthorizationError,
    _current_actor,
    authorize_patient,
    current_actor,
)
from backend.models.account import AccountKind


def _actor(kind: AccountKind, username: str, *scopes: str) -> ActorContext:
    return ActorContext(
        account_id=uuid.uuid4(),
        account_kind=kind,
        username=username,
        scopes=frozenset(scopes),
        session_id="test-session",
    )


def test_mcp_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MCP_ENABLED", raising=False)
    assert mcp_enabled() is False


def test_mcp_rejects_shared_key_auth_modes(monkeypatch):
    monkeypatch.setenv("MCP_AUTH_MODE", "service_key")
    with pytest.raises(RuntimeError, match="must be 'jwt'"):
        mcp_auth_mode()


def test_patient_actor_can_select_only_own_username():
    actor = _actor(AccountKind.patient, "patient-one", PATIENT_READ)
    token = _current_actor.set(actor)
    try:
        assert authorize_patient(" Patient-One ", PATIENT_READ) == "patient-one"
        with pytest.raises(MCPAuthorizationError, match="not authorized for this patient"):
            authorize_patient("patient-two", PATIENT_READ)
    finally:
        _current_actor.reset(token)


def test_missing_actor_is_denied():
    with pytest.raises(MCPAuthorizationError, match="authenticated MCP actor"):
        current_actor()


def test_clinician_requires_consent_and_cannot_send_as_patient(monkeypatch):
    actor = _actor(AccountKind.clinician, "doctor-one", PATIENT_READ)
    token = _current_actor.set(actor)
    try:
        monkeypatch.setattr("backend.mcp_auth._clinician_has_patient_access", lambda *_: False)
        with pytest.raises(MCPAuthorizationError, match="not authorized for this patient"):
            authorize_patient("patient-one", PATIENT_READ)

        monkeypatch.setattr("backend.mcp_auth._clinician_has_patient_access", lambda *_: True)
        assert authorize_patient("patient-one", PATIENT_READ) == "patient-one"

        with pytest.raises(MCPAuthorizationError, match="not authorized for this operation"):
            authorize_patient(
                "patient-one",
                PATIENT_EMAIL_SEND,
                allow_clinician=False,
            )
    finally:
        _current_actor.reset(token)


def test_mcp_middleware_rejects_missing_bearer_token():
    called = False
    sent = []

    async def downstream(scope, receive, send):
        nonlocal called
        called = True

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "headers": []}
    asyncio.run(MCPAuthenticationMiddleware(downstream)(scope, receive, send))

    assert called is False
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 401


def test_mcp_middleware_installs_resolved_actor(monkeypatch):
    actor = _actor(AccountKind.patient, "patient-one", PATIENT_READ)
    observed = []

    monkeypatch.setattr("backend.mcp_auth.actor_from_access_token", lambda token: actor)

    async def downstream(scope, receive, send):
        observed.append(current_actor())
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        return None

    scope = {
        "type": "http",
        "method": "POST",
        "headers": [(b"authorization", b"Bearer valid-test-token")],
    }
    asyncio.run(MCPAuthenticationMiddleware(downstream)(scope, receive, send))

    assert observed == [actor]
    with pytest.raises(MCPAuthorizationError):
        current_actor()
