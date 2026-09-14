"""Actor-bound authentication and authorization for the MCP tool surface.

The MCP transport accepts the same account JWT as the application API. The
token is resolved back to an active SQL account for every request, and tools
must additionally authorize the target patient. A caller-supplied username is
therefore only a selector; it is never authority by itself.
"""

from __future__ import annotations

import contextvars
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy import select
from starlette.responses import Response

from backend.auth.jwt import TokenError, decode_access_token
from backend.auth.sessions import SessionError, validate_access_session
from backend.db import get_session_factory
from backend.models.account import Account, AccountKind
from backend.models.consent import ConsentGrant, ConsentScope, ConsentStatus
from backend.models.patient import Patient

logger = logging.getLogger(__name__)

PATIENT_READ = "patient:read"
PATIENT_NOTE_WRITE = "patient:note:write"
PATIENT_EMAIL_SEND = "patient:email:send"
TRIALS_SEARCH = "trials:search"
CLINICAL_NOTE_GENERATE = "clinical-note:generate"
EVIDENCE_EXTRACT = "evidence:extract"

_PATIENT_SCOPES = frozenset(
    {
        PATIENT_READ,
        PATIENT_NOTE_WRITE,
        PATIENT_EMAIL_SEND,
        TRIALS_SEARCH,
        CLINICAL_NOTE_GENERATE,
        EVIDENCE_EXTRACT,
    }
)
_CLINICIAN_SCOPES = frozenset(
    {
        PATIENT_READ,
        PATIENT_NOTE_WRITE,
        TRIALS_SEARCH,
        CLINICAL_NOTE_GENERATE,
        EVIDENCE_EXTRACT,
    }
)


class MCPAuthenticationError(Exception):
    """The transport credential is missing, invalid, or no longer active."""


class MCPAuthorizationError(PermissionError):
    """The authenticated actor cannot perform the requested MCP operation."""


@dataclass(frozen=True)
class ActorContext:
    account_id: uuid.UUID
    account_kind: AccountKind
    username: str
    scopes: frozenset[str]
    session_id: str


_current_actor: contextvars.ContextVar[ActorContext | None] = contextvars.ContextVar(
    "flynnmed_mcp_actor", default=None
)


def actor_from_access_token(token: str) -> ActorContext:
    """Resolve a bearer JWT to current, server-authoritative account state."""
    try:
        payload = decode_access_token(token)
        account_id = uuid.UUID(payload.account_id)
    except (TokenError, ValueError) as exc:
        raise MCPAuthenticationError("Invalid access token.") from exc

    with get_session_factory()() as session:
        account = session.get(Account, account_id)
        if (
            account is None
            or not account.is_active
            or not account.email_verified
            or account.account_kind.value != payload.account_kind
        ):
            raise MCPAuthenticationError("Invalid access token.")
        try:
            validate_access_session(session, payload.session_id, account.id)
        except SessionError as exc:
            raise MCPAuthenticationError("Invalid access token.") from exc
        kind = account.account_kind
        username = account.username

    scopes = _PATIENT_SCOPES if kind == AccountKind.patient else _CLINICIAN_SCOPES
    return ActorContext(
        account_id=account_id,
        account_kind=kind,
        username=username,
        scopes=scopes,
        session_id=payload.session_id,
    )


def current_actor() -> ActorContext:
    actor = _current_actor.get()
    if actor is None:
        raise MCPAuthorizationError("An authenticated MCP actor is required.")
    return actor


def require_scopes(required_scopes: Iterable[str]) -> ActorContext:
    actor = current_actor()
    required = frozenset(required_scopes)
    if not required.issubset(actor.scopes):
        logger.warning(
            "MCP scope denied actor_account_id=%s actor_kind=%s required_scopes=%s",
            actor.account_id,
            actor.account_kind.value,
            sorted(required),
        )
        raise MCPAuthorizationError("The MCP actor is not authorized for this operation.")
    return actor


def _clinician_has_patient_access(actor: ActorContext, username: str) -> bool:
    now = datetime.now(timezone.utc)
    with get_session_factory()() as session:
        patient_account = session.execute(
            select(Account).where(Account.username == username)
        ).scalar_one_or_none()
        if patient_account is None or patient_account.account_kind != AccountKind.patient:
            return False
        patient = session.execute(
            select(Patient).where(Patient.account_id == patient_account.id)
        ).scalar_one_or_none()
        if patient is None:
            return False
        grants = session.execute(
            select(ConsentGrant).where(
                ConsentGrant.patient_id == patient.id,
                ConsentGrant.clinician_account_id == actor.account_id,
                ConsentGrant.status == ConsentStatus.active,
            )
        ).scalars()
        return any(
            (grant.expires_at is None or grant.expires_at > now)
            and ConsentScope.previsit_summary.value in (grant.scope or [])
            for grant in grants
        )


def authorize_patient(
    username: str,
    *required_scopes: str,
    allow_clinician: bool = True,
) -> str:
    """Authorize a stored-record tool and return its normalized patient key."""
    actor = require_scopes(required_scopes)
    normalized = username.strip().lower()
    allowed = False
    if actor.account_kind == AccountKind.patient:
        allowed = normalized == actor.username.strip().lower()
    elif actor.account_kind == AccountKind.clinician and allow_clinician:
        allowed = _clinician_has_patient_access(actor, normalized)

    if not allowed:
        logger.warning(
            "MCP patient access denied actor_account_id=%s actor_kind=%s",
            actor.account_id,
            actor.account_kind.value,
        )
        # Deliberately identical for unknown patients, missing consent, and
        # cross-account patient access: do not turn MCP into an identifier oracle.
        raise MCPAuthorizationError("The MCP actor is not authorized for this patient.")

    logger.info(
        "MCP patient access granted actor_account_id=%s actor_kind=%s scopes=%s",
        actor.account_id,
        actor.account_kind.value,
        sorted(required_scopes),
    )
    return normalized


class MCPAuthenticationMiddleware:
    """ASGI bearer gate that installs an ActorContext for each MCP request."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        raw_headers = dict(scope.get("headers", []))
        authorization = raw_headers.get(b"authorization", b"").decode("latin-1")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            await self._reject(scope, receive, send)
            return

        try:
            actor = actor_from_access_token(token)
        except MCPAuthenticationError:
            await self._reject(scope, receive, send)
            return

        context_token = _current_actor.set(actor)
        try:
            await self.app(scope, receive, send)
        finally:
            _current_actor.reset(context_token)

    @staticmethod
    async def _reject(scope, receive, send) -> None:
        if scope.get("type") == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        await Response(
            "Unauthorized",
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )(scope, receive, send)
