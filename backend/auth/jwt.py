"""Short-lived JWT access tokens for the SQL-backed authentication path."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import jwt as _pyjwt

from backend.config import jwt_secret_key

ALGORITHM = "HS256"
ISSUER = "flynnmed"
AUDIENCE = "flynnmed-api"
_DEFAULT_TTL_SECONDS = 15 * 60


class TokenError(Exception):
    """Raised for any invalid/expired/malformed token -- callers (backend/auth/
    dependencies.py) catch this one type rather than depending on PyJWT's
    exception hierarchy directly."""


@dataclass(frozen=True)
class TokenPayload:
    account_id: str
    account_kind: str
    jti: str
    issued_at: int
    expires_at: int
    session_id: str


def create_access_token(
    account_id: str,
    account_kind: str,
    *,
    session_id: str,
    ttl_seconds: int = _DEFAULT_TTL_SECONDS,
) -> str:
    now = int(time.time())
    claims = {
        "sub": account_id,
        "kind": account_kind,
        "iat": now,
        "exp": now + ttl_seconds,
        "jti": uuid.uuid4().hex,
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sid": session_id,
    }
    return _pyjwt.encode(claims, jwt_secret_key(), algorithm=ALGORITHM)


def decode_access_token(token: str) -> TokenPayload:
    try:
        claims = _pyjwt.decode(
            token,
            jwt_secret_key(),
            algorithms=[ALGORITHM],
            issuer=ISSUER,
            audience=AUDIENCE,
            options={"require": ["sub", "kind", "iat", "exp", "jti", "iss", "aud", "sid"]},
        )
    except _pyjwt.PyJWTError as exc:
        raise TokenError(str(exc)) from exc

    account_id = claims.get("sub")
    account_kind = claims.get("kind")
    if not account_id or not account_kind:
        raise TokenError("Token is missing required claims.")

    return TokenPayload(
        account_id=str(account_id),
        account_kind=str(account_kind),
        jti=str(claims.get("jti", "")),
        issued_at=int(claims.get("iat", 0)),
        expires_at=int(claims.get("exp", 0)),
        session_id=str(claims.get("sid", "")),
    )
