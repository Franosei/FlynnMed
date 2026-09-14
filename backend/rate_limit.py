"""Postgres-backed fixed-window limits shared by every API process."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from backend.config import jwt_secret_key, rate_limiting_enabled
from backend.db import get_session_factory
from backend.models.security import RateLimitBucket
from backend.auth.jwt import TokenError, decode_access_token


@dataclass(frozen=True)
class LimitRule:
    name: str
    methods: frozenset[str]
    prefix: str
    limit: int
    seconds: int


RULES = (
    LimitRule("login", frozenset({"POST"}), "/api/auth/login", 5, 300),
    LimitRule("signup", frozenset({"POST"}), "/api/auth/signup", 3, 3600),
    LimitRule("verification", frozenset({"POST"}), "/api/auth/verify", 5, 600),
    LimitRule("verification", frozenset({"POST"}), "/api/auth/resend", 3, 3600),
    LimitRule("clinician-registration", frozenset({"POST"}), "/api/clinician-registration", 5, 3600),
    LimitRule("access", frozenset({"POST"}), "/api/access/requests", 10, 3600),
    LimitRule("uploads", frozenset({"POST"}), "/api/uploads", 10, 600),
    LimitRule("email", frozenset({"POST"}), "/api/email/urgent", 3, 3600),
    LimitRule("admin-review", frozenset({"POST"}), "/api/admin/clinician-registrations", 10, 600),
    LimitRule("chat", frozenset({"POST"}), "/api/chat", 30, 60),
    LimitRule("mcp", frozenset({"GET", "POST", "DELETE"}), "/mcp", 60, 60),
)


def matching_rule(method: str, path: str) -> LimitRule | None:
    if method.upper() == "POST" and path.startswith("/api/notes/") and path.endswith("/email"):
        return LimitRule("email", frozenset({"POST"}), "/api/notes/*/email", 3, 3600)
    return next((r for r in RULES if method.upper() in r.methods and path.startswith(r.prefix)), None)


def _identity(scope) -> str:
    headers = {key.lower(): value for key, value in scope.get("headers", [])}
    auth = headers.get(b"authorization", b"").decode("latin1")
    client = scope.get("client") or ("unknown", 0)
    raw = str(client[0])
    if auth.lower().startswith("bearer "):
        try:
            raw = f"account:{decode_access_token(auth.partition(' ')[2]).account_id}"
        except TokenError:
            pass
    return hmac.new(jwt_secret_key().encode(), raw.encode(), hashlib.sha256).hexdigest()


def consume(rule: LimitRule, identity: str) -> tuple[bool, int]:
    now = datetime.now(timezone.utc)
    window_epoch = int(now.timestamp()) // rule.seconds * rule.seconds
    started = datetime.fromtimestamp(window_epoch, timezone.utc)
    expires = started + timedelta(seconds=rule.seconds)
    key = hashlib.sha256(f"{rule.name}:{identity}:{window_epoch}".encode()).hexdigest()
    statement = insert(RateLimitBucket).values(
        bucket_key=key, count=1, window_started_at=started, expires_at=expires
    )
    statement = statement.on_conflict_do_update(
        index_elements=[RateLimitBucket.bucket_key],
        set_={"count": RateLimitBucket.count + 1},
    ).returning(RateLimitBucket.count)
    with get_session_factory()() as db:
        count = int(db.execute(statement).scalar_one())
        db.execute(delete(RateLimitBucket).where(RateLimitBucket.expires_at < now))
        db.commit()
    return count <= rule.limit, max(1, int((expires - now).total_seconds()))


class DistributedRateLimitMiddleware:
    def __init__(self, app, *, enabled: bool | None = None):
        self.app = app
        self.enabled = rate_limiting_enabled() if enabled is None else enabled

    async def __call__(self, scope, receive, send):
        if not self.enabled or scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        rule = matching_rule(scope.get("method", ""), scope.get("path", ""))
        if rule is None:
            await self.app(scope, receive, send)
            return
        try:
            allowed, retry_after = await run_in_threadpool(consume, rule, _identity(scope))
        except Exception:
            response = JSONResponse({"detail": "Request controls are unavailable."}, status_code=503)
            await response(scope, receive, send)
            return
        if not allowed:
            response = JSONResponse(
                {"detail": "Too many requests. Try again later."},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
