import time

import jwt as pyjwt
import pytest

from backend.auth.jwt import TokenError, create_access_token, decode_access_token


@pytest.fixture(autouse=True)
def _fixed_secret(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-do-not-use-in-production-32bytes+")


def test_round_trip_encode_decode():
    token = create_access_token("11111111-1111-1111-1111-111111111111", "clinician")
    payload = decode_access_token(token)
    assert payload.account_id == "11111111-1111-1111-1111-111111111111"
    assert payload.account_kind == "clinician"
    assert payload.jti


def test_expired_token_raises_token_error():
    token = create_access_token("acct-1", "patient", ttl_seconds=-1)
    with pytest.raises(TokenError):
        decode_access_token(token)


def test_garbage_token_raises_token_error():
    with pytest.raises(TokenError):
        decode_access_token("not-a-real-token")


def test_wrong_signature_is_rejected():
    token = pyjwt.encode(
        {"sub": "acct-1", "kind": "patient", "iat": int(time.time()), "exp": int(time.time()) + 3600},
        "a-different-test-secret-with-32-bytes+",
        algorithm="HS256",
    )
    with pytest.raises(TokenError):
        decode_access_token(token)


def test_token_missing_required_claims_is_rejected(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-do-not-use-in-production-32bytes+")
    token = pyjwt.encode({"iat": int(time.time()), "exp": int(time.time()) + 3600}, "test-secret-do-not-use-in-production-32bytes+", algorithm="HS256")
    with pytest.raises(TokenError):
        decode_access_token(token)


def test_two_tokens_for_same_account_have_different_jti():
    token_a = create_access_token("acct-1", "patient")
    token_b = create_access_token("acct-1", "patient")
    assert decode_access_token(token_a).jti != decode_access_token(token_b).jti
