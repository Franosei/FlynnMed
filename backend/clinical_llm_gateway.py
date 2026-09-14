"""The sole policy boundary for calls that may contain clinical/PHI data."""

from __future__ import annotations

import logging
import os
import re
import uuid
from urllib.parse import urlparse

from backend.config import is_production

logger = logging.getLogger(__name__)

_EMAIL = re.compile(r"(?<![\w.-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")
_NHS = re.compile(r"(?i)\b(?:NHS|MRN|patient\s*(?:id|number))\s*[:#-]?\s*[A-Z0-9 -]{5,24}\b")


class ClinicalLLMPolicyError(RuntimeError):
    pass


def validate_clinical_llm_configuration() -> None:
    """Fail readiness when production PHI processing has not been governed."""
    if not is_production():
        return
    if not _enabled("LLM_PHI_PROCESSING_ALLOWED"):
        raise ClinicalLLMPolicyError("LLM_PHI_PROCESSING_ALLOWED must be true in production.")
    if not os.getenv("LLM_APPROVED_HOSTS", "").strip():
        raise ClinicalLLMPolicyError("LLM_APPROVED_HOSTS is required in production.")
    if not os.getenv("LLM_APPROVED_MODELS", "").strip():
        raise ClinicalLLMPolicyError("LLM_APPROVED_MODELS is required in production.")


def _redact(value):
    if isinstance(value, str):
        if value.startswith("data:") and ";base64," in value[:100]:
            return value
        value = _EMAIL.sub("[REDACTED_EMAIL]", value)
        value = _PHONE.sub("[REDACTED_PHONE]", value)
        return _NHS.sub("[REDACTED_PATIENT_ID]", value)
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact(item) for item in value)
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()}
    return value


def _enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


class _Gateway:
    def __init__(self, client, purpose: str, contains_phi: bool):
        self.client = client
        self.purpose = purpose
        self.contains_phi = contains_phi

    def prepare(self, kwargs: dict) -> dict:
        if self.contains_phi and is_production() and not _enabled("LLM_PHI_PROCESSING_ALLOWED"):
            raise ClinicalLLMPolicyError(
                "PHI-bearing model calls are disabled. Complete the processor review and set "
                "LLM_PHI_PROCESSING_ALLOWED=true."
            )
        base_url = str(getattr(self.client, "base_url", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")))
        parsed = urlparse(base_url)
        allowed_hosts = {
            host.strip().lower()
            for host in os.getenv("LLM_APPROVED_HOSTS", "api.openai.com").split(",")
            if host.strip()
        }
        if is_production() and (parsed.scheme != "https" or (parsed.hostname or "").lower() not in allowed_hosts):
            raise ClinicalLLMPolicyError("The configured model endpoint is not an approved HTTPS processor.")
        model = str(kwargs.get("model") or "")
        approved_models = {
            item.strip() for item in os.getenv("LLM_APPROVED_MODELS", "").split(",") if item.strip()
        }
        if is_production() and (not approved_models or (model and model not in approved_models)):
            raise ClinicalLLMPolicyError("The requested model is not present in LLM_APPROVED_MODELS.")
        request_id = uuid.uuid4().hex
        logger.info(
            "clinical_llm_request request_id=%s purpose=%s model=%s contains_phi=%s",
            request_id,
            self.purpose,
            model,
            self.contains_phi,
        )
        return _redact(kwargs)


class _NamespaceProxy:
    def __init__(self, target, gateway: _Gateway):
        self._target = target
        self._gateway = gateway

    def __getattr__(self, name):
        value = getattr(self._target, name)
        if name in {"create", "generate"} and callable(value):
            def guarded(*args, **kwargs):
                return value(*_redact(args), **self._gateway.prepare(kwargs))
            return guarded
        if callable(value) or isinstance(value, (str, bytes, int, float, bool, type(None))):
            return value
        return _NamespaceProxy(value, self._gateway)


def guard_clinical_client(client, *, purpose: str, contains_phi: bool = True):
    """Wrap an SDK client so every outbound generation passes policy/redaction."""
    return _NamespaceProxy(client, _Gateway(client, purpose, contains_phi))
