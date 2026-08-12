import json

import pytest

from backend.document_analysis_agent import (
    DocumentAnalysisAgent,
    DocumentAnalysisError,
    validate_document_upload,
)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.choices = [_FakeChoice(json.dumps(payload))]


class _FakeCompletions:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.last_kwargs = {}

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResponse(self.payload)


class _FakeChat:
    def __init__(self, payload: dict) -> None:
        self.completions = _FakeCompletions(payload)


class _FakeClient:
    def __init__(self, payload: dict) -> None:
        self.chat = _FakeChat(payload)


class _FakeLLM:
    ANSWER_MODEL = "gpt-4o"

    def __init__(self, payload: dict) -> None:
        self.client = _FakeClient(payload)


def test_validate_document_upload_rejects_non_pdf():
    with pytest.raises(DocumentAnalysisError):
        validate_document_upload(b"not really a pdf", "image/png", "photo.png")


def test_validate_document_upload_rejects_empty_bytes():
    with pytest.raises(DocumentAnalysisError):
        validate_document_upload(b"", "application/pdf", "report.pdf")


def test_document_agent_accepts_medical_document_and_builds_evidence_question():
    payload = {
        "is_medical_document": True,
        "medical_relevance_confidence": "high",
        "document_type": "lab result",
        "key_findings": ["Elevated creatinine", "Reduced eGFR"],
        "notable_values": ["Creatinine 180 umol/L", "eGFR 32 mL/min/1.73m2"],
        "evidence_search_queries": ["chronic kidney disease staging creatinine eGFR management"],
        "reason_if_rejected": "",
    }
    agent = DocumentAnalysisAgent(_FakeLLM(payload))

    result = agent.inspect(
        document_text="Lab report: Creatinine 180 umol/L, eGFR 32 mL/min/1.73m2.",
        filename="labs.pdf",
        user_note="Latest renal panel.",
        user_profile={"date_of_birth": "1975-03-14", "biological_sex": "Female"},
    )
    question = agent.build_clinical_question(result, "Latest renal panel.")

    assert result["analysis_status"] == "accepted"
    assert result["is_medical_document"] is True
    assert "Elevated creatinine" in question
    assert "PubMed/systematic review evidence" in question
    assert "Do not present a definitive diagnosis" in question


def test_document_agent_rejects_non_medical_document():
    payload = {
        "is_medical_document": False,
        "medical_relevance_confidence": "high",
        "document_type": "invoice",
        "key_findings": [],
        "notable_values": [],
        "evidence_search_queries": [],
        "reason_if_rejected": "This document is an invoice, not a medical record.",
    }
    agent = DocumentAnalysisAgent(_FakeLLM(payload))

    result = agent.inspect(
        document_text="Invoice #4021 -- consulting services -- $500 due.",
        filename="invoice.pdf",
    )

    assert result["analysis_status"] == "rejected"
    assert result["is_medical_document"] is False
    assert "invoice" in result["reason_if_rejected"]


def test_document_agent_rejects_when_confident_but_no_findings():
    """
    Mirrors ImageAnalysisAgent's equivalent safety net: even if the model
    claims is_medical_document=True, a result with zero findings and zero
    search queries is treated as rejected rather than trusted blindly.
    """
    payload = {
        "is_medical_document": True,
        "medical_relevance_confidence": "high",
        "document_type": "",
        "key_findings": [],
        "notable_values": [],
        "evidence_search_queries": [],
        "reason_if_rejected": "",
    }
    agent = DocumentAnalysisAgent(_FakeLLM(payload))

    result = agent.inspect(document_text="garbled text", filename="unclear.pdf")

    assert result["analysis_status"] == "rejected"
