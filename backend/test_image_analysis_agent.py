import json
from io import BytesIO

import pytest
from PIL import Image

from backend.image_analysis_agent import (
    ImageAnalysisAgent,
    ImageAnalysisError,
    normalize_image_upload,
    validate_image_upload,
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


def _image_bytes(image_format: str) -> bytes:
    output = BytesIO()
    Image.new("RGB", (4, 4), "red").save(output, format=image_format)
    return output.getvalue()


def test_validate_image_upload_rejects_non_images():
    with pytest.raises(ImageAnalysisError):
        validate_image_upload(b"not really an image", "application/pdf", "report.pdf")


def test_normalize_image_rejects_declared_type_mismatch_and_strips_trailing_bytes():
    original = _image_bytes("PNG")
    with pytest.raises(ImageAnalysisError):
        normalize_image_upload(original, "image/jpeg", "wrong.jpg")

    normalized, mime = normalize_image_upload(original + b"hidden trailing payload", "image/png", "ok.png")
    assert mime == "image/png"
    assert b"hidden trailing payload" not in normalized


def test_image_agent_accepts_medical_image_and_builds_evidence_question():
    payload = {
        "is_medical_image": True,
        "medical_relevance_confidence": "high",
        "image_focus": "skin change",
        "body_region_or_subject": "forearm",
        "observable_findings": ["patchy red area", "slight swelling"],
        "colour_or_texture_changes": ["redness", "dry texture"],
        "visible_red_flag_clues": ["spreading redness cannot be assessed from one image"],
        "evidence_search_queries": ["adult rash redness swelling differential diagnosis red flags"],
        "reason_if_rejected": "",
    }
    agent = ImageAnalysisAgent(_FakeLLM(payload))

    result = agent.inspect(
        image_bytes=_image_bytes("PNG"),
        mime_type="image/png",
        user_note="This rash changed colour today.",
        user_profile={"date_of_birth": "1980-01-01", "biological_sex": "Female"},
        filename="rash.png",
    )
    question = agent.build_clinical_question(result, "This rash changed colour today.")

    assert result["analysis_status"] == "accepted"
    assert result["is_medical_image"] is True
    assert "patchy red area" in question
    assert "PubMed/systematic review evidence" in question
    assert "Do not present a definitive diagnosis" in question


def test_image_agent_rejects_non_medical_visual_result():
    payload = {
        "is_medical_image": False,
        "medical_relevance_confidence": "high",
        "image_focus": "landscape",
        "body_region_or_subject": "",
        "observable_findings": ["trees and a road"],
        "colour_or_texture_changes": [],
        "visible_red_flag_clues": [],
        "evidence_search_queries": [],
        "reason_if_rejected": "The image appears unrelated to a health concern.",
    }
    agent = ImageAnalysisAgent(_FakeLLM(payload))

    result = agent.inspect(
        image_bytes=_image_bytes("JPEG"),
        mime_type="image/jpeg",
        filename="holiday.jpg",
    )

    assert result["analysis_status"] == "rejected"
    assert result["is_medical_image"] is False
    assert "unrelated" in result["reason_if_rejected"]
