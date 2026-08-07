from __future__ import annotations

import base64
import json
import os
from typing import Dict, List, Optional

from backend.user_store import compute_current_age


SUPPORTED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = int(os.getenv("IMAGE_ANALYSIS_MAX_BYTES", str(5 * 1024 * 1024)))


class ImageAnalysisError(ValueError):
    """Raised when an uploaded image cannot be accepted for analysis."""


def normalize_image_mime_type(mime_type: str, filename: str = "") -> str:
    cleaned = (mime_type or "").split(";")[0].strip().lower()
    if cleaned == "image/jpg":
        cleaned = "image/jpeg"
    if cleaned:
        return cleaned

    suffix = (filename or "").rsplit(".", 1)[-1].lower()
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }.get(suffix, "")


def validate_image_upload(image_bytes: bytes, mime_type: str, filename: str = "") -> str:
    normalized_mime = normalize_image_mime_type(mime_type, filename)
    if normalized_mime not in SUPPORTED_IMAGE_MIME_TYPES:
        raise ImageAnalysisError("Upload a JPG, PNG, or WebP medical image.")
    if not image_bytes:
        raise ImageAnalysisError("The uploaded image was empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        max_mb = max(1, MAX_IMAGE_BYTES // (1024 * 1024))
        raise ImageAnalysisError(f"Image uploads must be {max_mb} MB or smaller.")
    return normalized_mime


class ImageAnalysisAgent:
    """
    Vision intake layer for uploaded clinical images.

    It does not diagnose. It only decides whether the image is appropriate for
    medical analysis and extracts observable findings/search terms for the
    evidence-backed clinical pipeline.
    """

    def __init__(self, llm) -> None:
        self.llm = llm
        self.model = os.getenv(
            "OPENAI_VISION_MODEL",
            os.getenv("OPENAI_MODEL", getattr(llm, "ANSWER_MODEL", "gpt-4o-mini")),
        )

    def inspect(
        self,
        image_bytes: bytes,
        mime_type: str,
        user_note: str = "",
        user_profile: Optional[Dict] = None,
        filename: str = "",
    ) -> Dict:
        normalized_mime = validate_image_upload(image_bytes, mime_type, filename)
        data_url = self._to_data_url(image_bytes, normalized_mime)
        profile_text = self._profile_summary(user_profile or {})

        response = self.llm.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict medical-image intake guardrail for a clinical education assistant.\n"
                        "Your job is ONLY to classify the uploaded image and extract observable visual findings.\n\n"
                        "Accept only clinically relevant images: visible skin changes, rashes, wounds, bruising, "
                        "swelling, eye/throat/dental findings, medication labels, test strips, medical devices, "
                        "or other clear health-related images.\n"
                        "Reject non-medical images: scenery, food, pets, objects, social photos, unrelated documents, "
                        "or any image where no health-related concern is visible.\n\n"
                        "Rules:\n"
                        "- Do not diagnose.\n"
                        "- Do not identify the person or infer age, sex, pregnancy, ethnicity, or skin tone beyond "
                        "what is supplied in the profile.\n"
                        "- Use cautious observation language: visible, appears, pattern, colour change.\n"
                        "- If image quality is too poor for safe review, set is_medical_image=false and explain.\n"
                        "- Return only valid JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Stored user profile:\n{profile_text}\n\n"
                                f"User note with upload:\n{(user_note or 'No note supplied.').strip()}\n\n"
                                "Return JSON with exactly these keys:\n"
                                "{\n"
                                '  "is_medical_image": boolean,\n'
                                '  "medical_relevance_confidence": "high" | "medium" | "low",\n'
                                '  "image_focus": string,\n'
                                '  "body_region_or_subject": string,\n'
                                '  "observable_findings": string[],\n'
                                '  "colour_or_texture_changes": string[],\n'
                                '  "visible_red_flag_clues": string[],\n'
                                '  "evidence_search_queries": string[],\n'
                                '  "reason_if_rejected": string\n'
                                "}\n\n"
                                "Search queries should be clinical evidence queries, not diagnoses asserted as fact."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": "high"},
                        },
                    ],
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
            max_completion_tokens=900,
        )

        raw = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ImageAnalysisError("The image analysis model returned an unreadable result.") from exc

        return self._normalize_result(parsed, normalized_mime, filename)

    def build_clinical_question(self, visual_result: Dict, user_note: str = "") -> str:
        findings = self._render_list(visual_result.get("observable_findings"))
        colour_changes = self._render_list(visual_result.get("colour_or_texture_changes"))
        red_flags = self._render_list(visual_result.get("visible_red_flag_clues"))
        search_queries = self._render_list(visual_result.get("evidence_search_queries"))

        return (
            "Image-based clinical question for agentic evidence review.\n\n"
            "The user uploaded a clinical image. The vision intake step has already rejected non-medical "
            "uploads and has produced only observable findings, not a diagnosis.\n\n"
            f"User note: {(user_note or 'No note supplied.').strip()}\n"
            f"Image focus: {visual_result.get('image_focus') or 'not specified'}\n"
            f"Body region or subject: {visual_result.get('body_region_or_subject') or 'not specified'}\n"
            f"Observable findings:\n{findings}\n\n"
            f"Colour or texture changes:\n{colour_changes}\n\n"
            f"Visible red-flag clues to consider:\n{red_flags}\n\n"
            f"Evidence search queries requested:\n{search_queries}\n\n"
            "Task: Use the stored patient profile, age, medications, allergies, conditions, symptoms, "
            "and uploaded records where relevant. Search formal guidance and biomedical literature "
            "(including PubMed/systematic review evidence when available). Provide an evidence-cited "
            "analysis of what the visual findings could suggest, what cannot be determined from an image, "
            "red flags that require urgent care, and specific next steps. Do not present a definitive "
            "diagnosis from the image."
        )

    @staticmethod
    def _to_data_url(image_bytes: bytes, mime_type: str) -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _profile_summary(profile: Dict) -> str:
        parts: List[str] = []
        age = compute_current_age(profile.get("date_of_birth", ""))
        if age is not None:
            parts.append(f"Age: {age} years")
        sex = (profile.get("biological_sex") or "").strip()
        if sex and sex != "Prefer not to say":
            parts.append(f"Biological sex: {sex}")
        role = (profile.get("clinical_role") or profile.get("role") or "").strip()
        if role:
            parts.append(f"Account role: {role}")
        care_context = (profile.get("care_context") or "").strip()
        if care_context:
            parts.append(f"Care context: {care_context}")
        return "\n".join(parts) if parts else "No demographic profile details recorded."

    @classmethod
    def _normalize_result(cls, payload: Dict, mime_type: str, filename: str) -> Dict:
        confidence = str(payload.get("medical_relevance_confidence") or "low").strip().lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"

        observable_findings = cls._clean_list(payload.get("observable_findings"))
        search_queries = cls._clean_list(payload.get("evidence_search_queries"))
        is_medical = bool(payload.get("is_medical_image")) and confidence in {"high", "medium"}
        if is_medical and not observable_findings and not search_queries:
            is_medical = False

        reason = str(payload.get("reason_if_rejected") or "").strip()
        if not is_medical and not reason:
            reason = (
                "This image does not contain a clear medical concern that can be safely analysed."
            )

        return {
            "analysis_status": "accepted" if is_medical else "rejected",
            "is_medical_image": is_medical,
            "medical_relevance_confidence": confidence,
            "image_focus": str(payload.get("image_focus") or "").strip(),
            "body_region_or_subject": str(payload.get("body_region_or_subject") or "").strip(),
            "observable_findings": observable_findings[:8],
            "colour_or_texture_changes": cls._clean_list(payload.get("colour_or_texture_changes"))[:8],
            "visible_red_flag_clues": cls._clean_list(payload.get("visible_red_flag_clues"))[:6],
            "evidence_search_queries": search_queries[:5],
            "reason_if_rejected": reason,
            "uploaded_image_mime": mime_type,
            "uploaded_image_name": filename,
        }

    @staticmethod
    def _clean_list(value) -> List[str]:
        if not isinstance(value, list):
            return []
        cleaned = []
        seen = set()
        for item in value:
            text = str(item or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text[:240])
        return cleaned

    @classmethod
    def _render_list(cls, value) -> str:
        items = cls._clean_list(value)
        if not items:
            return "- Not specified"
        return "\n".join(f"- {item}" for item in items)
