from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from backend.user_store import compute_current_age


SUPPORTED_DOCUMENT_MIME_TYPES = {"application/pdf"}
# Larger default than images (IMAGE_ANALYSIS_MAX_BYTES) -- PDFs, especially
# multi-page lab reports or clinical letters, commonly run bigger.
MAX_DOCUMENT_BYTES = int(os.getenv("DOCUMENT_ANALYSIS_MAX_BYTES", str(15 * 1024 * 1024)))
# Keeps the intake prompt bounded regardless of document length -- mirrors the
# truncation idiom used elsewhere in this codebase for document text (see
# backend/document_relevance_agent.py's _MAX_INPUT_CHARS).
MAX_DOCUMENT_TEXT_CHARS = 6000


class DocumentAnalysisError(ValueError):
    """Raised when an uploaded document cannot be accepted for analysis."""


def normalize_document_mime_type(mime_type: str, filename: str = "") -> str:
    cleaned = (mime_type or "").split(";")[0].strip().lower()
    if cleaned:
        return cleaned

    suffix = (filename or "").rsplit(".", 1)[-1].lower()
    return {"pdf": "application/pdf"}.get(suffix, "")


def validate_document_upload(document_bytes: bytes, mime_type: str, filename: str = "") -> str:
    normalized_mime = normalize_document_mime_type(mime_type, filename)
    if normalized_mime not in SUPPORTED_DOCUMENT_MIME_TYPES:
        raise DocumentAnalysisError("Upload a PDF document.")
    if not document_bytes:
        raise DocumentAnalysisError("The uploaded document was empty.")
    if len(document_bytes) > MAX_DOCUMENT_BYTES:
        max_mb = max(1, MAX_DOCUMENT_BYTES // (1024 * 1024))
        raise DocumentAnalysisError(f"Document uploads must be {max_mb} MB or smaller.")
    return normalized_mime


class DocumentAnalysisAgent:
    """
    Text intake layer for uploaded clinical/reference documents (lab results,
    clinical letters, discharge summaries, research papers, guideline PDFs).

    It does not diagnose. It only decides whether the document is appropriate
    for medical analysis and extracts observable findings/search terms for
    the evidence-backed clinical pipeline -- same contract as
    ImageAnalysisAgent, just over extracted text instead of a vision call.
    """

    def __init__(self, llm) -> None:
        self.llm = llm
        self.model = getattr(llm, "AUX_MODEL", getattr(llm, "ANSWER_MODEL", "gpt-4o-mini"))

    def inspect(
        self,
        document_text: str,
        filename: str = "",
        user_note: str = "",
        user_profile: Optional[Dict] = None,
    ) -> Dict:
        profile_text = self._profile_summary(user_profile or {})
        truncated_text = (document_text or "").strip()[:MAX_DOCUMENT_TEXT_CHARS]

        response = self.llm.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict clinical-document intake guardrail for a clinical education "
                        "assistant.\n"
                        "Your job is ONLY to classify the uploaded document and extract observable "
                        "findings/content, not to diagnose or give advice.\n\n"
                        "Accept only clinically relevant documents: lab or imaging reports, clinical "
                        "letters or discharge summaries, research papers or clinical guideline PDFs, "
                        "prescription or medication lists, or other clear health-related documents.\n"
                        "Reject non-medical documents: invoices, contracts, unrelated correspondence, "
                        "or any document with no health-related content.\n\n"
                        "Rules:\n"
                        "- Do not diagnose.\n"
                        "- Do not identify the person or infer age, sex, pregnancy, or ethnicity beyond "
                        "what is supplied in the profile.\n"
                        "- Use cautious observation language: reports, states, indicates.\n"
                        "- If the extracted text is too sparse or garbled for safe review, set "
                        "is_medical_document=false and explain.\n"
                        "- Return only valid JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Stored user profile:\n{profile_text}\n\n"
                        f"User note with upload:\n{(user_note or 'No note supplied.').strip()}\n\n"
                        f"Filename: {filename or 'not supplied'}\n\n"
                        f"Extracted document text:\n{truncated_text or '(no text extracted)'}\n\n"
                        "Return JSON with exactly these keys:\n"
                        "{\n"
                        '  "is_medical_document": boolean,\n'
                        '  "medical_relevance_confidence": "high" | "medium" | "low",\n'
                        '  "document_type": string,\n'
                        '  "key_findings": string[],\n'
                        '  "notable_values": string[],\n'
                        '  "evidence_search_queries": string[],\n'
                        '  "reason_if_rejected": string\n'
                        "}\n\n"
                        "Search queries should be clinical evidence queries, not diagnoses asserted as fact."
                    ),
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
            raise DocumentAnalysisError("The document analysis model returned an unreadable result.") from exc

        return self._normalize_result(parsed, filename)

    def build_clinical_question(self, inspect_result: Dict, user_note: str = "") -> str:
        findings = self._render_list(inspect_result.get("key_findings"))
        notable_values = self._render_list(inspect_result.get("notable_values"))
        search_queries = self._render_list(inspect_result.get("evidence_search_queries"))

        return (
            "Document-based clinical question for agentic evidence review.\n\n"
            "The user uploaded a clinical document. The intake step has already rejected "
            "non-medical uploads and has produced only observable findings, not a diagnosis.\n\n"
            f"User note: {(user_note or 'No note supplied.').strip()}\n"
            f"Document type: {inspect_result.get('document_type') or 'not specified'}\n"
            f"Key findings:\n{findings}\n\n"
            f"Notable values:\n{notable_values}\n\n"
            f"Evidence search queries requested:\n{search_queries}\n\n"
            "Task: Use the stored patient profile, age, medications, allergies, conditions, symptoms, "
            "and uploaded records where relevant. Search formal guidance and biomedical literature "
            "(including PubMed/systematic review evidence when available). Provide an evidence-cited "
            "analysis of what the document findings could suggest, what cannot be determined from the "
            "document alone, red flags that require urgent care, and specific next steps. Do not "
            "present a definitive diagnosis from the document."
        )

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
    def _normalize_result(cls, payload: Dict, filename: str) -> Dict:
        confidence = str(payload.get("medical_relevance_confidence") or "low").strip().lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"

        key_findings = cls._clean_list(payload.get("key_findings"))
        search_queries = cls._clean_list(payload.get("evidence_search_queries"))
        is_medical = bool(payload.get("is_medical_document")) and confidence in {"high", "medium"}
        if is_medical and not key_findings and not search_queries:
            is_medical = False

        reason = str(payload.get("reason_if_rejected") or "").strip()
        if not is_medical and not reason:
            reason = (
                "This document does not contain clear medical content that can be safely analysed."
            )

        return {
            "analysis_status": "accepted" if is_medical else "rejected",
            "is_medical_document": is_medical,
            "medical_relevance_confidence": confidence,
            "document_type": str(payload.get("document_type") or "").strip(),
            "key_findings": key_findings[:8],
            "notable_values": cls._clean_list(payload.get("notable_values"))[:8],
            "evidence_search_queries": search_queries[:5],
            "reason_if_rejected": reason,
            "uploaded_document_name": filename,
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
