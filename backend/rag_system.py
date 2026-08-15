from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Generator, List, Optional, Tuple
from uuid import uuid4
import re

import numpy as np

from backend.anonymizer import DocumentAnonymizer
from backend.anonymization_agent import AnonymizationAgent
from backend.duplicate_detection_agent import DuplicateDetectionAgent
from backend.document_relevance_agent import DocumentRelevanceAgent
from backend.document_extractor import (
    extract_health_data_from_document,
    extract_health_data_from_images,
)
from backend.clinical_orchestrator import ClinicalOrchestrator
from backend.care_plan_store import CarePlanStore
from backend.conversation_context import build_conversation_context
from backend.document_analysis_agent import DocumentAnalysisAgent, DocumentAnalysisError
from backend.image_analysis_agent import ImageAnalysisAgent, ImageAnalysisError
from backend.image_generator import ImageGenerator
from backend.medication_checker import MedicationInteractionChecker, check_allergy_conflicts
from backend.memory_store import MemoryStore
from backend.product_config import PRODUCT_NAME, is_clinician_role
from backend.symptom_tracker import build_symptom_pattern_summary
from backend.triage_summary import build_default_triage, normalize_triage_output
from backend.video_generator import VideoGenerator
from backend.moderation_ml import ModerationEnsemble
from backend.official_guidance import OfficialGuidanceEngine
from backend.pubmed_search import PubMedCentralSearcher
from backend.query_expander import QueryExpander
from backend.relationship_engine import (
    derive_relationships,
    merge_relationships,
    relationship_summary,
)
from backend.safety_review import build_safety_reviews
from backend.summarizer import LLMHelper
from backend.user_store import UserStore
from backend.clinical_context_guard import (
    ClinicalContextDecision,
    validate_generated_answer,
)
from backend.agentic_health_contract import (
    remove_internal_language,
    remove_unknown_citations,
    validate_user_facing_language,
)
from backend.utils import (
    build_excerpt,
    extract_text_from_pdf,
    extract_text_from_pdf_bytes,
    render_pdf_pages_to_images,
    render_vital_for_prompt,
)


# Fail-closed policy (Evidence Ledger v2, #8): when claim-source verification
# can't be run or a required correction can't be applied even after a retry,
# this replaces the answer instead of shipping unverified text. Previously
# both failure paths silently fell back to the raw, unverified answer.
SAFE_VERIFICATION_FALLBACK_MESSAGE = (
    "## Unable To Verify This Answer\n\n"
    "I wasn't able to verify this answer against its sources right now, so I "
    "can't show it. Please try asking again in a moment, or check with a "
    "clinician for guidance on this question."
)


def _retry_once(fn, *args, **kwargs):
    """Runs fn once, retries exactly once more on any exception, and returns
    (result, succeeded). No backoff -- this wraps a single extra LLM call,
    not a batch job. Callers decide what "still failing" means for them."""
    for attempt in range(2):
        try:
            return fn(*args, **kwargs), True
        except Exception as exc:
            last_exc = exc
            if attempt == 1:
                print(f"[Orchestrator] {getattr(fn, '__name__', fn)} failed after retry: {last_exc}")
    return None, False


_CHAT_RECORD_SIGNAL_RE = re.compile(
    r"\b(?:i\s+(?:am|'m|have|had|take|use|started|was|wasn't|stopped)|"
    r"i've\s+(?:been|started|had)|my\s+(?:medication|medicine|allergy|reaction|"
    r"diagnosis|blood\s+pressure|heart\s+rate)|allerg(?:y|ic)|diagnosed\s+with|"
    r"prescribed\s+(?:to|for)\s+me|because\s+of|caus(?:e|es|ed|ing)|trigger(?:s|ed)?|"
    r"after\s+starting|since\s+starting)\b",
    re.IGNORECASE,
)


class RAGEngine:
    """
    Retrieval-augmented engine that combines user context, PubMed evidence, and
    audit metadata for a professional clinical chat experience.
    """

    def __init__(self, embedding_dir: str = "data/uploads"):
        self.embedding_dir = Path(embedding_dir)
        self.query_expander = QueryExpander()
        self.memory = MemoryStore()
        self.pubmed = PubMedCentralSearcher()
        self.anonymizer = DocumentAnonymizer()
        self.llm = LLMHelper()
        self.moderation = ModerationEnsemble()
        self.official_guidance = OfficialGuidanceEngine()
        self._primed_users: set[str] = set()
        self._orchestrator = ClinicalOrchestrator(
            memory=self.memory,
            pubmed=self.pubmed,
            official_guidance=self.official_guidance,
            llm=self.llm,
            query_expander=self.query_expander,
            moderation=self.moderation,
        )
        self._image_analysis_agent = ImageAnalysisAgent(self.llm)
        self._document_analysis_agent = DocumentAnalysisAgent(self.llm)
        self._anonymization_agent = AnonymizationAgent(self.llm)
        self._duplicate_agent = DuplicateDetectionAgent(self.llm)
        self._relevance_agent = DocumentRelevanceAgent(self.llm)
        self._image_generator = ImageGenerator()
        self._video_generator = VideoGenerator()
        self._medication_checker = MedicationInteractionChecker()

    def restore_user_context(self, user: Optional[str]) -> None:
        """
        Syncs persisted user document, symptom, and medication summaries into memory.
        """
        if not user:
            return

        normalized_user = user.strip().lower()
        pending_entries = []
        for summary_record in UserStore.get_document_summaries(normalized_user):
            summary_text = summary_record.get("summary", "").strip()
            if not summary_text:
                continue

            filename = summary_record.get("file", "uploaded document")
            pending_entries.append(
                {
                    "text": summary_text,
                    "metadata": {
                        "type": "user_summary",
                        "source": filename,
                        "title": f"User-uploaded record: {filename}",
                        "section": "document summary",
                        "stored_path": summary_record.get("stored_path", ""),
                        "entry_key": f"{normalized_user}:upload:{filename}",
                    },
                    "user": normalized_user,
                    "entry_key": f"{normalized_user}:upload:{filename}",
                }
            )

        symptom_summary = build_symptom_pattern_summary(
            UserStore.get_symptom_logs(normalized_user, limit=None)
        )
        if symptom_summary:
            pending_entries.append(
                {
                    "text": symptom_summary,
                    "metadata": {
                        "type": "user_summary",
                        "source": "Symptom tracker",
                        "title": "Tracked symptom timeline",
                        "section": "symptom tracking",
                        "entry_key": f"{normalized_user}:tracker:symptoms",
                    },
                    "user": normalized_user,
                    "entry_key": f"{normalized_user}:tracker:symptoms",
                }
            )
        else:
            self.memory.remove_entry(f"{normalized_user}:tracker:symptoms")

        condition_summary = self._build_condition_memory_summary(
            UserStore.get_conditions(normalized_user)
        )
        if condition_summary:
            pending_entries.append(
                {
                    "text": condition_summary,
                    "metadata": {
                        "type": "user_summary",
                        "source": "Condition history",
                        "title": "Recorded conditions and history",
                        "section": "conditions and history",
                        "entry_key": f"{normalized_user}:tracker:conditions",
                    },
                    "user": normalized_user,
                    "entry_key": f"{normalized_user}:tracker:conditions",
                }
            )
        else:
            self.memory.remove_entry(f"{normalized_user}:tracker:conditions")

        medication_summary = self._build_medication_memory_summary(
            UserStore.get_medications(normalized_user)
        )
        if medication_summary:
            pending_entries.append(
                {
                    "text": medication_summary,
                    "metadata": {
                        "type": "user_summary",
                        "source": "Medication list",
                        "title": "Current medication list",
                        "section": "medication list",
                        "entry_key": f"{normalized_user}:tracker:medications",
                    },
                    "user": normalized_user,
                    "entry_key": f"{normalized_user}:tracker:medications",
                }
            )
        else:
            self.memory.remove_entry(f"{normalized_user}:tracker:medications")

        vitals_summary = self._build_vitals_memory_summary(
            UserStore.get_vitals(normalized_user, limit=None)
        )
        if vitals_summary:
            pending_entries.append(
                {
                    "text": vitals_summary,
                    "metadata": {
                        "type": "user_summary",
                        "source": "Vitals and lab results",
                        "title": "Recorded vitals and lab results",
                        "section": "vitals and labs",
                        "entry_key": f"{normalized_user}:tracker:vitals",
                    },
                    "user": normalized_user,
                    "entry_key": f"{normalized_user}:tracker:vitals",
                }
            )
        else:
            self.memory.remove_entry(f"{normalized_user}:tracker:vitals")

        allergies_summary = self._build_allergies_memory_summary(
            UserStore.get_allergies(normalized_user)
        )
        if allergies_summary:
            pending_entries.append(
                {
                    "text": allergies_summary,
                    "metadata": {
                        "type": "user_summary",
                        "source": "Allergy record",
                        "title": "Known allergies and adverse reactions",
                        "section": "allergies",
                        "entry_key": f"{normalized_user}:tracker:allergies",
                    },
                    "user": normalized_user,
                    "entry_key": f"{normalized_user}:tracker:allergies",
                }
            )
        else:
            self.memory.remove_entry(f"{normalized_user}:tracker:allergies")

        self.memory.upsert_entries(pending_entries)
        self._primed_users.add(normalized_user)

    def _capture_explicit_chat_records(
        self, user: str, user_message: str
    ) -> List[Dict]:
        """Promote explicit first-person chat facts into structured records.

        This is deliberately additive. A chat statement can add or enrich a
        record but never deletes an existing medicine, allergy, or diagnosis.
        Questions, negations, and uncertain assistant inferences are excluded by
        the extractor prompt and this method never reads assistant messages.
        """
        if (
            not user
            or not _CHAT_RECORD_SIGNAL_RE.search(user_message or "")
            or not hasattr(self.llm, "extract_explicit_chat_record_facts")
        ):
            return []
        try:
            extracted = self.llm.extract_explicit_chat_record_facts(user_message)
        except Exception as exc:
            print(f"[ChatRecordCapture] extraction failed: {exc}")
            return []
        if not isinstance(extracted, dict):
            return []

        updates: List[Dict] = []
        source_note = "Captured from the patient's explicit chat statement."
        message_lower = (user_message or "").lower()

        def blocked_record(name: str, record_type: str) -> bool:
            escaped = re.escape(name.lower())
            if record_type == "medication":
                patterns = (
                    rf"\b(?:do\s+not|don't|not|never|no\s+longer|stopped)\s+"
                    rf"(?:take|taking|use|using)\b.{{0,40}}\b{escaped}\b",
                    rf"\b(?:can|could|should|may)\s+i\s+(?:take|use)\b.{{0,30}}\b{escaped}\b",
                )
            elif record_type == "allergy":
                patterns = (
                    rf"\b(?:not|never)\s+allergic\s+to\b.{{0,30}}\b{escaped}\b",
                    rf"\b(?:am\s+i|could\s+i\s+be|can\s+i\s+be)\s+allergic\b.{{0,30}}\b{escaped}\b",
                )
            elif record_type == "condition":
                patterns = (
                    rf"\b(?:do\s+not|don't|not|never)\s+have\b.{{0,30}}\b{escaped}\b",
                    rf"\b(?:do|could|might|may)\s+i\s+have\b.{{0,30}}\b{escaped}\b",
                )
            else:
                patterns = ()
            return any(re.search(pattern, message_lower) for pattern in patterns)

        existing_medications = {
            str(item.get("name") or "").strip().lower(): item
            for item in UserStore.get_medications(user)
            if str(item.get("name") or "").strip()
        }
        vague_medicine_names = {
            "medicine", "medication", "medications", "tablet", "tablets",
            "capsule", "capsules", "antibiotic", "antibiotics",
        }
        for item in extracted.get("medications") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if (
                not name
                or name.lower() in vague_medicine_names
                or blocked_record(name, "medication")
            ):
                continue
            existing = existing_medications.get(name.lower(), {})
            payload = {
                "name": name,
                "dose": str(item.get("dose") or existing.get("dose") or "").strip(),
                "schedule": str(
                    item.get("schedule") or existing.get("schedule") or ""
                ).strip(),
                "reason": str(
                    item.get("reason") or existing.get("reason") or ""
                ).strip(),
                "started_on": str(
                    item.get("started_on") or existing.get("started_on") or ""
                ).strip(),
                "notes": str(existing.get("notes") or source_note).strip(),
            }
            if UserStore.save_medication(user, payload):
                updates.append({"record_type": "medication", "name": name})

        existing_allergies = {
            str(item.get("name") or "").strip().lower(): item
            for item in UserStore.get_allergies(user)
            if str(item.get("name") or "").strip()
        }
        for item in extracted.get("allergies") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name or blocked_record(name, "allergy"):
                continue
            existing = existing_allergies.get(name.lower(), {})
            payload = {
                "name": name,
                "reaction": str(
                    item.get("reaction") or existing.get("reaction") or ""
                ).strip(),
                "severity": str(
                    item.get("severity") or existing.get("severity") or "unknown"
                ).strip(),
                "allergy_type": str(
                    item.get("allergy_type")
                    or existing.get("allergy_type")
                    or "other"
                ).strip(),
                "confirmed": True,
                "notes": str(existing.get("notes") or source_note).strip(),
            }
            if UserStore.save_allergy(user, payload):
                updates.append({"record_type": "allergy", "name": name})

        existing_conditions = {
            str(item.get("name") or "").strip().lower(): item
            for item in UserStore.get_conditions(user)
            if str(item.get("name") or "").strip()
        }
        for item in extracted.get("conditions") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name or blocked_record(name, "condition"):
                continue
            existing = existing_conditions.get(name.lower(), {})
            payload = {
                "name": name,
                "status": str(
                    item.get("status") or existing.get("status") or "unknown"
                ).strip(),
                "recorded_on": str(
                    item.get("recorded_on") or existing.get("recorded_on") or ""
                ).strip(),
                "notes": str(
                    item.get("notes") or existing.get("notes") or source_note
                ).strip(),
            }
            if UserStore.save_condition(user, payload):
                updates.append({"record_type": "condition", "name": name})

        existing_symptoms = {
            (
                str(item.get("symptom") or "").strip().lower(),
                str(item.get("logged_for") or "").strip(),
            )
            for item in UserStore.get_symptom_logs(user, limit=None)
        }
        for item in extracted.get("symptoms") or []:
            if not isinstance(item, dict):
                continue
            symptom = str(item.get("symptom") or "").strip()
            logged_for = str(item.get("logged_for") or date.today().isoformat()).strip()
            if not symptom or (symptom.lower(), logged_for) in existing_symptoms:
                continue
            saved = UserStore.add_symptom_log(
                user,
                symptom=symptom,
                logged_for=logged_for,
                severity=item.get("severity", 0),
                triggers=str(item.get("triggers") or "").strip(),
                notes=str(item.get("notes") or source_note).strip(),
            )
            if saved:
                updates.append({"record_type": "symptom", "name": symptom})

        existing_vitals = {
            (
                str(item.get("type") or "").strip().lower(),
                str(item.get("value") or "").strip().lower(),
                str(item.get("recorded_on") or "").strip(),
            )
            for item in UserStore.get_vitals(user, limit=None)
        }
        for item in extracted.get("vitals") or []:
            if not isinstance(item, dict):
                continue
            vital_type = str(item.get("type") or "").strip()
            value = str(item.get("value") or "").strip()
            recorded_on = str(item.get("recorded_on") or date.today().isoformat()).strip()
            key = (vital_type.lower(), value.lower(), recorded_on)
            if not vital_type or not value or key in existing_vitals:
                continue
            saved = UserStore.save_vitals_entry(
                user,
                {
                    "type": vital_type,
                    "value": value,
                    "unit": str(item.get("unit") or "").strip(),
                    "recorded_on": recorded_on,
                    "notes": str(item.get("notes") or source_note).strip(),
                },
            )
            if saved:
                updates.append({"record_type": "vital", "name": vital_type})

        relationships = [
            {**item, "source": item.get("source") or "conversation"}
            for item in (extracted.get("relationships") or [])
            if isinstance(item, dict)
        ]
        asks_causal_question = bool(
            re.search(
                r"\b(?:can|could|does|did|is|are)\b.{0,50}"
                r"\b(?:cause|causes|trigger|triggers|worsen|worsens|improve|improves)\b",
                message_lower,
            )
            and not re.search(r"\b(?:i\s+(?:think|suspect|believe)|in\s+my\s+case)\b", message_lower)
        )
        if asks_causal_question:
            relationships = []
        relationships = merge_relationships(
            relationships,
            derive_relationships(
                medications=[
                    item for item in (extracted.get("medications") or [])
                    if isinstance(item, dict)
                ],
                allergies=[
                    item for item in (extracted.get("allergies") or [])
                    if isinstance(item, dict)
                ],
                conditions=[
                    item for item in (extracted.get("conditions") or [])
                    if isinstance(item, dict)
                ],
                symptom_logs=[
                    item for item in (extracted.get("symptoms") or [])
                    if isinstance(item, dict)
                ],
                vitals=[
                    item for item in (extracted.get("vitals") or [])
                    if isinstance(item, dict)
                ],
                source="conversation",
            ),
        )
        if relationships:
            before = len(UserStore.get_clinical_relationships(user))
            after = UserStore.save_clinical_relationships(user, relationships)
            if len(after) >= before:
                for item in relationships:
                    updates.append(
                        {
                            "record_type": "relationship",
                            "name": (
                                f"{item.get('source_name', '')} "
                                f"{str(item.get('relation') or '').replace('_', ' ')} "
                                f"{item.get('target_name', '')}"
                            ).strip(),
                        }
                    )
        return updates

    def ingest_documents(
        self,
        user: Optional[str] = None,
        file_paths: Optional[List[Path]] = None,
        content_hashes: Optional[Dict[str, str]] = None,
    ) -> List[Dict]:
        """
        Loads uploaded documents, anonymizes them, summarizes them, and persists
        a retrieval-friendly document summary per user.
        """
        normalized_user = user.strip().lower() if user else None
        explicit_upload_paths = file_paths is not None
        documents = [
            Path(path)
            for path in (file_paths or self._default_document_paths(normalized_user))
        ]
        indexed_documents = []
        known_uploads = (
            {item.get("file") for item in UserStore.get_uploads(normalized_user)}
            if normalized_user
            else set()
        )

        for path in documents:
            if not path.exists() or path.suffix.lower() != ".pdf":
                continue

            is_new_upload = explicit_upload_paths or path.name not in known_uploads
            text_error = ""
            try:
                raw_text = extract_text_from_pdf(path)
            except Exception as exc:
                raw_text = ""
                text_error = f"PDF text could not be read: {exc}"

            # Two agentic gates before any processing happens, both of which
            # reject outright (delete the file, skip summarization/extraction/
            # indexing entirely) rather than partially process. Only run for
            # genuinely new uploads with readable text -- a re-index pass over
            # already-known files shouldn't re-run either check.
            if not text_error and is_new_upload:
                relevance = self._relevance_agent.check(raw_text, path.name)
                if not relevance.get("is_relevant", True):
                    path.unlink(missing_ok=True)
                    indexed_documents.append(
                        {
                            "file": path.name,
                            "rejected": True,
                            "rejection_type": "unrelated",
                            "rejection_reason": relevance.get("reason")
                            or "This does not appear to be a personal health or clinical document.",
                        }
                    )
                    continue

                if normalized_user:
                    existing_summaries = UserStore.get_document_summaries(
                        normalized_user
                    )
                    duplicate_match = (
                        self._duplicate_agent.check(
                            raw_text, path.name, existing_summaries
                        )
                        if existing_summaries
                        else None
                    )
                    if duplicate_match:
                        path.unlink(missing_ok=True)
                        indexed_documents.append(
                            {
                                "file": path.name,
                                "rejected": True,
                                "rejection_type": "duplicate",
                                "rejection_reason": "This document has already been uploaded.",
                                "duplicate_of": duplicate_match["matches_file"],
                            }
                        )
                        continue

            summary_error = ""
            if text_error:
                summary = "Uploaded document could not be read as text."
                summary_error = text_error
            else:
                try:
                    regex_redacted = self.anonymizer.anonymize(raw_text)
                    anonymized = self._anonymization_agent.anonymize(regex_redacted)
                    summary = self.llm.summarize_user_health_record(anonymized)
                except Exception as exc:
                    summary_error = f"Document summary failed: {exc}"
                    summary = (
                        build_excerpt(raw_text, max_chars=900)
                        or "Uploaded document could not be summarized."
                    )

            memory_key = f"{normalized_user or 'global'}:upload:{path.name}"
            self.memory.upsert_entries(
                [
                    {
                        "text": summary,
                        "metadata": {
                            "type": "user_summary",
                            "source": path.name,
                            "title": f"User-uploaded record: {path.name}",
                            "section": "document summary",
                            "stored_path": str(path),
                            "entry_key": memory_key,
                        },
                        "user": normalized_user,
                        "entry_key": memory_key,
                    }
                ]
            )

            extracted: Dict = {}
            if normalized_user:
                if is_new_upload:
                    UserStore.add_upload(
                        normalized_user,
                        path.name,
                        stored_path=str(path),
                        content_hash=(content_hashes or {}).get(path.name),
                    )
                    known_uploads.add(path.name)

                    # Auto-populate health data from the document (new uploads only)
                    extracted = extract_health_data_from_document(raw_text, path.name)
                    source_note = f"Auto-extracted from {path.name}"

                    # Some exports (portal printouts, scans) render lab values inside a
                    # gauge/badge/chart widget rather than as real text -- the text layer
                    # then has the surrounding labels but none of the actual numbers, so
                    # text-based extraction comes back empty even though the page clearly
                    # has data. Fall back to reading the rendered page images directly.
                    has_any_data = any(
                        extracted.get(key)
                        for key in ("vitals", "medications", "allergies", "conditions")
                    )
                    if not has_any_data:
                        try:
                            page_images = render_pdf_pages_to_images(path)
                            vision_extracted = extract_health_data_from_images(
                                page_images, path.name
                            )
                        except Exception as exc:
                            vision_extracted = {
                                "extraction_errors": [f"Vision fallback failed: {exc}"]
                            }
                        vision_has_data = any(
                            vision_extracted.get(key)
                            for key in (
                                "vitals",
                                "medications",
                                "allergies",
                                "conditions",
                            )
                        )
                        if vision_has_data:
                            vision_extracted["extraction_method"] = "vision_fallback"
                            extracted = vision_extracted
                        else:
                            extracted.setdefault("extraction_errors", []).extend(
                                vision_extracted.get("extraction_errors")
                                or [
                                    "Vision-based fallback extraction also found no structured health data."
                                ]
                            )

                    # Vitals / lab results -- content-based dedup (type + value + date)
                    existing_vitals = UserStore.get_vitals(normalized_user, limit=None)
                    existing_keys = {
                        (
                            v.get("type", "").lower(),
                            v.get("value", "").lower(),
                            v.get("recorded_on", ""),
                        )
                        for v in existing_vitals
                    }
                    for vital in extracted.get("vitals", []):
                        vtype = str(vital.get("type") or "").strip().lower()
                        vval = str(vital.get("value") or "").strip().lower()
                        vdate = str(vital.get("recorded_on") or "").strip()
                        if not vtype or not vval:
                            continue
                        if (vtype, vval, vdate) in existing_keys:
                            continue
                        existing_keys.add((vtype, vval, vdate))
                        UserStore.save_vitals_entry(
                            normalized_user,
                            {
                                "type": vtype,
                                "value": str(vital.get("value") or "").strip(),
                                "unit": str(vital.get("unit") or "").strip(),
                                "recorded_on": vdate,
                                "notes": (
                                    f"{vital.get('notes', '').strip()} [{source_note}]"
                                    if vital.get("notes")
                                    else f"[{source_note}]"
                                ).strip(),
                            },
                        )

                    # Medications -- UserStore.save_medication deduplicates by name
                    for med in extracted.get("medications", []):
                        if not str(med.get("name") or "").strip():
                            continue
                        UserStore.save_medication(
                            normalized_user,
                            {
                                "name": str(med.get("name") or "").strip(),
                                "dose": str(med.get("dose") or "").strip(),
                                "schedule": str(med.get("schedule") or "").strip(),
                                "reason": str(med.get("reason") or "").strip(),
                                "started_on": str(med.get("started_on") or "").strip(),
                                "notes": (
                                    f"{med.get('notes', '').strip()} [{source_note}]"
                                    if med.get("notes")
                                    else f"[{source_note}]"
                                ).strip(),
                            },
                        )

                    # Allergies -- UserStore.save_allergy deduplicates by name
                    for allergy in extracted.get("allergies", []):
                        if not str(allergy.get("name") or "").strip():
                            continue
                        UserStore.save_allergy(
                            normalized_user,
                            {
                                "name": str(allergy.get("name") or "").strip(),
                                "reaction": str(allergy.get("reaction") or "").strip(),
                                "severity": str(
                                    allergy.get("severity") or "unknown"
                                ).strip(),
                                "allergy_type": str(
                                    allergy.get("allergy_type") or "other"
                                ).strip(),
                                "confirmed": bool(allergy.get("confirmed", True)),
                                "notes": f"[{source_note}]",
                            },
                        )

                    # Conditions / past history: UserStore.save_condition deduplicates by name.
                    for condition in extracted.get("conditions", []):
                        if isinstance(condition, dict):
                            condition_name = str(condition.get("name") or "").strip()
                            if not condition_name:
                                continue
                            condition_notes = str(condition.get("notes") or "").strip()
                            UserStore.save_condition(
                                normalized_user,
                                {
                                    "name": condition_name,
                                    "status": str(
                                        condition.get("status") or "unknown"
                                    ).strip(),
                                    "recorded_on": str(
                                        condition.get("recorded_on") or ""
                                    ).strip(),
                                    "notes": (
                                        f"{condition_notes} [{source_note}]"
                                        if condition_notes
                                        else f"[{source_note}]"
                                    ).strip(),
                                },
                            )
                        elif str(condition or "").strip():
                            UserStore.save_condition(
                                normalized_user,
                                {
                                    "name": str(condition).strip(),
                                    "status": "unknown",
                                    "notes": f"[{source_note}]",
                                },
                            )

                    document_relationships = merge_relationships(
                        [
                            {
                                **item,
                                "certainty": item.get("certainty") or "documented",
                                "source": f"document:{path.name}",
                            }
                            for item in (extracted.get("relationships") or [])
                            if isinstance(item, dict)
                        ],
                        derive_relationships(
                            medications=extracted.get("medications") or [],
                            allergies=extracted.get("allergies") or [],
                            conditions=extracted.get("conditions") or [],
                            vitals=extracted.get("vitals") or [],
                            source=f"document:{path.name}",
                        ),
                        [
                            {
                                "source_type": entity_type,
                                "source_name": str(entity.get(name_key) or "").strip(),
                                "relation": "recorded_in",
                                "target_type": "document",
                                "target_name": path.name,
                                "certainty": "documented",
                                "evidence": f"Extracted from {path.name}",
                                "source": f"document:{path.name}",
                            }
                            for entity_type, records, name_key in (
                                ("medication", extracted.get("medications") or [], "name"),
                                ("allergy", extracted.get("allergies") or [], "name"),
                                ("condition", extracted.get("conditions") or [], "name"),
                                ("vital", extracted.get("vitals") or [], "type"),
                            )
                            for entity in records
                            if isinstance(entity, dict)
                            and str(entity.get(name_key) or "").strip()
                        ],
                    )
                    if document_relationships:
                        UserStore.save_clinical_relationships(
                            normalized_user, document_relationships
                        )

                UserStore.save_document_summary(
                    normalized_user,
                    path.name,
                    summary,
                    stored_path=str(path),
                )

            indexed_documents.append(
                {
                    "file": path.name,
                    "stored_path": str(path),
                    "summary": summary,
                    "summary_error": summary_error,
                    "extracted": extracted,
                    "is_new": is_new_upload,
                }
            )

        if normalized_user and indexed_documents:
            try:
                self.refresh_longitudinal_memory_from_documents(
                    user=normalized_user,
                    indexed_documents=indexed_documents,
                )
            except Exception as exc:
                print(f"Longitudinal memory refresh failed after upload: {exc}")
            self._primed_users.add(normalized_user)

        return indexed_documents

    def handle_user_question(
        self,
        question: str,
        chat_history: Optional[List[dict]] = None,
        stream: bool = False,
        user: Optional[str] = None,
    ) -> Dict:
        """
        Responds to a user query with a structured payload that includes answer markdown,
        clickable sources, personal context traceability, and audit metadata.
        """
        del stream
        bundle = self._prepare_answer_bundle(
            question=question, user=user, chat_history=chat_history
        )
        if bundle["kind"] == "final":
            return self._enrich_prebuilt_payload(
                question=question, payload=bundle["payload"], user=user
            )

        _pd = bundle.get("policy_decision")
        clinical_decision = bundle.get("clinical_decision")
        if clinical_decision and clinical_decision.deterministic_response:
            role_key = (
                bundle.get("role_config").role_key
                if bundle.get("role_config")
                else "patient"
            )
            raw_answer = clinical_decision.render_markdown(
                role_key, bundle["combined_sources"]
            )
        else:
            raw_answer = self.llm.answer_question(
                question=question,
                context=bundle["full_context"],
                chat_history=bundle.get("previous_five_chat", []),
                stream=False,
                user_profile=bundle["user_profile"],
                source_briefings=bundle["combined_sources"],
                longitudinal_memory=bundle["longitudinal_memory_summary"],
                conversation_summary=bundle.get("conversation_summary", ""),
                patient_history_context=bundle.get("patient_history_context", ""),
                evidence_dossier=bundle.get("evidence_dossier"),
                role_config=bundle.get("role_config"),
                escalation_banner=_pd.escalation_banner if _pd else "",
                policy_context_note="\n".join(_pd.context_notes) if _pd else "",
                clinical_context=(
                    bundle.get("clinical_context").as_prompt_block()
                    if bundle.get("clinical_context")
                    and bundle.get("clinical_context").status != "insufficient"
                    else ""
                ),
                selected_skills=bundle.get("selected_skills", []),
                current_location=bundle.get("current_location", ""),
                task_mode=bundle.get("task_mode"),
                response_completion_guidance=bundle.get(
                    "response_completion_guidance", ""
                ),
                is_patient_scoped=bundle.get("target_patient_data_provided", False),
            )
        return self._finalize_answer_payload(
            question=question, raw_answer=raw_answer, bundle=bundle
        )

    def stream_user_question_events(
        self,
        question: str,
        chat_history: Optional[List[dict]] = None,
        user: Optional[str] = None,
        allow_generated_media: bool = True,
        extra_trace_metadata: Optional[Dict] = None,
        require_live_evidence: bool = False,
        target_patient_data: Optional[Dict] = None,
    ) -> Generator[Dict, None, None]:
        """
        Streams retrieval progress events and final answer tokens so the UI can
        show live search status followed by incremental generation.

        target_patient_data: see _prepare_answer_bundle's docstring -- passes
        straight through, unused by anything else in this method. `user`
        keeps meaning the acting/authenticated account throughout (audit,
        rate-limiting, interaction-trace), never the data source, when this
        is supplied.
        """
        yield {
            "type": "status",
            "message": "Searching live guidance, Europe PMC, and your saved context...",
        }

        # Detect if an illustration or video is needed early (fast regex, before retrieval)
        needs_illustration = (
            self._image_generator.detect_illustration_need(question)
            if allow_generated_media
            else False
        )
        needs_video = (
            self._video_generator.detect_video_request(question)
            if allow_generated_media
            else False
        )
        # Video takes priority over static illustration when both match
        if needs_video:
            needs_illustration = False

        # Evidence Ledger v2, #5: a document-analysis chat turn (see
        # stream_document_analysis_events) tags this turn's patient-fact
        # snapshot "document_extracted" instead of the default -- the extra
        # trace metadata it passes is the only signal available this early
        # (bundle["extra_trace_metadata"] isn't attached until after
        # _prepare_answer_bundle returns, further down).
        fact_source = (
            "document_extracted"
            if extra_trace_metadata and "document_analysis" in extra_trace_metadata
            else "structured_patient_record"
        )
        bundle = self._prepare_answer_bundle(
            question=question,
            user=user,
            chat_history=chat_history,
            target_patient_data=target_patient_data,
            fact_source=fact_source,
        )
        if bundle["kind"] == "final":
            payload = self._enrich_prebuilt_payload(
                question=question, payload=bundle["payload"], user=user
            )
            payload["record_updates"] = bundle.get("record_updates", [])
            if extra_trace_metadata:
                payload.setdefault("trace", {}).update(extra_trace_metadata)
            yield {
                "type": "final",
                "payload": payload,
            }
            return
        if extra_trace_metadata:
            bundle["extra_trace_metadata"] = dict(extra_trace_metadata)
        if require_live_evidence and not bundle.get("combined_sources"):
            payload = self._build_limited_payload(
                question=question,
                normalized_user=bundle.get("normalized_user"),
                personal_context=bundle.get("personal_context", []),
                retrieval_mode="required_image_evidence_unavailable",
                expanded_queries=bundle.get("expanded_queries", [question]),
            )
            if extra_trace_metadata:
                payload.setdefault("trace", {}).update(extra_trace_metadata)
                payload["image_analysis"] = extra_trace_metadata.get(
                    "image_analysis", {}
                )
                payload["image_original_question"] = extra_trace_metadata.get(
                    "image_original_question", ""
                )
            yield {"type": "final", "payload": payload}
            return

        yield {
            "type": "status",
            "message": "Composing the answer from the retrieved evidence...",
        }
        streamed_chunks: List[str] = []
        policy_decision = bundle.get("policy_decision")
        clinical_decision = bundle.get("clinical_decision")
        if clinical_decision and clinical_decision.deterministic_response:
            role_key = (
                bundle.get("role_config").role_key
                if bundle.get("role_config")
                else "patient"
            )
            deterministic_answer = clinical_decision.render_markdown(
                role_key, bundle["combined_sources"]
            )
            streamed_chunks.append(deterministic_answer)
        else:
            for chunk in self.llm.answer_question(
                question=question,
                context=bundle["full_context"],
                chat_history=bundle.get("previous_five_chat", []),
                stream=True,
                user_profile=bundle["user_profile"],
                source_briefings=bundle["combined_sources"],
                longitudinal_memory=bundle["longitudinal_memory_summary"],
                conversation_summary=bundle.get("conversation_summary", ""),
                patient_history_context=bundle.get("patient_history_context", ""),
                evidence_dossier=bundle.get("evidence_dossier"),
                role_config=bundle.get("role_config"),
                escalation_banner=policy_decision.escalation_banner
                if policy_decision
                else "",
                policy_context_note="\n".join(policy_decision.context_notes)
                if policy_decision
                else "",
                clinical_context=(
                    bundle.get("clinical_context").as_prompt_block()
                    if bundle.get("clinical_context")
                    and bundle.get("clinical_context").status != "insufficient"
                    else ""
                ),
                selected_skills=bundle.get("selected_skills", []),
                current_location=bundle.get("current_location", ""),
                task_mode=bundle.get("task_mode"),
                response_completion_guidance=bundle.get(
                    "response_completion_guidance", ""
                ),
                is_patient_scoped=bundle.get("target_patient_data_provided", False),
            ):
                streamed_chunks.append(chunk)

        raw_answer = "".join(streamed_chunks).strip()

        # Generate illustration or video after streaming (non-blocking for tokens)
        illustration = None
        video_result = None
        video_rate_limit_msg = ""

        if needs_video and user:
            from backend.user_store import UserStore as _US

            last_video_at = _US.get_last_video_generated_at(user)
            rate = self._video_generator.check_rate_limit(last_video_at)
            if not rate.allowed:
                video_rate_limit_msg = rate.message
            else:
                yield {
                    "type": "status",
                    "message": "Generating Sora-2 video (this may take a moment)...",
                }
                try:
                    video_result = self._video_generator.generate_video(
                        question=question,
                        context_answer=raw_answer[:400],
                    )
                    if video_result:
                        _US.record_video_generated(user)
                except Exception as exc:
                    print(f"Video generation failed: {exc}")

        elif needs_illustration:
            yield {"type": "status", "message": "Generating illustration..."}
            try:
                illustration = self._image_generator.generate_illustration(
                    question=question,
                    context_answer=raw_answer[:400],
                )
            except Exception as exc:
                print(f"Illustration generation failed: {exc}")

        payload = self._finalize_answer_payload(
            question=question,
            raw_answer=raw_answer,
            bundle=bundle,
        )
        if illustration:
            payload["image_url"] = illustration.image_url
            payload["image_bytes"] = (
                illustration.image_bytes
            )  # bytes when gpt-image-1, None otherwise
            payload["image_caption"] = illustration.caption
        if video_result:
            payload["video_url"] = video_result.video_url
            payload["video_caption"] = video_result.caption
        if video_rate_limit_msg:
            payload["video_rate_limit_msg"] = video_rate_limit_msg

        # Buffer model output until the post-generation context gate has
        # inspected it. A wrong-specialty answer must never flash in the UI
        # before the persisted response is corrected.
        yield {"type": "token", "delta": payload.get("answer_markdown", raw_answer)}

        # Literal transformations are complete outputs; generated health
        # follow-ups would violate the requested task and can introduce facts.
        task_mode = bundle.get("task_mode")
        if task_mode and task_mode.is_transformation:
            payload["follow_up_questions"] = []
        else:
            try:
                role_config = bundle.get("role_config")
                _norm_user = (user or "").strip().lower()
                follow_up_context = self._build_follow_up_patient_context(
                    vitals=UserStore.get_vitals(_norm_user, limit=None),
                    medications=UserStore.get_medications(_norm_user),
                    allergies=UserStore.get_allergies(_norm_user),
                    conditions=UserStore.get_conditions(_norm_user),
                    symptom_logs=UserStore.get_symptom_logs(_norm_user, limit=None),
                )
                follow_up_questions = self.llm.generate_follow_up_questions(
                    question=question,
                    answer=raw_answer,
                    chat_history=chat_history,
                    user_profile=bundle.get("user_profile", {}),
                    patient_context=follow_up_context,
                    role_key=role_config.role_key if role_config else "patient",
                    is_patient_scoped=bundle.get("target_patient_data_provided", False),
                )
                payload["follow_up_questions"] = follow_up_questions
            except Exception as exc:
                print(f"Follow-up generation failed: {exc}")
                payload["follow_up_questions"] = []

        yield {"type": "final", "payload": payload}

    def stream_image_analysis_events(
        self,
        image_bytes: bytes,
        mime_type: str,
        filename: str,
        user_note: str = "",
        chat_history: Optional[List[dict]] = None,
        user: Optional[str] = None,
    ) -> Generator[Dict, None, None]:
        """
        Streams uploaded-image analysis through the same agentic evidence
        workflow as chat, after a strict medical-image intake screen.
        """
        normalized_user = user.strip().lower() if user else None
        yield {
            "type": "status",
            "message": "Checking whether the image is clinically relevant...",
        }

        profile = UserStore.get_user_profile(normalized_user) if normalized_user else {}
        try:
            visual_result = self._image_analysis_agent.inspect(
                image_bytes=image_bytes,
                mime_type=mime_type,
                user_note=user_note,
                user_profile=profile,
                filename=filename,
            )
        except ImageAnalysisError:
            raise
        except Exception as exc:
            raise ImageAnalysisError(f"Image analysis is unavailable: {exc}") from exc

        if visual_result.get("analysis_status") != "accepted":
            yield {
                "type": "final",
                "payload": self._build_image_refusal_payload(
                    user_note=user_note,
                    normalized_user=normalized_user,
                    visual_result=visual_result,
                ),
            }
            return

        yield {
            "type": "status",
            "message": "Searching clinical guidance and research for the image findings...",
        }
        clinical_question = self._image_analysis_agent.build_clinical_question(
            visual_result=visual_result,
            user_note=user_note,
        )
        extra_trace = {
            "image_analysis": visual_result,
            "image_original_question": user_note,
            "image_uploaded_filename": filename,
            "image_analysis_model": self._image_analysis_agent.model,
        }

        for event in self.stream_user_question_events(
            question=clinical_question,
            chat_history=chat_history,
            user=user,
            allow_generated_media=False,
            extra_trace_metadata=extra_trace,
            require_live_evidence=True,
        ):
            if event.get("type") == "final":
                payload = event.get("payload", {})
                payload["image_analysis"] = visual_result
                payload["image_original_question"] = user_note
                payload.setdefault("trace", {}).update(extra_trace)
                yield {"type": "final", "payload": payload}
            else:
                yield event

    def stream_document_analysis_events(
        self,
        document_bytes: bytes,
        mime_type: str,
        filename: str,
        user_note: str = "",
        chat_history: Optional[List[dict]] = None,
        user: Optional[str] = None,
    ) -> Generator[Dict, None, None]:
        """
        Streams uploaded-document analysis through the same agentic evidence
        workflow as chat, after a strict clinical-document intake screen --
        the text-document counterpart to stream_image_analysis_events. Never
        touches document_extractor.py's personal-health-record ingestion
        path (used by the "Documents" upload panel); this is a separate,
        parallel capability for asking research/analysis questions about an
        uploaded document in chat.
        """
        normalized_user = user.strip().lower() if user else None
        yield {
            "type": "status",
            "message": "Checking whether the document is clinically relevant...",
        }

        try:
            raw_text = extract_text_from_pdf_bytes(document_bytes)
        except Exception as exc:
            raise DocumentAnalysisError(f"Could not read this PDF: {exc}") from exc

        if not raw_text.strip():
            yield {
                "type": "final",
                "payload": self._build_document_refusal_payload(
                    user_note=user_note,
                    normalized_user=normalized_user,
                    filename=filename,
                    inspect_result={
                        "analysis_status": "rejected",
                        "reason_if_rejected": (
                            "No readable text was found in this document -- it may be a scanned "
                            "image rather than a text-based PDF. Try a text-based export, or "
                            "upload it as an image instead."
                        ),
                        "uploaded_document_name": filename,
                    },
                ),
            }
            return

        profile = UserStore.get_user_profile(normalized_user) if normalized_user else {}
        try:
            inspect_result = self._document_analysis_agent.inspect(
                document_text=raw_text,
                filename=filename,
                user_note=user_note,
                user_profile=profile,
            )
        except DocumentAnalysisError:
            raise
        except Exception as exc:
            raise DocumentAnalysisError(f"Document analysis is unavailable: {exc}") from exc

        if inspect_result.get("analysis_status") != "accepted":
            yield {
                "type": "final",
                "payload": self._build_document_refusal_payload(
                    user_note=user_note,
                    normalized_user=normalized_user,
                    filename=filename,
                    inspect_result=inspect_result,
                ),
            }
            return

        yield {
            "type": "status",
            "message": "Searching clinical guidance and research for the document findings...",
        }
        clinical_question = self._document_analysis_agent.build_clinical_question(
            inspect_result=inspect_result,
            user_note=user_note,
        )
        extra_trace = {
            "document_analysis": inspect_result,
            "document_original_question": user_note,
            "document_uploaded_filename": filename,
            "document_analysis_model": self._document_analysis_agent.model,
        }

        for event in self.stream_user_question_events(
            question=clinical_question,
            chat_history=chat_history,
            user=user,
            allow_generated_media=False,
            extra_trace_metadata=extra_trace,
            require_live_evidence=True,
        ):
            if event.get("type") == "final":
                payload = event.get("payload", {})
                payload["document_analysis"] = inspect_result
                payload["document_original_question"] = user_note
                payload.setdefault("trace", {}).update(extra_trace)
                yield {"type": "final", "payload": payload}
            else:
                yield event

    def _build_document_refusal_payload(
        self,
        user_note: str,
        normalized_user: Optional[str],
        filename: str,
        inspect_result: Dict,
    ) -> Dict:
        reason = (
            inspect_result.get("reason_if_rejected")
            or "This document does not contain clear medical content that can be safely analysed."
        )
        answer = (
            "## Document Not Analysed\n\n"
            f"{reason}\n\n"
            "Please upload a clear medical document, such as a lab or imaging report, clinical "
            "letter, discharge summary, research paper, or guideline PDF. I can only analyse "
            "medical documents and will not assess unrelated files."
        )
        trace_id = f"trace-{uuid4().hex[:12]}"
        trace = {
            "trace_id": trace_id,
            "created_at": self._utc_now(),
            "question": user_note or f"Uploaded document for analysis: {filename}",
            "answer_preview": answer[:280],
            "sources": [],
            "personal_context": [],
            "retrieval_mode": "document_rejected",
            "risk_level": "routine",
            "document_analysis": inspect_result,
        }
        if normalized_user:
            UserStore.save_interaction_trace(normalized_user, trace)
        return {
            "answer_markdown": answer,
            "answer_text": answer,
            "sources": [],
            "personal_context": [],
            "longitudinal_memory": "",
            "triage_summary": {},
            "medication_alerts": [],
            "resolved_medications": [],
            "trace": trace,
            "document_analysis": inspect_result,
        }

    def _prepare_answer_bundle(
        self,
        question: str,
        user: Optional[str] = None,
        chat_history: Optional[List[dict]] = None,
        target_patient_data: Optional[Dict] = None,
        fact_source: str = "structured_patient_record",
    ) -> Dict:
        """
        target_patient_data: when supplied, sources clinical context from this
        pre-fetched bundle instead of UserStore.get_*(normalized_user) -- this
        is what lets a clinician's patient-scoped chat answer using a
        DIFFERENT patient's data than the acting/authenticated account's own
        (UserStore/SqlUserStore only ever resolve "the calling account's own
        Patient row," by design -- see backend/clinician_chat_data.py, which
        builds this bundle via a consent-checked read). `user` keeps meaning
        "the acting account" throughout regardless (audit/rate-limit/trace),
        never the data source. When omitted (every existing call site),
        behavior is unchanged.

        fact_source (Evidence Ledger v2, #5): the PatientFact.source value
        this turn's patient-fact snapshot should be persisted with -- see
        stream_document_analysis_events, which passes "document_extracted"
        instead of the default.
        """
        normalized_user = user.strip().lower() if user else None
        conversation_context = build_conversation_context(chat_history, question)
        record_updates: List[Dict] = []
        capture_profile: Optional[Dict] = None

        # Promote explicit patient-authored facts before loading context so the
        # same answer can use a newly stated medicine or allergy immediately.
        # Never write a clinician's general evidence question into a patient chart.
        if normalized_user and target_patient_data is None:
            capture_profile = UserStore.get_user_profile(normalized_user)
            capture_role = capture_profile.get("clinical_role") or capture_profile.get(
                "role", ""
            )
            if not is_clinician_role(capture_role):
                record_updates = self._capture_explicit_chat_records(
                    normalized_user, question
                )

        if target_patient_data is not None:
            # restore_user_context is intentionally skipped here: it mutates
            # a shared, per-engine embedding store keyed by username. Keying
            # it to the clinician's own account would populate/query nothing
            # useful; keying it to the patient's real username risks racing
            # that patient's own concurrent live session against the same
            # store. This is a documented v1 scope limit, not a bug -- the
            # clinician chat still gets full grounding from the structured
            # bundle below via prepare_bundle, just not the long-term
            # semantic-embedding recall layer patient sessions get.
            medications = target_patient_data.get("medications", [])
            symptom_logs = target_patient_data.get("symptom_logs", [])
            triage_summaries = target_patient_data.get("triage_summaries", [])
            allergies = target_patient_data.get("allergies", [])
            conditions = target_patient_data.get("conditions", [])
            vitals = target_patient_data.get("vitals", [])
            relationships = target_patient_data.get("clinical_relationships", [])
            care_plans = target_patient_data.get("care_plans", [])
            clinical_notes = target_patient_data.get("clinical_notes", [])
            document_summaries = target_patient_data.get("document_summaries", [])
            user_profile = target_patient_data.get("user_profile", {})
            longitudinal_memory_summary = self._compose_longitudinal_memory_summary(
                base_summary=(target_patient_data.get("longitudinal_memory_base") or "").strip(),
                symptom_summary=build_symptom_pattern_summary(symptom_logs),
                condition_summary=self._build_condition_memory_summary(conditions),
                medication_summary=self._build_medication_memory_summary(medications),
                vitals_summary=self._build_vitals_memory_summary(vitals),
                allergies_summary=self._build_allergies_memory_summary(allergies),
                relationships_summary=relationship_summary(relationships),
            )
        else:
            # Parallelize all UserStore reads + context restoration concurrently.
            # restore_user_context populates the in-memory embedding store;
            # the orchestrator's semantic search step happens after PubMed retrieval
            # (~2-3 s later), so restoration is always complete in time.
            with ThreadPoolExecutor(max_workers=12) as _pre_exec:
                _restore_f = _pre_exec.submit(self.restore_user_context, normalized_user)
                _memory_f = (
                    _pre_exec.submit(self.get_combined_longitudinal_memory, normalized_user)
                    if normalized_user
                    else None
                )
                _med_f = (
                    _pre_exec.submit(UserStore.get_medications, normalized_user)
                    if normalized_user
                    else None
                )
                _symptom_f = (
                    _pre_exec.submit(UserStore.get_symptom_logs, normalized_user, None)
                    if normalized_user
                    else None
                )
                _triage_f = (
                    _pre_exec.submit(UserStore.get_triage_summaries, normalized_user, None)
                    if normalized_user
                    else None
                )
                _allergy_f = (
                    _pre_exec.submit(UserStore.get_allergies, normalized_user)
                    if normalized_user
                    else None
                )
                _condition_f = (
                    _pre_exec.submit(UserStore.get_conditions, normalized_user)
                    if normalized_user
                    else None
                )
                _vitals_f = (
                    _pre_exec.submit(UserStore.get_vitals, normalized_user)
                    if normalized_user
                    else None
                )
                _docs_f = (
                    _pre_exec.submit(UserStore.get_document_summaries, normalized_user)
                    if normalized_user
                    else None
                )
                _relationships_f = (
                    _pre_exec.submit(
                        UserStore.get_clinical_relationships, normalized_user
                    )
                    if normalized_user
                    else None
                )
                _care_plans_f = (
                    _pre_exec.submit(CarePlanStore.list_plans, normalized_user)
                    if normalized_user
                    else None
                )
                _clinical_notes_f = (
                    _pre_exec.submit(UserStore.get_clinical_notes, normalized_user)
                    if normalized_user
                    else None
                )

                _restore_f.result()
                longitudinal_memory_summary = _memory_f.result() if _memory_f else ""
                user_profile = capture_profile or {}
                medications = _med_f.result() if _med_f else []
                symptom_logs = _symptom_f.result() if _symptom_f else []
                triage_summaries = _triage_f.result() if _triage_f else []
                allergies = _allergy_f.result() if _allergy_f else []
                conditions = _condition_f.result() if _condition_f else []
                vitals = _vitals_f.result() if _vitals_f else []
                document_summaries = _docs_f.result() if _docs_f else []
                relationships = (
                    _relationships_f.result() if _relationships_f else []
                )
                care_plans = _care_plans_f.result() if _care_plans_f else []
                clinical_notes = _clinical_notes_f.result() if _clinical_notes_f else []

        relationships = merge_relationships(
            relationships,
            derive_relationships(
                medications=medications,
                allergies=allergies,
                conditions=conditions,
                symptom_logs=symptom_logs,
                vitals=vitals,
                triage_summaries=triage_summaries,
                care_plans=care_plans,
                clinical_notes=clinical_notes,
                safety_reviews=build_safety_reviews(
                    vitals=vitals,
                    symptoms=symptom_logs,
                    medications=medications,
                    allergies=allergies,
                    conditions=conditions,
                    triage_summaries=triage_summaries,
                    document_summaries=document_summaries,
                    clinical_relationships=relationships,
                    longitudinal_memory=longitudinal_memory_summary,
                    saved_states=(
                        UserStore.get_safety_review_states(normalized_user)
                        if normalized_user and target_patient_data is None
                        else {}
                    ),
                ),
            ),
        )

        # Build a fast relevance graph from prior records (< 50 ms, no LLM).
        from backend.context_graph import build_context_graph

        context_graph = build_context_graph(
            question=question,
            conditions=conditions,
            medications=medications,
            symptom_logs=symptom_logs,
            vitals=vitals,
            allergies=allergies,
            triage_summaries=triage_summaries,
            relationships=relationships,
            longitudinal_memory=longitudinal_memory_summary,
        )

        bundle = self._orchestrator.prepare_bundle(
            question=question,
            user=normalized_user,
            user_profile=user_profile,
            longitudinal_memory_summary=longitudinal_memory_summary,
            medications=medications,
            triage_summaries=triage_summaries,
            allergies=allergies,
            conditions=conditions,
            vitals=vitals,
            document_summaries=document_summaries,
            context_graph=context_graph,
            chat_history=chat_history,
            previous_five_chat=conversation_context.previous_five,
            conversation_summary=conversation_context.full_summary,
            patient_statement_summary=conversation_context.patient_statement_summary,
        )
        bundle["record_updates"] = record_updates
        bundle["previous_five_chat"] = conversation_context.previous_five
        bundle["conversation_summary"] = conversation_context.full_summary
        bundle["patient_statement_summary"] = conversation_context.patient_statement_summary
        # Sole signal the follow-up-question generator (stream_user_question_events)
        # has for distinguishing "clinician asking about a specific patient's chart"
        # from "clinician asking a general, patient-agnostic evidence question" --
        # see generate_follow_up_questions' is_patient_scoped parameter.
        bundle["target_patient_data_provided"] = target_patient_data is not None
        if bundle.get("kind") == "answer":
            medication_check = self._build_medication_check(
                question=question,
                intent=bundle.get("intent"),
                medications=medications,
                allergies=allergies,
                question_medications=bundle.get("question_medications", []),
            )
            bundle["medication_check"] = medication_check
            bundle["symptom_logs"] = symptom_logs
            bundle["medications"] = medications
            bundle["conditions"] = conditions
            if medication_check.get("alerts") or medication_check.get("allergy_conflicts"):
                bundle["full_context"] = self._append_medication_context(
                    bundle["full_context"],
                    medication_check["alerts"],
                    medication_check.get("allergy_conflicts", []),
                )
            # Evidence Ledger Phase 1: persist source identity + passage-level
            # detail, mutating combined_sources in place with source_version/
            # retrieved_at/exact_passage. Never allowed to block the answer
            # already being generated from these same sources -- guarded here
            # too, on top of persist_evidence_for_bundle's own internal
            # per-source guard, in case the DB itself is unreachable.
            try:
                from backend.evidence_ledger import persist_evidence_for_bundle

                persist_evidence_for_bundle(
                    bundle.get("combined_sources", []), bundle.get("evidence_dossier")
                )
            except Exception as exc:
                print(f"[EvidenceLedger] persist_evidence_for_bundle failed: {exc}")
            # Evidence Ledger v2, #4: detect same-intervention cross-source
            # contradictions now (combined_sources' source_artifact_id fields
            # were just populated above), so a positive result can still
            # reach the answer prompt via full_context below. Persistence
            # happens later, in _finalize_answer_payload, once the real
            # trace_id exists -- findings are carried on the bundle until
            # then. Never allowed to block the answer.
            try:
                from backend.contradiction_detector import detect_contradictions

                contradictions = detect_contradictions(self.llm, bundle.get("evidence_dossier"))
                bundle["contradictions"] = contradictions
                if contradictions:
                    bundle["full_context"] = self._append_contradiction_context(
                        bundle["full_context"], contradictions
                    )
            except Exception as exc:
                print(f"[ContradictionDetector] detect_contradictions failed: {exc}")
            # Evidence Ledger Phase 3: persist a snapshot of the patient's
            # structured facts (conditions/medications/allergies/vitals/
            # symptoms) so a future AnswerClaim can cite exactly which
            # patient fact was used. Same never-blocks-the-answer guarantee.
            try:
                from backend.patient_fact_ledger import persist_patient_facts_for_bundle

                persist_patient_facts_for_bundle(
                    user_profile.get("patient_record_id"),
                    medications=medications,
                    conditions=conditions,
                    allergies=allergies,
                    vitals=vitals,
                    symptom_logs=symptom_logs,
                    source=fact_source,
                )
            except Exception as exc:
                print(f"[PatientFactLedger] persist_patient_facts_for_bundle failed: {exc}")
        return bundle

    def _build_moderation_payload(
        self,
        question: str,
        normalized_user: Optional[str],
        safe_msg: str,
        category: str,
        details: Dict,
    ) -> Dict:
        trace_id = f"trace-{uuid4().hex[:12]}"
        trace = {
            "trace_id": trace_id,
            "created_at": self._utc_now(),
            "question": question,
            "answer_preview": safe_msg[:280],
            "sources": [],
            "retrieval_mode": "moderation_block",
            "moderation_category": category,
            "moderation_details": details,
        }
        if normalized_user:
            UserStore.save_interaction_trace(normalized_user, trace)
        return {
            "answer_markdown": safe_msg,
            "answer_text": safe_msg,
            "sources": [],
            "personal_context": [],
            "trace": trace,
        }

    def _build_image_refusal_payload(
        self,
        user_note: str,
        normalized_user: Optional[str],
        visual_result: Dict,
    ) -> Dict:
        reason = (
            visual_result.get("reason_if_rejected")
            or "This image does not contain a clear medical concern that can be safely analysed."
        )
        answer = (
            "## Image Not Analysed\n\n"
            f"{reason}\n\n"
            "Please upload a clear image of the health concern, such as a rash, wound, swelling, "
            "colour change, medication label, test strip, or other medical finding. I can only "
            "analyse medical images and will not assess unrelated pictures."
        )
        trace_id = f"trace-{uuid4().hex[:12]}"
        trace = {
            "trace_id": trace_id,
            "created_at": self._utc_now(),
            "question": user_note or "Uploaded image for analysis",
            "answer_preview": answer[:280],
            "sources": [],
            "personal_context": [],
            "retrieval_mode": "image_rejected",
            "risk_level": "routine",
            "image_analysis": visual_result,
        }
        if normalized_user:
            UserStore.save_interaction_trace(normalized_user, trace)
        return {
            "answer_markdown": answer,
            "answer_text": answer,
            "sources": [],
            "personal_context": [],
            "longitudinal_memory": "",
            "triage_summary": {},
            "medication_alerts": [],
            "resolved_medications": [],
            "trace": trace,
            "image_analysis": visual_result,
        }

    def _build_limited_payload(
        self,
        question: str,
        normalized_user: Optional[str],
        personal_context: List[Dict],
        retrieval_mode: str,
        expanded_queries: List[str],
    ) -> Dict:
        limited_answer = self._build_limited_evidence_response(personal_context)
        trace_id = f"trace-{uuid4().hex[:12]}"
        trace = {
            "trace_id": trace_id,
            "created_at": self._utc_now(),
            "question": question,
            "answer_preview": limited_answer[:280],
            "sources": [],
            "personal_context": personal_context,
            "retrieval_mode": retrieval_mode,
            "expanded_queries": expanded_queries,
            "model": self.llm.model,
        }
        if normalized_user:
            UserStore.save_interaction_trace(normalized_user, trace)
        return {
            "answer_markdown": limited_answer,
            "answer_text": limited_answer,
            "sources": [],
            "personal_context": personal_context,
            "trace": trace,
        }

    def _finalize_answer_payload(
        self,
        question: str,
        raw_answer: str,
        bundle: Dict,
    ) -> Dict:
        clinical_context: Optional[ClinicalContextDecision] = bundle.get(
            "clinical_context"
        )
        task_mode = bundle.get("task_mode")
        context_validation = validate_generated_answer(raw_answer, clinical_context)
        if not context_validation["valid"] and clinical_context:
            raw_answer = clinical_context.correction_message()

        # Citation existence is a pre-display requirement, not merely an audit.
        # Unknown model-generated markers are removed before links are rendered.
        source_ids = [
            source.get("source_id", "") for source in bundle.get("combined_sources", [])
        ]
        raw_answer = remove_unknown_citations(raw_answer, source_ids)

        # Claim-source alignment gate. Runs before any deterministic banners/
        # disclaimers/safety-net are appended below, so a rewrite here can only
        # ever touch the model's own text -- it never risks mangling the
        # scaffolding that's added fresh afterward regardless of what happened
        # here. Like the wrong-specialty context gate above, this must run
        # before the answer is delivered, not after -- there is no streaming
        # path where the client sees tokens before this function returns (see
        # stream_user_question_events: the full answer is buffered and this
        # function's result is what gets yielded, once).
        combined_sources = bundle.get("combined_sources", [])
        claim_alignment: List[Dict] = []
        claim_correction_applied = False
        # Hoisted out of the `if combined_sources:` block below (instead of only
        # existing inside it) so it's still in scope when Evidence Ledger Phase 4
        # persists claim classifications after trace_id is computed further down.
        uncited_supported_claims: List[Dict] = []
        # Fail-closed (#8): set when verification couldn't be completed even
        # after a retry, in which case raw_answer below is replaced with
        # SAFE_VERIFICATION_FALLBACK_MESSAGE rather than shipping unverified
        # text. Read further down by the Evidence Ledger Phase 4 persistence
        # call to record an "unsupported_blocked" audit row.
        answer_blocked = False
        if combined_sources:
            claim_alignment, alignment_ok = _retry_once(
                self.llm.check_claim_source_alignment,
                answer_markdown=raw_answer,
                source_briefings=combined_sources,
            )
            if not alignment_ok:
                claim_alignment = []
                answer_blocked = True
                raw_answer = SAFE_VERIFICATION_FALLBACK_MESSAGE

            if not answer_blocked:
                unsupported_claims = [
                    c
                    for c in claim_alignment
                    if c.get("status") == "general_knowledge" and c.get("requires_evidence")
                ]
                # A claim the check confirmed IS supported by a specific source, but
                # whose [S#] marker never made it into the generated text, is the
                # other half of the same problem: the model drew on the evidence
                # correctly but didn't attribute it. Only flagged when NONE of the
                # claim's source_ids appear anywhere in the text yet, so an already-
                # cited source never gets a redundant second marker inserted.
                uncited_supported_claims = [
                    c
                    for c in claim_alignment
                    if c.get("status") == "supported"
                    and c.get("source_ids")
                    and not any(f"[{sid}]" in raw_answer for sid in c["source_ids"])
                ]
                if unsupported_claims or uncited_supported_claims:
                    # apply_claim_corrections never raises (it catches its own
                    # OpenAI call and returns the original text on failure),
                    # so "failed" here means a no-op return, not an exception
                    # -- retry once on that condition instead of on exceptions.
                    rewritten = ""
                    for _attempt in range(2):
                        rewritten = self.llm.apply_claim_corrections(
                            answer_markdown=raw_answer,
                            unsupported_claims=unsupported_claims,
                            source_briefings=combined_sources,
                            uncited_supported_claims=uncited_supported_claims,
                        )
                        if rewritten and rewritten.strip() and rewritten != raw_answer:
                            break
                    if rewritten and rewritten.strip() and rewritten != raw_answer:
                        raw_answer = rewritten
                        claim_correction_applied = True
                    elif unsupported_claims:
                        # Both attempts were a no-op while genuinely
                        # unsupported claims remain -- don't ship them.
                        answer_blocked = True
                        raw_answer = SAFE_VERIFICATION_FALLBACK_MESSAGE

        role_config = bundle.get("role_config")
        raw_answer = self._append_clinical_evidence_trail(
            raw_answer,
            bundle.get("combined_sources", []),
            role_config.role_key if role_config else "patient",
        )

        language_valid, language_violations = validate_user_facing_language(raw_answer)
        if not language_valid:
            raw_answer = remove_internal_language(raw_answer)
            language_valid, remaining_violations = validate_user_facing_language(
                raw_answer
            )
            if remaining_violations or not raw_answer:
                raw_answer = (
                    "I need a little more information to answer safely. Please provide the specific "
                    "symptom, medicine, or report wording you want help with."
                )

        answer_markdown = self._link_citations(raw_answer, bundle["combined_sources"])

        # Prepend escalation banner to answer if policy triggered one
        policy_decision = bundle.get("policy_decision")
        if policy_decision and policy_decision.escalation_banner:
            answer_markdown = policy_decision.escalation_banner + answer_markdown

        # Append disclaimer only if the LLM hasn't already included equivalent text
        if policy_decision and policy_decision.disclaimer:
            _disc_marker = f"{PRODUCT_NAME} provides evidence-based"
            _clinical_marker = "This summary is for clinical decision-support"
            _edu_marker = "This information is for educational purposes"
            already_present = any(
                m in answer_markdown
                for m in (_disc_marker, _clinical_marker, _edu_marker)
            )
            if not already_present:
                answer_markdown = answer_markdown + policy_decision.disclaimer

        # Append vulnerability notice near top if applicable
        if policy_decision and policy_decision.vulnerability_notice:
            answer_markdown = policy_decision.vulnerability_notice + answer_markdown

        intent = bundle.get("intent")
        risk_level = intent.risk_level if intent else "routine"
        combined_sources = bundle.get("combined_sources", [])
        medication_check = bundle.get("medication_check", {})
        clinical_decision = bundle.get("clinical_decision")
        evidence_quality_report = bundle.get("evidence_quality_report", {})

        # Build triage summary first so safety netting can use its LLM-derived triggers
        if task_mode and task_mode.is_transformation:
            triage_summary = build_default_triage(intent, policy_decision)
        else:
            triage_summary = self._build_triage_summary(
                question=question,
                answer_markdown=answer_markdown,
                intent=intent,
                policy_decision=policy_decision,
                clinical_decision=clinical_decision,
            )

        # Append structured safety netting block -- triggers come from triage_summary, no hardcoding
        safety_net = self._build_safety_net_block(
            risk_level, triage_summary, role_config
        )
        clinician_escalation_present = any(
            marker in answer_markdown
            for marker in (
                "## Escalate Now If",
                "## Escalate Immediately If",
                "## Escalate Or Refer If",
                "## Get Urgent Help If",
            )
        )
        if (
            safety_net
            and "**Return immediately if**" not in answer_markdown
            and not clinician_escalation_present
        ):
            answer_markdown = answer_markdown + safety_net

        trace_id = f"trace-{uuid4().hex[:12]}"

        # Evidence Ledger Phase 4: persist the claim-source-alignment check
        # already run above as first-class AnswerClaim rows, with best-effort
        # links to the EvidenceClaim/PatientFact rows Phases 2/3 persisted.
        # Never allowed to block the answer already finalized above.
        try:
            from backend.answer_claim_ledger import persist_answer_claims_for_bundle

            persist_answer_claims_for_bundle(
                trace_id,
                bundle.get("user_profile", {}).get("patient_record_id"),
                claim_alignment=claim_alignment,
                uncited_supported_claims=uncited_supported_claims,
                claim_correction_applied=claim_correction_applied,
                answer_blocked=answer_blocked,
            )
        except Exception as exc:
            print(f"[AnswerClaimLedger] persist_answer_claims_for_bundle failed: {exc}")

        # Evidence Ledger v2, #4: persist the contradiction pairs detected
        # earlier in _prepare_answer_bundle, now that trace_id exists so
        # they correlate with this answer's AnswerClaim rows for the #11
        # lineage view. Never allowed to block the answer.
        try:
            from backend.contradiction_detector import persist_contradictions_for_bundle

            persist_contradictions_for_bundle(
                trace_id, bundle.get("contradictions", []), combined_sources
            )
        except Exception as exc:
            print(f"[ContradictionDetector] persist_contradictions_for_bundle failed: {exc}")

        from backend.evidence_ranker import EvidenceRanker

        tiers_present = EvidenceRanker.get_tiers_present(combined_sources)

        trace = {
            "trace_id": trace_id,
            "created_at": self._utc_now(),
            "question": question,
            "answer_preview": raw_answer[:280],
            "sources": combined_sources,
            "personal_context": bundle["personal_context"],
            "retrieval_mode": bundle["retrieval_mode"],
            "expanded_queries": bundle["expanded_queries"],
            "memory_match_count": len(bundle["matches"]),
            "model": self.llm.model,
            # Clinical governance fields
            "role_key": role_config.role_key if role_config else "patient",
            "intent_category": intent.intent_category if intent else "",
            "risk_level": intent.risk_level if intent else "routine",
            "escalation_triggered": bool(
                policy_decision and policy_decision.action != "allow"
            ),
            "crisis_detected": intent.crisis_detected if intent else False,
            "evidence_tiers_present": tiers_present,
            "pathway_used": intent.pathway_hint if intent else "",
            "vulnerable_flags": intent.vulnerable_flags if intent else [],
            "policy_gates_applied": policy_decision.gates_as_dicts()
            if policy_decision
            else [],
            "medication_alert_count": len(medication_check.get("alerts", [])),
            "decision_logic_version": clinical_decision.logic_version
            if clinical_decision
            else "",
            "pathway_decision": clinical_decision.as_dict()
            if clinical_decision
            else {},
            "rule_hits": [item.as_dict() for item in clinical_decision.triggered_rules]
            if clinical_decision
            else [],
            "guideline_references": [
                item.as_dict() for item in clinical_decision.guideline_references
            ]
            if clinical_decision
            else [],
            "evidence_quality": evidence_quality_report,
            "claim_alignment": claim_alignment,
            "claim_correction_applied": claim_correction_applied,
            "clinical_context": clinical_context.as_dict() if clinical_context else {},
            "clinical_context_validation": context_validation,
            "user_facing_validation": {
                "valid": language_valid,
                "violations": language_violations,
            },
            "task_mode": task_mode.mode if task_mode else "clinical_answer",
            "presentation_audience": (
                task_mode.presentation_audience
                if task_mode
                else (role_config.role_key if role_config else "patient")
            ),
            "task_mode_reason": task_mode.reason if task_mode else "",
        }
        extra_trace_metadata = bundle.get("extra_trace_metadata")
        if isinstance(extra_trace_metadata, dict):
            trace.update(extra_trace_metadata)
        if bundle["normalized_user"]:
            UserStore.save_interaction_trace(bundle["normalized_user"], trace)
            UserStore.save_triage_summary(
                bundle["normalized_user"],
                {
                    **triage_summary,
                    "question": question,
                    "trace_id": trace_id,
                },
            )
        return {
            "answer_markdown": answer_markdown,
            "answer_text": raw_answer,
            "sources": combined_sources,
            "personal_context": bundle["personal_context"],
            "longitudinal_memory": bundle["longitudinal_memory_summary"],
            "triage_summary": triage_summary,
            "medication_alerts": medication_check.get("alerts", []),
            "resolved_medications": self._summarize_resolved_medications(
                medication_check.get("resolved_medications", [])
            ),
            "evidence_quality": evidence_quality_report,
            "clinical_context": clinical_context.as_dict() if clinical_context else {},
            "record_updates": bundle.get("record_updates", []),
            "trace": trace,
        }

    def refresh_longitudinal_memory_from_turn(
        self,
        user: Optional[str],
        user_message: str,
        personal_context: Optional[List[Dict]] = None,
    ) -> str:
        normalized_user = user.strip().lower() if user else None
        if not normalized_user:
            return ""

        new_information = self._build_longitudinal_memory_turn_input(
            user_message=user_message,
            personal_context=personal_context or [],
        )
        self._refresh_longitudinal_memory(
            user=normalized_user,
            new_information=new_information,
            source_label="conversation",
        )
        self.restore_user_context(normalized_user)
        return self.get_combined_longitudinal_memory(normalized_user)

    def refresh_longitudinal_memory_from_documents(
        self,
        user: Optional[str],
        indexed_documents: List[Dict],
    ) -> str:
        normalized_user = user.strip().lower() if user else None
        if not normalized_user or not indexed_documents:
            return ""

        new_information = "\n\n".join(
            f"{item.get('file', 'Document')}:\n{item.get('summary', '').strip()}"
            for item in indexed_documents
            if item.get("summary", "").strip()
        )
        self._refresh_longitudinal_memory(
            user=normalized_user,
            new_information=new_information,
            source_label="uploaded documents",
        )
        self.restore_user_context(normalized_user)
        return self.get_combined_longitudinal_memory(normalized_user)

    def get_combined_longitudinal_memory(self, user: Optional[str]) -> str:
        normalized_user = user.strip().lower() if user else None
        if not normalized_user:
            return ""

        stored_memory = UserStore.get_longitudinal_memory(normalized_user)
        base_summary = (stored_memory.get("summary") or "").strip()
        symptom_summary = build_symptom_pattern_summary(
            UserStore.get_symptom_logs(normalized_user, limit=None)
        )
        condition_summary = self._build_condition_memory_summary(
            UserStore.get_conditions(normalized_user)
        )
        medication_summary = self._build_medication_memory_summary(
            UserStore.get_medications(normalized_user)
        )
        vitals_summary = self._build_vitals_memory_summary(
            UserStore.get_vitals(normalized_user, limit=None)
        )
        allergies_summary = self._build_allergies_memory_summary(
            UserStore.get_allergies(normalized_user)
        )
        relationships = merge_relationships(
            UserStore.get_clinical_relationships(normalized_user),
            derive_relationships(
                medications=UserStore.get_medications(normalized_user),
                allergies=UserStore.get_allergies(normalized_user),
                conditions=UserStore.get_conditions(normalized_user),
                symptom_logs=UserStore.get_symptom_logs(normalized_user, limit=None),
                vitals=UserStore.get_vitals(normalized_user, limit=None),
                triage_summaries=UserStore.get_triage_summaries(
                    normalized_user, limit=None
                ),
                care_plans=CarePlanStore.list_plans(normalized_user),
                clinical_notes=UserStore.get_clinical_notes(normalized_user),
                safety_reviews=build_safety_reviews(
                    vitals=UserStore.get_vitals(normalized_user, limit=None),
                    symptoms=UserStore.get_symptom_logs(normalized_user, limit=None),
                    medications=UserStore.get_medications(normalized_user),
                    allergies=UserStore.get_allergies(normalized_user),
                    conditions=UserStore.get_conditions(normalized_user),
                    triage_summaries=UserStore.get_triage_summaries(
                        normalized_user, limit=None
                    ),
                    document_summaries=UserStore.get_document_summaries(
                        normalized_user
                    ),
                    clinical_relationships=UserStore.get_clinical_relationships(
                        normalized_user
                    ),
                    longitudinal_memory=base_summary,
                    saved_states=UserStore.get_safety_review_states(normalized_user),
                ),
            ),
        )
        return self._compose_longitudinal_memory_summary(
            base_summary=base_summary,
            symptom_summary=symptom_summary,
            condition_summary=condition_summary,
            medication_summary=medication_summary,
            vitals_summary=vitals_summary,
            allergies_summary=allergies_summary,
            relationships_summary=relationship_summary(relationships),
        )

    def build_summary_pdf_for_user(self, user: Optional[str]) -> bytes:
        normalized_user = user.strip().lower() if user else None
        if not normalized_user:
            return b""

        user_profile = UserStore.get_user_profile(normalized_user)
        from backend.role_router import RoleRouter

        role_key = (
            RoleRouter()
            .resolve(user_profile.get("clinical_role") or user_profile.get("role", ""))
            .role_key
        )

        from backend.gp_summary import build_summary_pdf

        return build_summary_pdf(
            user_profile=user_profile,
            symptom_logs=UserStore.get_symptom_logs(normalized_user, limit=None),
            medications=UserStore.get_medications(normalized_user),
            uploads=UserStore.get_uploads(normalized_user),
            longitudinal_memory=self.get_combined_longitudinal_memory(normalized_user),
            role_key=role_key,
            triage_summaries=UserStore.get_triage_summaries(
                normalized_user, limit=None
            ),
            recent_chats=UserStore.get_chat_history(normalized_user),
            allergies=UserStore.get_allergies(normalized_user),
            conditions=UserStore.get_conditions(normalized_user),
            vitals=UserStore.get_vitals(normalized_user, limit=None),
        )

    # Keep old name as a shim
    def build_gp_summary_pdf_for_user(self, user: Optional[str]) -> bytes:
        return self.build_summary_pdf_for_user(user)

    def _refresh_longitudinal_memory(
        self,
        user: str,
        new_information: str,
        source_label: str,
    ) -> str:
        cleaned_information = (new_information or "").strip()
        if not cleaned_information:
            return ""

        existing_memory = UserStore.get_longitudinal_memory(user)
        existing_summary = (existing_memory.get("summary") or "").strip()
        updated_summary = self.llm.refresh_longitudinal_memory(
            existing_memory=existing_summary,
            new_information=cleaned_information,
            user_profile=UserStore.get_user_profile(user),
            source_label=source_label,
        )
        normalized_summary = self._normalize_longitudinal_memory_summary(
            updated_summary
        )
        UserStore.save_longitudinal_memory(
            user,
            normalized_summary,
            source=source_label,
            metadata={
                "input_length": len(cleaned_information),
                "summary_length": len(normalized_summary),
            },
        )
        return normalized_summary

    def _default_document_paths(self, user: Optional[str]) -> List[Path]:
        if user:
            upload_dir = UserStore.get_upload_dir(user)
            return sorted(upload_dir.glob("*.pdf"))
        if not self.embedding_dir.exists():
            return []
        return sorted(self.embedding_dir.glob("*.pdf"))

    def _retrieve_pubmed_for_queries(
        self, queries: List[str], user: Optional[str]
    ) -> None:
        pending_entries = []
        article_batches: List[Tuple[str, List[Dict[str, str]]]] = []

        with ThreadPoolExecutor(max_workers=min(3, max(1, len(queries)))) as executor:
            query_futures = {
                executor.submit(self.pubmed.search_article_records, query, 2): query
                for query in queries
            }
            for future, query in query_futures.items():
                try:
                    article_batches.append((query, future.result()))
                except Exception as exc:
                    print(f"PubMed search failed for '{query}': {exc}")

        article_records: List[Tuple[str, Dict[str, str]]] = []
        for query, records in article_batches:
            for record in records:
                article_records.append((query, record))

        with ThreadPoolExecutor(
            max_workers=min(6, max(1, len(article_records)))
        ) as executor:
            section_futures = {
                executor.submit(self.pubmed.fetch_article_sections, record["pmcid"]): (
                    query,
                    record,
                )
                for query, record in article_records
            }
            for future, payload in section_futures.items():
                query, record = payload
                try:
                    sections = future.result()
                except Exception as exc:
                    print(
                        f"PubMed full-text fetch failed for {record.get('pmcid', '')}: {exc}"
                    )
                    sections = {}

                best_section_name, best_section_text = self._select_best_pubmed_section(
                    sections
                )
                if best_section_text:
                    entry_key = (
                        f"{user or 'global'}:pmc:{record['pmcid']}:{best_section_name}"
                    )
                    pending_entries.append(
                        {
                            "text": best_section_text,
                            "metadata": {
                                "type": "pubmed",
                                "source_type": "pubmed_literature",
                                "pmcid": record["pmcid"],
                                "section": best_section_name,
                                "title": record.get("title", "Untitled article"),
                                "journal": record.get("journal", ""),
                                "year": record.get("year", ""),
                                "authors": record.get("authors", ""),
                                "url": record.get("url", ""),
                                "query": query,
                                "entry_key": entry_key,
                            },
                            "user": user,
                            "entry_key": entry_key,
                        }
                    )

                abstract_text = record.get("abstract", "")
                if abstract_text:
                    entry_key = f"{user or 'global'}:pmc:{record['pmcid']}:abstract"
                    pending_entries.append(
                        {
                            "text": abstract_text,
                            "metadata": {
                                "type": "pubmed",
                                "source_type": "pubmed_literature",
                                "pmcid": record["pmcid"],
                                "section": "abstract",
                                "title": record.get("title", "Untitled article"),
                                "journal": record.get("journal", ""),
                                "year": record.get("year", ""),
                                "authors": record.get("authors", ""),
                                "url": record.get("url", ""),
                                "query": query,
                                "entry_key": entry_key,
                            },
                            "user": user,
                            "entry_key": entry_key,
                        }
                    )

        self.memory.add_entries(pending_entries)

    def _split_matches(
        self, matches: List[Tuple[Dict, float]]
    ) -> Tuple[List[Dict], List[Tuple[Dict, float]]]:
        personal_context = []
        pubmed_matches = []

        for entry, score in matches:
            metadata = entry.get("metadata", {})
            if metadata.get("type") == "user_summary":
                personal_context.append(
                    {
                        "title": metadata.get(
                            "title", metadata.get("source", "Uploaded document")
                        ),
                        "source": metadata.get("source", ""),
                        "snippet": build_excerpt(entry.get("text", "")),
                        "score": round(score, 3),
                    }
                )
            elif metadata.get("type") == "pubmed":
                pubmed_matches.append((entry, score))

        return personal_context[:2], pubmed_matches[:4]

    def _build_source_briefings(self, matches: List[Tuple[Dict, float]]) -> List[Dict]:
        sources = []
        seen = set()

        for entry, score in matches:
            metadata = entry.get("metadata", {})
            key = (metadata.get("pmcid"), metadata.get("section"))
            if key in seen:
                continue
            seen.add(key)
            source_id = f"S{len(sources) + 1}"
            sources.append(
                {
                    "source_id": source_id,
                    "pmcid": metadata.get("pmcid", ""),
                    "title": metadata.get("title", "Untitled article"),
                    "journal": metadata.get("journal", ""),
                    "year": metadata.get("year", ""),
                    "authors": metadata.get("authors", ""),
                    "section": metadata.get("section", "retrieved text")
                    .replace("_", " ")
                    .title(),
                    "url": metadata.get("url", ""),
                    "query": metadata.get("query", ""),
                    "similarity": round(score, 3),
                    "snippet": build_excerpt(entry.get("text", "")),
                    "detail_snippet": build_excerpt(
                        entry.get("text", ""), max_chars=800
                    ),
                    "source_type": metadata.get("source_type", "pubmed_literature"),
                    "provider": "Europe PMC / PubMed Central",
                }
            )

        return sources

    @staticmethod
    def _combine_sources(
        pubmed_sources: List[Dict], official_sources: List[Dict]
    ) -> List[Dict]:
        combined = []
        seen = set()

        for source in [*official_sources, *pubmed_sources]:
            key = source.get("url") or f"{source.get('title')}::{source.get('section')}"
            if key in seen:
                continue
            seen.add(key)
            combined.append(dict(source))

        for index, source in enumerate(combined, start=1):
            source["source_id"] = f"S{index}"
        return combined

    def _rank_sources(
        self, question: str, sources: List[Dict], top_k: int = 6
    ) -> List[Dict]:
        if not sources:
            return []

        query_vector = self.memory._embed_text(question)
        source_texts = [
            (
                " ".join(
                    part
                    for part in (
                        source.get("title", ""),
                        source.get("section", ""),
                        source.get("detail_snippet", ""),
                        source.get("snippet", ""),
                        source.get("query", ""),
                    )
                    if part
                )
                or source.get("title", "Retrieved source")
            )
            for source in sources
        ]
        source_vectors = self.memory._embed_texts(source_texts)
        scored_sources = []
        for source, source_vector in zip(sources, source_vectors):
            score = float(np.dot(query_vector, source_vector))
            payload = dict(source)
            payload["relevance"] = round(score, 3)
            scored_sources.append(payload)

        scored_sources.sort(key=lambda item: item.get("relevance", 0.0), reverse=True)
        ranked = scored_sources[:top_k]
        for index, source in enumerate(ranked, start=1):
            source["source_id"] = f"S{index}"
        return ranked

    def _build_search_queries(self, question: str) -> List[str]:
        queries = [question]
        try:
            queries.extend(self.query_expander.expand(question))
        except Exception as exc:
            print(f"Query expansion fallback: {exc}")
        return list(dict.fromkeys(query for query in queries if query))[:3]

    @staticmethod
    def _build_longitudinal_memory_turn_input(
        user_message: str,
        personal_context: List[Dict],
    ) -> str:
        parts = []
        cleaned_message = (user_message or "").strip()
        if cleaned_message:
            parts.append(f"Latest user message:\n{cleaned_message}")

        if personal_context:
            context_lines = [
                f"- {item.get('title', item.get('source', 'Context'))}: {item.get('snippet', '').strip()}"
                for item in personal_context
                if item.get("snippet", "").strip()
            ]
            if context_lines:
                parts.append(
                    "Relevant patient-specific context already on file:\n"
                    + "\n".join(context_lines)
                )

        return "\n\n".join(parts)

    @staticmethod
    def _select_best_pubmed_section(sections: Dict[str, str]) -> Tuple[str, str]:
        for key in ("discussion", "conclusion", "introduction"):
            text = (sections.get(key) or "").strip()
            if text:
                return key, text
        return "", ""

    @staticmethod
    def _normalize_longitudinal_memory_summary(summary: str) -> str:
        cleaned = " ".join((summary or "").split()).strip()
        if cleaned.lower() == "no durable patient-specific memory recorded yet.":
            return ""
        return summary.strip()

    @staticmethod
    def _compose_longitudinal_memory_summary(
        base_summary: str,
        symptom_summary: str,
        condition_summary: str,
        medication_summary: str,
        vitals_summary: str = "",
        allergies_summary: str = "",
        relationships_summary: str = "",
    ) -> str:
        parts = []
        cleaned_base = (base_summary or "").strip()
        if cleaned_base:
            parts.append(cleaned_base)
        if condition_summary:
            parts.append(condition_summary)
        if medication_summary:
            parts.append(medication_summary)
        if allergies_summary:
            parts.append(allergies_summary)
        if relationships_summary:
            parts.append(relationships_summary)
        if vitals_summary:
            parts.append(vitals_summary)
        if symptom_summary:
            parts.append(symptom_summary)
        return "\n\n".join(parts).strip()

    @staticmethod
    def _build_condition_memory_summary(conditions: List[Dict]) -> str:
        condition_lines = []
        for condition in conditions:
            name = (condition.get("name") or "").strip()
            if not name:
                continue
            status = (condition.get("status") or "").strip()
            line = name
            if status and status != "unknown":
                line += f" ({status})"
            condition_lines.append(line)
        if not condition_lines:
            return ""
        return "Conditions and history:\n" + "\n".join(
            f"- {item}" for item in condition_lines[:10]
        )

    @staticmethod
    def _build_medication_memory_summary(medications: List[Dict]) -> str:
        medication_lines = []
        for medication in medications:
            name = (medication.get("name") or "").strip()
            if not name:
                continue
            line = name
            extras = [
                part
                for part in (
                    medication.get("dose", "").strip(),
                    medication.get("schedule", "").strip(),
                )
                if part
            ]
            if extras:
                line += " - " + ", ".join(extras)
            medication_lines.append(line)
        if not medication_lines:
            return ""
        return "Current medication list:\n" + "\n".join(
            f"- {item}" for item in medication_lines[:8]
        )

    @staticmethod
    def _build_vitals_memory_summary(vitals: List[Dict]) -> str:
        """
        Builds a concise summary of the most recent reading for each vital/lab type.
        Groups by type and keeps only the latest date to avoid noise.
        """
        latest: Dict[str, Dict] = {}
        for entry in vitals:
            vtype = (entry.get("type") or "").strip().lower()
            if not vtype:
                continue
            recorded = (
                entry.get("recorded_on") or entry.get("created_at") or ""
            ).strip()
            existing = latest.get(vtype)
            if existing is None:
                latest[vtype] = entry
            else:
                existing_date = (
                    existing.get("recorded_on") or existing.get("created_at") or ""
                ).strip()
                if recorded > existing_date:
                    latest[vtype] = entry

        if not latest:
            return ""

        lines = []
        for vtype in sorted(latest):
            entry = latest[vtype]
            if not (entry.get("value") or "").strip():
                continue
            lines.append(render_vital_for_prompt(entry))

        if not lines:
            return ""
        return "Recent vitals and lab results:\n" + "\n".join(
            f"- {item}" for item in lines[:20]
        )

    @staticmethod
    def _build_allergies_memory_summary(allergies: List[Dict]) -> str:
        lines = []
        for allergy in allergies:
            name = (allergy.get("name") or "").strip()
            if not name:
                continue
            reaction = (allergy.get("reaction") or "").strip()
            severity = (allergy.get("severity") or "").strip()
            line = name
            if reaction:
                line += f" -- {reaction}"
            if severity and severity not in ("unknown", ""):
                line += f" ({severity})"
            lines.append(line)
        if not lines:
            return ""
        return "Known allergies and adverse reactions:\n" + "\n".join(
            f"- {item}" for item in lines[:10]
        )

    @staticmethod
    def _build_follow_up_patient_context(
        vitals: List[Dict],
        medications: List[Dict],
        allergies: List[Dict],
        conditions: List[Dict],
        symptom_logs: List[Dict],
    ) -> str:
        """
        Builds the structured patient context used exclusively for follow-up question generation.
        Rules:
        - Vitals/labs: deduplicate by type (most recent per type), include only if recorded within 30 days.
        - Medications: include all regardless of age -- the LLM may ask whether the patient is still
          taking a medication if it could be causative.
        - Allergies: include all -- always relevant if causally connected to the current issue.
        - Conditions: include active/current only.
        - Symptoms: include only those logged within the last 30 days.
        """
        today = date.today()
        cutoff = today - timedelta(days=30)
        sections: List[str] = []

        # --- VITALS: most recent per type, last 30 days only ---
        latest_vitals: Dict[str, Dict] = {}
        for entry in vitals:
            vtype = (entry.get("type") or "").strip().lower()
            if not vtype:
                continue
            recorded_str = (entry.get("recorded_on") or "").strip()
            existing = latest_vitals.get(vtype)
            existing_date = (existing.get("recorded_on") or "") if existing else ""
            if existing is None or recorded_str > existing_date:
                latest_vitals[vtype] = entry

        vital_lines: List[str] = []
        for vtype, entry in sorted(latest_vitals.items()):
            recorded_str = (entry.get("recorded_on") or "").strip()
            if not (entry.get("value") or "").strip():
                continue
            try:
                if date.fromisoformat(recorded_str) >= cutoff:
                    vital_lines.append(
                        render_vital_for_prompt(entry, date_prefix="recorded ")
                    )
            except (ValueError, TypeError):
                pass

        if vital_lines:
            sections.append(
                "RECENT VITALS AND LABS (last 30 days -- quote these exact values in follow-up questions):\n"
                + "\n".join(f"- {line}" for line in vital_lines)
            )
        else:
            sections.append(
                "RECENT VITALS AND LABS: None recorded in the last 30 days -- do not reference vitals."
            )

        # --- MEDICATIONS: all, no date filter ---
        med_lines: List[str] = []
        for med in medications:
            name = (med.get("name") or "").strip()
            if not name:
                continue
            dose = (med.get("dose") or "").strip()
            schedule = (med.get("schedule") or "").strip()
            reason = (med.get("reason") or "").strip()
            line = name
            if dose:
                line += f" {dose}"
            if schedule:
                line += f" {schedule}"
            if reason:
                line += f" (for {reason})"
            med_lines.append(line)

        if med_lines:
            sections.append(
                "MEDICATIONS ON RECORD (ask whether the patient is still taking it if you think it "
                "could be causing or worsening the current issue):\n"
                + "\n".join(f"- {line}" for line in med_lines)
            )

        # --- ALLERGIES: all, no date filter ---
        allergy_lines: List[str] = []
        for allergy in allergies:
            name = (allergy.get("name") or "").strip()
            if not name:
                continue
            reaction = (allergy.get("reaction") or "").strip()
            severity = (allergy.get("severity") or "").strip()
            allergy_type = (allergy.get("allergy_type") or "").strip()
            line = name
            if allergy_type and allergy_type != "other":
                line += f" [{allergy_type}]"
            if reaction:
                line += f" → {reaction}"
            if severity and severity not in ("unknown", ""):
                line += f" ({severity})"
            allergy_lines.append(line)

        if allergy_lines:
            sections.append(
                "KNOWN ALLERGIES (use these if possibly related to the current issue):\n"
                + "\n".join(f"- {line}" for line in allergy_lines)
            )

        # --- CONDITIONS: active / not resolved ---
        condition_lines: List[str] = []
        for cond in conditions:
            name = (cond.get("name") or "").strip()
            if not name:
                continue
            status = (cond.get("status") or "unknown").strip().lower()
            if status in ("past", "resolved"):
                continue
            recorded = (cond.get("recorded_on") or "").strip()
            line = name
            if status and status != "unknown":
                line += f" ({status})"
            if recorded:
                line += f" -- since {recorded}"
            condition_lines.append(line)

        if condition_lines:
            sections.append(
                "ACTIVE CONDITIONS:\n"
                + "\n".join(f"- {line}" for line in condition_lines)
            )

        # --- SYMPTOMS: last 30 days only ---
        symptom_lines: List[str] = []
        for log in symptom_logs:
            symptom = (log.get("symptom") or "").strip()
            if not symptom:
                continue
            logged_for = (log.get("logged_for") or log.get("logged_at") or "").strip()
            try:
                if date.fromisoformat(logged_for[:10]) >= cutoff:
                    severity_val = log.get("severity")
                    line = symptom
                    if severity_val is not None:
                        line += f" (severity {severity_val}/10)"
                    if logged_for:
                        line += f" on {logged_for[:10]}"
                    symptom_lines.append(line)
            except (ValueError, TypeError):
                pass

        if symptom_lines:
            sections.append(
                "RECENT SYMPTOMS (last 30 days):\n"
                + "\n".join(f"- {line}" for line in symptom_lines)
            )

        return "\n\n".join(sections)

    @staticmethod
    def _build_safety_net_block(
        risk_level: str, triage_summary: dict, role_config
    ) -> str:
        """
        Appends a structured safety netting block for elevated/urgent/crisis answers.
        Escalation triggers come entirely from the LLM-generated triage_summary -- no
        hardcoded clinical content here.
        """
        if risk_level not in ("elevated", "urgent", "crisis"):
            return ""

        is_clinical = role_config and role_config.role_key in (
            "doctor",
            "nurse",
            "midwife",
            "physiotherapist",
            "healthcare_professional",
        )

        # Use LLM-generated escalation triggers; fall back to what_to_monitor if absent
        triggers = [
            str(t).strip()
            for t in (triage_summary.get("escalation_triggers") or [])
            if str(t).strip()
        ]
        if not triggers:
            triggers = [
                str(t).strip()
                for t in (triage_summary.get("what_to_monitor") or [])
                if str(t).strip()
            ]

        if not triggers:
            return ""

        trigger_lines = "\n".join(f"- {t}" for t in triggers[:5])

        if is_clinical:
            return (
                "\n\n---\n"
                "**Safety Netting -- Return Criteria**\n\n"
                "Reassess or escalate if any of the following occur:\n"
                f"{trigger_lines}\n\n"
                "Document the safety-netting advice given and the agreed review timeframe."
            )
        else:
            return (
                "\n\n---\n"
                "**Return immediately if:**\n\n"
                f"{trigger_lines}\n\n"
                "If symptoms are severe or rapidly worsening, contact your local emergency services."
            )

    def _build_medication_check(
        self,
        question: str,
        intent,
        medications: List[Dict],
        allergies: Optional[List[Dict]] = None,
        question_medications: Optional[List[str]] = None,
    ) -> Dict:
        question_lower = (question or "").lower()
        stored_names = [
            medication.get("name", "").strip()
            for medication in medications
            if medication.get("name", "").strip()
        ]
        names_from_question = list(question_medications or [])
        if not names_from_question:
            try:
                names_from_question = self.llm.extract_medication_mentions(question)
            except Exception as exc:
                print(f"Medication extraction failed: {exc}")

        names_in_question = [
            name for name in stored_names if name.lower() in question_lower
        ]

        candidate_names: List[str] = []
        for name in [*names_from_question, *names_in_question]:
            if name and name.lower() not in {item.lower() for item in candidate_names}:
                candidate_names.append(name)

        intent_category = getattr(intent, "intent_category", "")
        if intent_category == "medication_query" and len(candidate_names) < 2:
            for name in stored_names:
                if name.lower() not in {item.lower() for item in candidate_names}:
                    candidate_names.append(name)
                if len(candidate_names) >= 6:
                    break

        if not candidate_names:
            return {
                "resolved_medications": [],
                "unresolved_medications": [],
                "alerts": [],
                "allergy_conflicts": [],
            }
        result = self._medication_checker.check_interactions(candidate_names[:6])
        allergy_conflicts: List[Dict] = []
        for resolved in result.get("resolved_medications", []):
            for conflict in check_allergy_conflicts(resolved, allergies or []):
                allergy_conflicts.append(
                    {
                        **conflict,
                        "medication_name": resolved.get("canonical_name")
                        or resolved.get("query_name", "medicine"),
                    }
                )
        result["allergy_conflicts"] = allergy_conflicts
        return result

    @staticmethod
    def _append_medication_context(
        context: str,
        alerts: List[Dict],
        allergy_conflicts: Optional[List[Dict]] = None,
    ) -> str:
        if not alerts and not allergy_conflicts:
            return context
        alert_lines = [
            f"- {alert.get('pair')}: {alert.get('summary')}" for alert in alerts[:3]
        ]
        alert_lines.extend(
            f"- {conflict.get('medication_name', 'Medicine')} and recorded allergy "
            f"{conflict.get('allergy_name', '')}: {conflict.get('summary', '')}"
            for conflict in (allergy_conflicts or [])[:3]
        )
        return (
            context
            + "\n\nMedication interaction or recorded-allergy safety flags:\n"
            + "\n".join(alert_lines)
        )

    @staticmethod
    def _append_contradiction_context(context: str, contradictions: List[Dict]) -> str:
        """Evidence Ledger v2, #4: tells the answer model when retrieved
        sources disagree on the same intervention, so it describes both
        positions instead of silently picking one -- see
        backend/contradiction_detector.py for how these pairs are found."""
        if not contradictions:
            return context
        lines = [
            f"- {item.get('topic') or 'A finding'}: {item.get('description', '')} "
            f"({item.get('source_a_id')} vs {item.get('source_b_id')})"
            for item in contradictions[:3]
        ]
        return (
            context
            + "\n\nSources disagree on the following -- describe both positions rather "
            "than presenting only one as settled:\n" + "\n".join(lines)
        )

    @staticmethod
    def _summarize_resolved_medications(resolved_medications: List[Dict]) -> List[Dict]:
        return [
            {
                "query_name": item.get("query_name", ""),
                "canonical_name": item.get("canonical_name", ""),
                "effective_time": item.get("effective_time", ""),
            }
            for item in resolved_medications
        ]

    def _build_triage_summary(
        self,
        question: str,
        answer_markdown: str,
        intent,
        policy_decision,
        clinical_decision=None,
    ) -> Dict:
        if clinical_decision is not None:
            fallback = build_default_triage(intent, policy_decision)
            return normalize_triage_output(
                clinical_decision.build_triage_summary(),
                fallback,
            )
        fallback = build_default_triage(intent, policy_decision)
        intent_summary = (
            f"intent={getattr(intent, 'intent_category', '')}; "
            f"risk={getattr(intent, 'risk_level', '')}; "
            f"escalation_reason={getattr(intent, 'escalation_reason', '')}"
        )
        try:
            model_triage = self.llm.build_structured_triage(
                question=question,
                answer_markdown=answer_markdown,
                fallback_triage=fallback,
                intent_summary=intent_summary,
            )
        except Exception as exc:
            print(f"Structured triage generation failed: {exc}")
            model_triage = {}
        return normalize_triage_output(model_triage, fallback)

    def _enrich_prebuilt_payload(
        self,
        question: str,
        payload: Dict,
        user: Optional[str],
    ) -> Dict:
        enriched = dict(payload)
        trace = enriched.get("trace", {})
        risk_level = trace.get("risk_level") or (
            "crisis" if trace.get("crisis_detected") else "routine"
        )
        intent = SimpleNamespace(
            risk_level=risk_level,
            crisis_detected=trace.get("crisis_detected", False),
            escalation_reason=trace.get("moderation_category")
            or trace.get("retrieval_mode", ""),
        )
        policy = SimpleNamespace(
            action="escalate_only" if risk_level in ("urgent", "crisis") else "allow"
        )
        triage_summary = build_default_triage(intent, policy)
        enriched["triage_summary"] = triage_summary
        enriched.setdefault("medication_alerts", [])
        enriched.setdefault("resolved_medications", [])
        normalized_user = user.strip().lower() if user else None
        if normalized_user:
            UserStore.save_triage_summary(
                normalized_user,
                {
                    **triage_summary,
                    "question": question,
                    "trace_id": trace.get("trace_id"),
                },
            )
        return enriched

    @staticmethod
    def _build_personal_context_text(personal_context: List[Dict]) -> str:
        if not personal_context:
            return ""

        return "\n".join(
            f"- {item['title']}: {item['snippet']}" for item in personal_context
        )

    @staticmethod
    def _link_citations(answer_text: str, sources: List[Dict]) -> str:
        source_map = {
            source["source_id"]: source.get("url")
            for source in sources
            if source.get("source_id")
        }

        def replace_match(match: re.Match) -> str:
            source_id = match.group(1)
            url = source_map.get(source_id)
            if not url:
                return f"`[{source_id}]`"
            return f"[{source_id}]({url})"

        linked_answer = re.sub(r"\[(S\d+)\]", replace_match, answer_text)
        return linked_answer

    @staticmethod
    def _append_clinical_evidence_trail(
        answer_text: str, sources: List[Dict], role_key: str
    ) -> str:
        """Map evidence cited in a professional answer to its source title."""
        clinical_roles = {
            "doctor",
            "nurse",
            "midwife",
            "physiotherapist",
            "healthcare_professional",
        }
        if role_key not in clinical_roles or "## Evidence Used" in answer_text:
            return answer_text

        cited_ids: List[str] = []
        for source_id in re.findall(r"\[(S\d+)\]", answer_text):
            if source_id not in cited_ids:
                cited_ids.append(source_id)

        by_id = {source.get("source_id"): source for source in sources}
        evidence_lines = []
        for source_id in cited_ids[:3]:
            source = by_id.get(source_id)
            if not source:
                continue
            title = str(source.get("title") or "Evidence source").strip()
            authority = str(
                source.get("authority")
                or source.get("journal")
                or source.get("provider")
                or ""
            ).strip()
            label = f"{authority}: {title}" if authority else title
            evidence_lines.append(f"- [{source_id}] {label}")

        if not evidence_lines:
            return answer_text
        return answer_text.rstrip() + "\n\n## Evidence Used\n" + "\n".join(evidence_lines)

    @staticmethod
    def _build_limited_evidence_response(personal_context: List[Dict]) -> str:
        return (
            "I can't safely confirm an answer from the information available here. "
            "Tell me the exact symptom, medicine, or report wording you want help with. "
            "If this question came with an image, upload a clear close-up and briefly describe "
            "what the image shows and what has changed."
        )

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()
