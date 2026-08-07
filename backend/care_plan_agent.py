"""
Agentic care-plan generator for FlynnMed.

Uses an OpenAI tool-calling loop to gather NHS/NICE guidelines and PubMed
evidence before synthesising a structured, evidence-based care plan.
"""
from __future__ import annotations

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional

import openai

from backend.evidence_extractor import _extract_one_article
from backend.clinical_context_guard import (
    ClinicalContextDecision,
    build_review_required_plan,
    decision_from_dict,
    source_matches_context,
    validate_care_plan,
    validate_generated_answer,
)
from backend.official_guidance import OfficialGuidanceEngine
from backend.patient_history import build_patient_history_context
from backend.pubmed_search import PubMedCentralSearcher
from backend.query_expander import QueryExpander

# Sources the extractor confirms are near-zero relevance and don't answer the
# question are excluded outright -- same threshold and rationale as
# evidence_extractor.build_evidence_dossier's chat-pipeline fix.
_MISMATCH_THRESHOLD = 0.1

_TOOLS: List[Dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_nhs_guidelines",
            "description": (
                "Search NHS.uk and NICE for evidence-based clinical guidelines, monitoring "
                "targets, treatment thresholds, and care pathways for a given condition."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "condition": {
                        "type": "string",
                        "description": "Condition name, e.g. 'Type 2 Diabetes' or 'Hypertension'"
                    },
                    "aspect": {
                        "type": "string",
                        "description": (
                            "Specific aspect to retrieve, e.g. 'monitoring targets', "
                            "'medication guidance', 'when to refer', 'lifestyle advice', "
                            "'escalation criteria'"
                        )
                    }
                },
                "required": ["condition"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_pubmed_evidence",
            "description": (
                "Search PubMed Central for systematic reviews, RCTs, and meta-analyses "
                "on clinical interventions, lifestyle modifications, or monitoring strategies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "PubMed search query"
                    },
                    "focus": {
                        "type": "string",
                        "description": (
                            "The evidence focus, e.g. 'diet interventions', 'exercise RCT', "
                            "'sleep quality', 'medication adherence', 'self-monitoring outcomes'"
                        )
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_lifestyle_recommendations",
            "description": (
                "Retrieve evidence-based lifestyle guidance for a specific domain "
                "(diet, exercise, sleep, weight, stress, alcohol, smoking) for a condition."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string"},
                    "lifestyle_area": {
                        "type": "string",
                        "description": (
                            "One of: 'diet and nutrition', 'physical activity', "
                            "'sleep hygiene', 'weight management', 'stress and mental health', "
                            "'alcohol', 'smoking cessation'"
                        )
                    }
                },
                "required": ["condition", "lifestyle_area"]
            }
        }
    },
]

_PLAN_SCHEMA = {
    "type": "object",
    "required": [
        "condition", "title", "goals", "daily_tasks", "weekly_tasks",
        "medication_reminders", "lab_reminders", "escalation_thresholds",
        "lifestyle", "missed_care_checklist", "evidence_summary", "safety_notes"
    ],
    "additionalProperties": True,
    "properties": {
        "condition": {"type": "string"},
        "title": {"type": "string"},
        "goals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "metric": {"type": "string"},
                    "target_months": {"type": "integer"}
                }
            }
        },
        "daily_tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "time_of_day": {
                        "type": "string",
                        "enum": ["morning", "afternoon", "evening", "bedtime", "any"]
                    },
                    "rationale": {"type": "string"}
                }
            }
        },
        "weekly_tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "rationale": {"type": "string"}
                }
            }
        },
        "medication_reminders": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "medication": {"type": "string"},
                    "dose": {"type": "string"},
                    "timing": {"type": "string"},
                    "notes": {"type": "string"}
                }
            }
        },
        "lab_reminders": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "test": {"type": "string"},
                    "frequency_months": {"type": "integer"},
                    "notes": {"type": "string"},
                    "target_value": {"type": "string"}
                }
            }
        },
        "escalation_thresholds": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symptom": {"type": "string"},
                    "threshold": {"type": "string"},
                    "action": {"type": "string"},
                    "urgency": {
                        "type": "string",
                        "enum": ["call_999", "a_and_e", "gp_same_day", "gp_routine", "self_monitor"]
                    }
                }
            }
        },
        "lifestyle": {
            "type": "object",
            "properties": {
                "diet": {"type": "string"},
                "exercise": {"type": "string"},
                "sleep": {"type": "string"},
                "weight": {"type": "string"},
                "mental_health": {"type": "string"},
                "smoking": {"type": "string"},
                "alcohol": {"type": "string"},
                "other": {"type": "string"}
            }
        },
        "missed_care_checklist": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "string"},
                    "frequency_months": {"type": "integer"},
                    "notes": {"type": "string"}
                }
            }
        },
        "evidence_summary": {"type": "string"},
        "safety_notes": {"type": "string"}
    }
}


class CarePlanAgent:
    MAX_ITERATIONS = 7

    def __init__(self) -> None:
        self._client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        self._model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._guidance = OfficialGuidanceEngine()
        self._pubmed = PubMedCentralSearcher()
        self._query_expander = QueryExpander()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        condition: str,
        user_context: Dict,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> Dict:
        def emit(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        profile = user_context.get("profile", {})
        meds = user_context.get("medications", [])
        existing_conditions = user_context.get("conditions", [])
        # Tail, not head: a clarification the user answered (e.g. resolving an
        # ambiguous term) is appended to the end by the caller and must survive
        # truncation.
        chat_summary = (user_context.get("chat_summary") or "")[-800:]
        role = profile.get("role") or "Patient"
        clinical_context: Optional[ClinicalContextDecision] = user_context.get("clinical_context")
        self._clinical_context = clinical_context

        if clinical_context and clinical_context.requires_clarification:
            emit("Clinical context needs clarification before a plan can be generated.")
            return build_review_required_plan(clinical_context)
        if (
            clinical_context
            and clinical_context.domain
            and clinical_context.topic
            and clinical_context.requested_topic
            and clinical_context.requested_topic.strip().lower() != clinical_context.topic.strip().lower()
        ):
            # A clarification answer can resolve an initially wrong plan label
            # (for example, the user selected asthma but confirmed the record is
            # a urine-flow test). Generate against the confirmed topic.
            condition = clinical_context.topic

        patient_history = user_context.get("patient_history") or build_patient_history_context(
            longitudinal_memory=user_context.get("longitudinal_memory") or "",
            medications=meds,
            triage_summaries=user_context.get("triage_summaries") or [],
            user_profile=profile,
            allergies=user_context.get("allergies") or [],
            conditions=existing_conditions,
            vitals=user_context.get("vitals") or [],
        )
        history_block = patient_history.as_prompt_block() or "No recorded patient history."

        self._extraction_context = {
            "question": f"Care plan for {condition}",
            "patient_summary": history_block,
            "medications": [m.get("name", "") for m in meds if isinstance(m, dict) and m.get("name")],
            "conditions": [c.get("name", "") for c in existing_conditions if isinstance(c, dict) and c.get("name")],
        }

        system_prompt = f"""You are a specialist evidence-based care-plan assistant embedded in FlynnMed, a UK-focused clinical AI platform (NHS-aligned).

Build a comprehensive, personalised care plan for: **{condition}**

Patient context:
- Account role: {role}
{history_block}
- Recent health chat context: {chat_summary if chat_summary else "none"}

{clinical_context.as_prompt_block() if clinical_context else "Clinical context adjudication: no specialty conflict detected; do not infer a diagnosis from a request alone."}

AGENT RULES:
1. Use tools to gather real evidence -- do not invent guidelines or statistics.
2. Search at minimum: (a) NHS/NICE monitoring targets and treatment thresholds, (b) lifestyle evidence (diet, exercise, sleep), (c) escalation/red-flag criteria.
3. Only include lifestyle advice that is relevant to the confirmed topic and supported by retrieved evidence. Leave irrelevant domains out; never fill a template with generic advice.
4. Daily tasks must be concrete and achievable (e.g. "Take metformin 500mg with breakfast" -- not "manage diabetes").
5. Lab reminders must use correct NHS/NICE frequencies in months.
6. Escalation thresholds must use values only when the retrieved guidance supports them. Do not invent targets or ranges.
7. Safety notes must flag medication interactions, red flags, and safeguarding concerns specific to this condition.
8. After gathering evidence, generate the final plan JSON only -- no prose outside the JSON.
9. Treat the clinical context adjudication above as binding. Do not reinterpret a measurement, unit, or test as another specialty. If the requested topic is not established as a diagnosis, describe it as a concern to review rather than a confirmed condition; record unresolved ambiguity in safety_notes.
10. If evidence is insufficient for a task, target, medication, or screening interval, omit it and say that it needs clinician confirmation. When data is ambiguous, preserve that ambiguity instead of guessing."""

        messages: List[Dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create a full, evidence-based care plan for: {condition}"},
        ]

        emit(f"Starting evidence search for {condition}...")

        for _ in range(self.MAX_ITERATIONS):
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=_TOOLS,
                tool_choice="auto",
                temperature=0.15,
            )
            msg = response.choices[0].message

            # Build the assistant message dict manually (avoids Pydantic serialisation issues)
            assistant_msg: Dict = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            if not msg.tool_calls:
                break

            for tc in msg.tool_calls:
                fname = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                if fname == "search_nhs_guidelines":
                    cond = args.get("condition", condition)
                    aspect = args.get("aspect", "")
                    if clinical_context and clinical_context.query_terms:
                        cond = " ".join(clinical_context.query_terms)
                    emit(f"Searching NHS/NICE: {cond} -- {aspect or 'guidelines'}")
                    result = self._nhs(f"{cond} {aspect}".strip())

                elif fname == "search_pubmed_evidence":
                    query = args.get("query", condition)
                    if clinical_context and clinical_context.query_terms:
                        query = " ".join(clinical_context.query_terms)
                    emit(f"Searching PubMed: {query}")
                    result = self._pubmed_search(query)

                elif fname == "search_lifestyle_recommendations":
                    cond = args.get("condition", condition)
                    area = args.get("lifestyle_area", "lifestyle")
                    if clinical_context and clinical_context.query_terms:
                        cond = " ".join(clinical_context.query_terms)
                    emit(f"Searching lifestyle evidence: {area} for {cond}")
                    result = self._nhs(f"{cond} {area} recommendations")

                else:
                    result = "Unknown tool."

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result or "No relevant results found.",
                })

        emit("Synthesising your personalised care plan...")

        messages.append({
            "role": "user",
            "content": (
                "Now produce the final care plan as a valid JSON object matching this schema:\n"
                f"{json.dumps(_PLAN_SCHEMA, indent=2)}\n\n"
                "Requirements:\n"
                "- goals: 3-5 specific measurable goals with target_months\n"
                "- daily_tasks: 4-8 actionable daily actions with time_of_day set\n"
                "- weekly_tasks: 3-5 actions\n"
                "- lab_reminders: ALL NHS/NICE recommended tests with correct frequency_months and target_value\n"
                "- escalation_thresholds: 4-6 items with real clinical values in threshold field\n"
                "- lifestyle: include only relevant evidence-backed domains; do not fill irrelevant domains\n"
                "- missed_care_checklist: include only reviews or screenings supported by the retrieved evidence\n"
                "- evidence_summary: name the retrieved sources or guideline identifiers only when present\n"
                "Return ONLY the JSON object."
            ),
        })

        final = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.05,
        )

        raw = final.choices[0].message.content or "{}"
        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            plan = {}

        now = datetime.now(timezone.utc).isoformat()
        plan.setdefault("condition", condition)
        plan.setdefault("title", f"{condition} Care Plan")
        plan.setdefault("goals", [])
        plan.setdefault("daily_tasks", [])
        plan.setdefault("weekly_tasks", [])
        plan.setdefault("medication_reminders", [])
        plan.setdefault("lab_reminders", [])
        plan.setdefault("escalation_thresholds", [])
        plan.setdefault("lifestyle", {})
        plan.setdefault("missed_care_checklist", [])
        plan.setdefault("evidence_summary", "Evidence references returned by the retrieval agent.")
        plan.setdefault("safety_notes", "Confirm patient-specific actions with the relevant healthcare professional.")
        if clinical_context:
            validation = validate_care_plan(plan, clinical_context)
            plan["clinical_context"] = clinical_context.as_dict()
            plan["validation"] = {
                "status": "passed" if validation["valid"] else "review_required",
                "violations": validation["violations"],
            }
            if not validation["valid"]:
                emit("The generated plan did not pass the specialty safety check; generation was stopped.")
                return build_review_required_plan(clinical_context)
        plan.update({
            "id": uuid.uuid4().hex,
            "status": "active",
            "created_at": now,
            "updated_at": now,
            "after_visit_notes": [],
            "gp_prep_summary": None,
        })

        for key in ("goals", "daily_tasks", "weekly_tasks", "medication_reminders",
                    "lab_reminders", "escalation_thresholds", "missed_care_checklist"):
            for item in plan.get(key, []):
                item.setdefault("id", uuid.uuid4().hex[:12])

        return plan

    def generate_gp_prep(self, plan: Dict, user_context: Dict) -> str:
        profile = user_context.get("profile", {})
        name = (profile.get("display_name") or "the patient").split()[0]
        clinical_context: Optional[ClinicalContextDecision] = (
            user_context.get("clinical_context")
            or decision_from_dict(plan.get("clinical_context"))
        )

        patient_history = user_context.get("patient_history") or build_patient_history_context(
            longitudinal_memory=user_context.get("longitudinal_memory") or "",
            medications=user_context.get("medications") or [],
            triage_summaries=user_context.get("triage_summaries") or [],
            user_profile=profile,
            allergies=user_context.get("allergies") or [],
            conditions=user_context.get("conditions") or [],
            vitals=user_context.get("vitals") or [],
        )
        history_block = patient_history.as_prompt_block() or "No recorded patient history."

        prompt = (
            f"You are FlynnMed helping {name} prepare for their GP appointment about "
            f"{plan.get('condition', 'their condition')}.\n\n"
            f"Patient history:\n{history_block}\n\n"
            f"{clinical_context.as_prompt_block() if clinical_context else 'Clinical context adjudication: do not infer a diagnosis from the plan title alone.'}\n\n"
            f"Care plan summary:\n"
            f"Goals: {json.dumps(plan.get('goals', []))}\n"
            f"Medications: {json.dumps(plan.get('medication_reminders', []))}\n"
            f"Upcoming labs: {json.dumps(plan.get('lab_reminders', []))}\n"
            f"Warning signs: {json.dumps(plan.get('escalation_thresholds', []))}\n"
            f"Missed care: {json.dumps(plan.get('missed_care_checklist', []))}\n"
            f"After-visit notes: {json.dumps(plan.get('after_visit_notes', []))}\n\n"
            "Write a structured GP appointment preparation guide with these sections:\n"
            "## Questions to ask your GP (5-7 specific questions)\n"
            "## Medications to review\n"
            "## Tests and results to request\n"
            "## Symptoms or concerns to mention\n"
            "## What you want to achieve from this appointment\n\n"
            "If anything in the care plan data above is ambiguous or uncertain, add it as an "
            "item under \"Symptoms or concerns to mention\" so the patient can clarify it with "
            "their GP directly.\n\n"
            "Keep it concise, patient-friendly, and in plain language."
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        result = response.choices[0].message.content or ""
        if clinical_context and not validate_generated_answer(result, clinical_context)["valid"]:
            return clinical_context.correction_message()
        return result

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _filter_relevant(
        self, results: List[Dict], title_key: str, snippet_keys: List[str]
    ) -> List[Dict]:
        """
        Filters raw NHS/PubMed search results through the same specialty-mismatch
        extractor the chat pipeline uses (evidence_extractor._extract_one_article),
        dropping sources confirmed to concern a different clinical meaning/system
        than this patient's confirmed data before they can reach the synthesis LLM.
        """
        if not results:
            return []

        ctx = getattr(self, "_extraction_context", None) or {
            "question": "",
            "patient_summary": "Patient profile not recorded",
            "medications": [],
            "conditions": [],
        }

        sources = []
        for i, r in enumerate(results):
            snippet = ""
            for key in snippet_keys:
                snippet = (r.get(key) or "").strip()
                if snippet:
                    break
            sources.append({"source_id": f"src-{i}", "title": r.get(title_key, "") or "Untitled", "snippet": snippet})

        decision = getattr(self, "_clinical_context", None)
        if decision and decision.domain:
            compatible = [
                result for result, source in zip(results, sources)
                if source_matches_context(source["title"], source["snippet"], decision)
            ]
            results = compatible
            if not results:
                return []
            sources = [
                source for source in sources
                if source_matches_context(source["title"], source["snippet"], decision)
            ]

        fake_llm = SimpleNamespace(client=self._client)
        with ThreadPoolExecutor(max_workers=min(4, len(sources))) as executor:
            futures = [
                executor.submit(
                    _extract_one_article,
                    fake_llm, source, ctx["question"], ctx["patient_summary"],
                    ctx["medications"], ctx["conditions"],
                )
                for source in sources
            ]
            extracted = [future.result() for future in futures]

        return [
            result
            for result, art in zip(results, extracted)
            if not art.specialty_mismatch
            and not (art.alignment_confidence < _MISMATCH_THRESHOLD and not art.answers_question)
        ]

    def _nhs(self, query: str) -> str:
        try:
            results = self._guidance.search([query], per_source_limit=2)
            if not results:
                return "No NHS/NICE results found."
            relevant = self._filter_relevant(results[:4], title_key="title", snippet_keys=["snippet", "content"])
            if not relevant:
                return "No relevant NHS/NICE results found for this patient's confirmed data."
            parts = []
            for r in relevant:
                title = r.get("title", "")
                snippet = (r.get("snippet") or r.get("content") or "")[:500]
                url = r.get("url", "")
                parts.append(f"[{title}]\n{snippet}\nSource: {url}")
            return "\n\n---\n\n".join(parts)
        except Exception as exc:
            return f"NHS search error: {exc}"

    def _pubmed_search(self, query: str) -> str:
        try:
            # The agent-chosen query and the query-variant/HyDE expansion run
            # concurrently -- neither depends on the other -- so the extra
            # variant search below adds at most one more PubMed round trip,
            # not a serial LLM-call-then-search chain.
            with ThreadPoolExecutor(max_workers=2) as executor:
                primary_future = executor.submit(self._pubmed.search_article_records, query, 3)
                expansion_future = executor.submit(self._query_expander.expand_with_hyde, query)
                records = primary_future.result()
                try:
                    hyde_result = expansion_future.result()
                except Exception as exc:
                    print(f"[CarePlanAgent] query expansion failed: {exc}")
                    hyde_result = None

            if hyde_result and hyde_result.query_variants:
                best_variant = hyde_result.query_variants[0].strip()
                if best_variant and best_variant.lower() != query.strip().lower():
                    try:
                        records = records + self._pubmed.search_article_records(best_variant, 2)
                    except Exception as exc:
                        print(f"[CarePlanAgent] variant PubMed search failed: {exc}")

            if not records:
                return "No PubMed results found."
            relevant = self._filter_relevant(records[:5], title_key="title", snippet_keys=["abstract"])
            if not relevant:
                return "No relevant PubMed results found for this patient's confirmed data."
            parts = []
            for r in relevant:
                title = r.get("title", "")
                abstract = (r.get("abstract") or "")[:450]
                year = r.get("year", "")
                journal = r.get("journal", "")
                parts.append(f"[{title} -- {journal} {year}]\n{abstract}")
            return "\n\n---\n\n".join(parts)
        except Exception as exc:
            return f"PubMed error: {exc}"
