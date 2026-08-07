"""
ClinicalOrchestrator: central workflow engine for the health assistant.

Coordinates role detection, risk classification, safety policy gating,
and an LLM-driven agentic retrieval loop that decides which tools to
call before generating the final answer.

Architecture
------------
Deterministic safety layer (runs first, always):
  1. Role resolution
  2. Patient history context
  3. Crisis pre-screen (regex)
  4. Moderation
  5. Intent classification
  6. Policy gate (8 hard safety gates)
  7. Pathway context

Agentic retrieval layer (LLM drives this):
  8. AgenticRetrievalLoop: the model chooses which tools to call
     - search_nhs_guidance    NHS/NICE Tier 1 evidence
     - search_pubmed          PubMed Central Tier 2-3 evidence
     - check_drug_interactions openFDA drug label warnings
     - search_patient_documents patient uploaded records
     - search_clinical_trials ClinicalTrials.gov
  9. Fallback retrieval if agent returns nothing
  10. Evidence ranking (deterministic quality gate)
  11. Evidence dossier (anti-hallucination extraction)
  12. Context assembly for the final LLM answer
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from backend.clinical_decision_support import (
    ClinicalDecision,
    ClinicalDecisionSupportEngine,
)
from backend.clinical_context_guard import (
    ClinicalContextDecision,
    adjudicate_patient_context,
    source_matches_context,
)
from backend.evidence_ranker import EvidenceRanker
from backend.intent_risk_classifier import IntentClassification, IntentRiskClassifier
from backend.patient_history import PatientHistoryContext, build_patient_history_context
from backend.policy_engine import PolicyEngine, PolicyDecision
from backend.response_templates import build_crisis_response
from backend.agentic_health_contract import (
    current_location_from_profile,
    operating_contract_prompt,
    select_skills,
)
from backend.role_router import RoleConfig, RoleRouter
from backend.task_mode import TaskModeDecision, decide_task_mode
from backend.utils import build_excerpt

if TYPE_CHECKING:
    from backend.context_graph import ContextGraph
    from backend.evidence_schema import ExtractedEvidenceDossier
    from backend.memory_store import MemoryStore
    from backend.moderation_ml import ModerationEnsemble
    from backend.official_guidance import OfficialGuidanceEngine
    from backend.pubmed_search import PubMedCentralSearcher
    from backend.query_expander import QueryExpander
    from backend.summarizer import LLMHelper


# ---------------------------------------------------------------------------
# Agentic retrieval loop
# ---------------------------------------------------------------------------


class AgenticRetrievalLoop:
    """
    LLM-driven tool-calling retrieval agent.

    Given the clinical question and patient context the model decides which
    sources to fetch -- NHS guidance, PubMed, drug interactions, personal
    documents, clinical trials -- and in what order. It runs until it has
    enough evidence or exhausts its iteration budget.

    This replaces the hardcoded parallel retrieval pipeline with a
    model-driven workflow.
    """

    _TOOLS: List[Dict] = [
        {
            "type": "function",
            "function": {
                "name": "search_nhs_guidance",
                "description": (
                    "Search NHS and NICE official guidance for UK clinical guidelines, "
                    "treatment recommendations, prescribing information, and patient safety advice. "
                    "Call this first for any clinical, medication, or condition question. "
                    "Returns Tier 1 (highest-authority) evidence."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Clinical search query, e.g. 'hypertension management in adults with CKD'",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_pubmed",
                "description": (
                    "Search PubMed Central for biomedical research literature including "
                    "clinical trials, systematic reviews, and research articles. "
                    "Use for specific conditions, treatment mechanisms, or when NHS guidance "
                    "needs supporting research evidence. "
                    "Returns Tier 2-3 evidence."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Medical search query, e.g. 'metformin HbA1c type 2 diabetes systematic review'",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_drug_interactions",
                "description": (
                    "Look up openFDA drug label data for interaction warnings, contraindications, "
                    "side effects, and dosing information. "
                    "Use whenever the question involves medications or the patient's medication "
                    "list may interact with the topic being discussed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "medications": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Medication names to check, e.g. ['metformin', 'lisinopril']",
                        }
                    },
                    "required": ["medications"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_patient_documents",
                "description": (
                    "Search the patient's uploaded health documents and personal records. "
                    "Use when the question relates to their specific test results, uploaded "
                    "discharge letters, clinic letters, or personal health history."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to look for in the patient's personal documents",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_clinical_trials",
                "description": (
                    "Search ClinicalTrials.gov for recruiting clinical trials. "
                    "Use only when the patient explicitly asks about trials, experimental "
                    "treatments, or eligibility for research studies. Call at most once."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "condition": {
                            "type": "string",
                            "description": "Medical condition to find trials for",
                        },
                        "location": {
                            "type": "string",
                            "description": "Preferred trial location (default: United Kingdom)",
                        },
                    },
                    "required": ["condition"],
                },
            },
        },
    ]

    def __init__(
        self,
        llm: "LLMHelper",
        official_guidance: "OfficialGuidanceEngine",
        pubmed: "PubMedCentralSearcher",
        memory: "MemoryStore",
        user: Optional[str],
        query_expander: Optional["QueryExpander"] = None,
    ) -> None:
        self.llm = llm
        self.official_guidance = official_guidance
        self.pubmed = pubmed
        self.memory = memory
        self.user = user
        self.query_expander = query_expander
        # Set once per run() call so the agent's own tool calls (e.g. _search_personal,
        # invoked later in the same run during the discretionary loop) can reuse the
        # HyDE passage generated up front instead of the query alone.
        self._current_hyde_passage = ""

    def run(
        self,
        question: str,
        patient_summary: str,
        role_key: str,
        pathway_hint: str,
        patient_medications: Optional[List[str]] = None,
        current_location: str = "",
        selected_skills: Optional[List[str]] = None,
        max_iterations: int = 5,
    ) -> Dict:
        """
        Run the agentic retrieval loop.

        Returns a dict with:
          collected_sources   list of source dicts (NHS, PubMed, openFDA)
          personal_context    list of personal-document match dicts
          trial_results       list of clinical trial dicts
          tool_calls_made     audit log of every tool call made
        """
        pathway_guidance = {
            "maternity": (
                "Prioritise RCOG and NICE maternity guidelines. "
                "Check pregnancy contraindications if medications are mentioned."
            ),
            "msk": (
                "Prioritise NICE MSK guidelines and physiotherapy evidence. "
                "Search for the specific injury or condition plus rehabilitation."
            ),
            "medications": (
                "Always check drug interactions. "
                "Search NHS/BNF for prescribing guidance."
            ),
            "chronic_conditions": (
                "Prioritise NICE chronic disease guidelines. "
                "Focus on long-term management and patient-specific risks."
            ),
        }.get(
            pathway_hint,
            "Search current official guidance first, then research evidence if needed.",
        )

        med_hint = ""
        if patient_medications:
            med_hint = (
                f"\nPatient's current medications: {', '.join(patient_medications[:8])}. "
                "Consider checking drug interactions if relevant."
            )

        system_prompt = (
            "You are a clinical evidence retrieval agent for a worldwide health-information assistant.\n"
            f"{operating_contract_prompt(selected_skills or ['evidence_retrieval'], current_location)}\n\n"
            "Your task: decide which ADDITIONAL tools to call to gather the right evidence BEFORE the "
            "answer is written. search_nhs_guidance and search_pubmed have already been run once with "
            "the base question (see the tool results below) -- that baseline search is mandatory and "
            "always happens, so do not skip evidence gathering.\n"
            "Do NOT answer the question yourself.\n\n"
            f"Clinical role: {role_key}\n"
            f"Pathway: {pathway_hint}\n"
            f"Retrieval strategy: {pathway_guidance}{med_hint}\n\n"
            "Rules:\n"
            "- Call search_nhs_guidance again only if a more specific or differently-worded query would "
            "surface better guidance than the baseline search did.\n"
            "- Call search_pubmed again only if a more specific research query (e.g. a named drug, "
            "mechanism, or sub-topic) would surface better evidence than the baseline search did.\n"
            "- Call check_drug_interactions if the question involves medications or interactions.\n"
            "- Call search_patient_documents if the question relates to the patient's own records.\n"
            "- Call search_clinical_trials ONLY if the question explicitly asks about trials.\n"
            "- If the patient context above already gives a specific, confirmed meaning for an "
            "otherwise ambiguous term in the question, use that confirmed meaning/terminology in "
            "every search query below -- never search using the raw ambiguous wording alone, "
            "since that risks retrieving guidance for the wrong meaning entirely.\n"
            "- Make at most 3 additional tool calls total. Stop as soon as you have sufficient evidence.\n"
            "- When you have finished gathering evidence, respond with the word DONE."
        )

        messages: List[Dict] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Gather evidence for this question: {question}\n\n"
                    f"Patient context:\n{patient_summary}"
                ),
            },
        ]

        collected_sources: List[Dict] = []
        personal_context: List[Dict] = []
        trial_results: List[Dict] = []
        tool_calls_made: List[Dict] = []
        hyde_passage = ""
        query_variants: List[str] = []

        # Baseline evidence retrieval is mandatory for every question -- NHS/NICE
        # guidance and PubMed/PMC literature must always be attempted at least once,
        # regardless of what the agent decides. This closes the gap where the agent
        # judged a question didn't "need" evidence and the final answer was written
        # from unbacked general knowledge with no citation at all.
        #
        # The query-variant + HyDE expansion call rides in the SAME parallel batch as
        # the two mandatory searches rather than running before them, so it adds no
        # serial step to the critical path -- worst case it's the slowest of the three
        # and the batch waits on it exactly as long as it already waits on NHS/PubMed.
        def _run_expansion():
            if not self.query_expander:
                return None
            return self.query_expander.expand_with_hyde(question, patient_summary)

        mandatory_tools = (
            ("search_nhs_guidance", self._search_nhs),
            ("search_pubmed", self._search_pubmed),
        )
        hyde_result = None
        with ThreadPoolExecutor(max_workers=3) as executor:
            search_futures = {
                executor.submit(search_fn, question): tool_name
                for tool_name, search_fn in mandatory_tools
            }
            expansion_future = executor.submit(_run_expansion)

            for future in search_futures:
                tool_name = search_futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"[AgenticLoop] mandatory {tool_name} failed: {exc}")
                    continue

                sources = result.get("sources") or []
                print(f"[AgenticLoop] mandatory {tool_name}({question!r}) -> {len(sources)} source(s)")
                collected_sources.extend(sources)
                tool_calls_made.append(
                    {
                        "tool": tool_name,
                        "args": {"query": question},
                        "iteration": 0,
                        "mandatory": True,
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"[Mandatory baseline {tool_name} result] "
                            + result.get("summary", "No results.")[:2000]
                        ),
                    }
                )

            try:
                hyde_result = expansion_future.result()
            except Exception as exc:
                print(f"[AgenticLoop] query expansion failed: {exc}")

        # One bounded follow-up round using the single best variant (not all 3, to
        # keep the added cost small) -- only fires when expansion actually produced
        # something different from the raw question, and only these two calls wait
        # on the expansion result rather than the whole request.
        if hyde_result and hyde_result.query_variants:
            best_variant = hyde_result.query_variants[0]
            query_variants = hyde_result.query_variants
            hyde_passage = hyde_result.hypothetical_passage
            self._current_hyde_passage = hyde_passage
            if best_variant and best_variant.strip().lower() != question.strip().lower():
                with ThreadPoolExecutor(max_workers=2) as executor:
                    variant_futures = {
                        executor.submit(search_fn, best_variant): tool_name
                        for tool_name, search_fn in mandatory_tools
                    }
                    for future in variant_futures:
                        tool_name = variant_futures[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            print(f"[AgenticLoop] variant {tool_name} failed: {exc}")
                            continue

                        sources = result.get("sources") or []
                        print(
                            f"[AgenticLoop] variant {tool_name}({best_variant!r}) "
                            f"-> {len(sources)} source(s)"
                        )
                        collected_sources.extend(sources)
                        tool_calls_made.append(
                            {
                                "tool": tool_name,
                                "args": {"query": best_variant},
                                "iteration": 0,
                                "mandatory": True,
                                "variant": True,
                            }
                        )

            if hyde_passage:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "[Query expansion] Additional retrieval-friendly query variants "
                            f"already generated for this question: {query_variants}. "
                            "A hypothetical reference passage was also generated for retrieval "
                            "matching purposes only (never cite or repeat it to the user): "
                            f"{hyde_passage[:400]}"
                        ),
                    }
                )

        for iteration in range(max_iterations):
            try:
                response = self.llm.client.chat.completions.create(
                    model=self.llm.AUX_MODEL,
                    messages=messages,
                    tools=self._TOOLS,
                    tool_choice="auto",
                    temperature=0,
                    max_completion_tokens=400,
                )
            except Exception as exc:
                print(f"[AgenticLoop] LLM call failed on iteration {iteration}: {exc}")
                break

            msg = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # Build a serializable dict for the assistant turn
            assistant_entry: Dict = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_entry)

            # No tool calls: agent is done retrieving
            if not msg.tool_calls or finish_reason == "stop":
                break

            # Execute each tool call and collect results
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}

                tool_calls_made.append(
                    {
                        "tool": fn_name,
                        "args": args,
                        "iteration": iteration,
                    }
                )
                print(f"[AgenticLoop] {fn_name}({args})")

                result = self._execute_tool(fn_name, args)

                if "sources" in result:
                    collected_sources.extend(result["sources"])
                if "personal_matches" in result:
                    personal_context.extend(result["personal_matches"])
                if "trials" in result:
                    trial_results.extend(result["trials"])

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result.get("summary", "No results.")[:2000],
                    }
                )

        return {
            "collected_sources": collected_sources,
            "personal_context": personal_context,
            "trial_results": trial_results,
            "tool_calls_made": tool_calls_made,
            "hyde_passage": hyde_passage,
            "query_variants": query_variants,
        }

    # -- Tool implementations ------------------------------------------------

    def _execute_tool(self, name: str, args: Dict) -> Dict:
        try:
            if name == "search_nhs_guidance":
                return self._search_nhs(args.get("query", ""))
            if name == "search_pubmed":
                return self._search_pubmed(args.get("query", ""))
            if name == "check_drug_interactions":
                return self._check_drug_interactions(args.get("medications", []))
            if name == "search_patient_documents":
                return self._search_personal(args.get("query", ""))
            if name == "search_clinical_trials":
                return self._search_trials(
                    args.get("condition", ""),
                    args.get("location", ""),
                )
            return {"summary": f"Unknown tool: {name}"}
        except Exception as exc:
            print(f"[AgenticLoop] Tool {name} raised: {exc}")
            return {"summary": f"{name} error: {exc}", "sources": []}

    def _search_nhs(self, query: str) -> Dict:
        if not query:
            return {"summary": "No query provided.", "sources": []}
        sources = self.official_guidance.search([query], 1)
        return {
            "sources": sources,
            "summary": f"Found {len(sources)} NHS/NICE sources for '{query}'.",
        }

    def _search_pubmed(self, query: str) -> Dict:
        if not query:
            return {"summary": "No query provided.", "sources": []}
        records = self.pubmed.search_article_records(query, 2)
        sources: List[Dict] = []
        memory_entries: List[Dict] = []

        for record in records:
            pmcid = record.get("pmcid", "")
            try:
                sections = self.pubmed.fetch_article_sections(pmcid)
            except Exception:
                sections = {}

            # Pick the best section
            section_text = ""
            section_name = "abstract"
            for key in ("discussion", "conclusion", "introduction"):
                text = (sections.get(key) or "").strip()
                if text:
                    section_text = text
                    section_name = key
                    break

            if section_text:
                entry_key = f"{self.user or 'global'}:pmc:{pmcid}:{section_name}"
                sources.append(
                    {
                        "source_id": f"pmc-{pmcid}",
                        "title": record.get("title", "Untitled"),
                        "journal": record.get("journal", ""),
                        "year": record.get("year", ""),
                        "authors": record.get("authors", ""),
                        "url": record.get("url", ""),
                        "pmcid": pmcid,
                        "section": section_name,
                        "snippet": section_text[:300],
                        "detail_snippet": section_text[:800],
                        "source_type": "pubmed_literature",
                        "provider": "Europe PMC / PubMed Central",
                        "query": query,
                    }
                )
                memory_entries.append(
                    {
                        "text": section_text,
                        "metadata": {
                            "type": "pubmed",
                            "source_type": "pubmed_literature",
                            "pmcid": pmcid,
                            "section": section_name,
                            "title": record.get("title", "Untitled"),
                            "journal": record.get("journal", ""),
                            "year": record.get("year", ""),
                            "authors": record.get("authors", ""),
                            "url": record.get("url", ""),
                            "query": query,
                            "entry_key": entry_key,
                        },
                        "user": self.user,
                        "entry_key": entry_key,
                    }
                )

            # Also store the abstract
            abstract = record.get("abstract", "")
            if abstract:
                abs_key = f"{self.user or 'global'}:pmc:{pmcid}:abstract"
                sources.append(
                    {
                        "source_id": f"pmc-{pmcid}-abs",
                        "title": record.get("title", "Untitled"),
                        "journal": record.get("journal", ""),
                        "year": record.get("year", ""),
                        "authors": record.get("authors", ""),
                        "url": record.get("url", ""),
                        "pmcid": pmcid,
                        "section": "abstract",
                        "snippet": abstract[:300],
                        "detail_snippet": abstract[:800],
                        "source_type": "pubmed_literature",
                        "provider": "Europe PMC / PubMed Central",
                        "query": query,
                    }
                )
                memory_entries.append(
                    {
                        "text": abstract,
                        "metadata": {
                            "type": "pubmed",
                            "source_type": "pubmed_literature",
                            "pmcid": pmcid,
                            "section": "abstract",
                            "title": record.get("title", "Untitled"),
                            "journal": record.get("journal", ""),
                            "year": record.get("year", ""),
                            "authors": record.get("authors", ""),
                            "url": record.get("url", ""),
                            "query": query,
                            "entry_key": abs_key,
                        },
                        "user": self.user,
                        "entry_key": abs_key,
                    }
                )

        if memory_entries:
            try:
                self.memory.add_entries(memory_entries)
            except Exception as exc:
                print(f"[AgenticLoop] Memory add failed: {exc}")

        return {
            "sources": sources,
            "summary": f"Found {len(sources)} PubMed sources for '{query}'.",
        }

    def _check_drug_interactions(self, medications: List[str]) -> Dict:
        if not medications:
            return {"summary": "No medications provided.", "sources": []}
        try:
            from backend.medication_checker import MedicationChecker

            checker = MedicationChecker()
            result = checker.check_interactions(medications)
            alerts = result.get("alerts", [])
            sources: List[Dict] = []
            for alert in alerts[:4]:
                pair = alert.get("pair", "medication pair")
                summary_text = alert.get("summary", "")
                if summary_text:
                    sources.append(
                        {
                            "source_id": f"fda-{pair.replace(' ', '-')[:40]}",
                            "title": f"Drug interaction: {pair}",
                            "snippet": str(summary_text)[:300],
                            "detail_snippet": str(summary_text)[:800],
                            "source_type": "official_guidance",
                            "provider": "openFDA",
                            "url": "https://open.fda.gov/",
                            "query": f"drug interactions {' '.join(medications)}",
                        }
                    )
            msg = (
                f"Drug interaction check for {', '.join(medications)}: "
                f"{len(alerts)} alert(s), {len(result.get('resolved_medications', []))} resolved, "
                f"{len(result.get('unresolved_medications', []))} unresolved."
            )
            return {"sources": sources, "summary": msg}
        except Exception as exc:
            return {"summary": f"Drug interaction check failed: {exc}", "sources": []}

    def _search_personal(self, query: str) -> Dict:
        if not query or not self.user:
            return {"summary": "No query or user.", "personal_matches": []}
        matches = self.memory.search(
            query=query, user=self.user, hypothetical_passage=self._current_hyde_passage
        )
        personal: List[Dict] = []
        for entry, score in matches[:4]:
            meta = entry.get("metadata", {})
            if meta.get("type") == "user_summary":
                personal.append(
                    {
                        "title": meta.get(
                            "title", meta.get("source", "Uploaded document")
                        ),
                        "source": meta.get("source", ""),
                        "snippet": build_excerpt(entry.get("text", "")),
                        "score": round(score, 3),
                    }
                )
        return {
            "personal_matches": personal,
            "summary": f"Found {len(personal)} matches in patient documents.",
        }

    def _search_trials(self, condition: str, location: str = "") -> Dict:
        if not condition:
            return {"summary": "No condition provided.", "trials": []}
        try:
            from backend.clinical_trials import TrialSearchProfile, find_matching_trials
            from backend.user_store import UserStore

            if self.user:
                profile = UserStore.get_user_profile(self.user)
                from backend.clinical_trials import build_trial_search_profile

                search_profile = build_trial_search_profile(
                    profile=profile,
                    memory=UserStore.get_longitudinal_memory(self.user),
                    symptom_logs=UserStore.get_symptom_logs(self.user, limit=None),
                    medications=UserStore.get_medications(self.user),
                    allergies=UserStore.get_allergies(self.user),
                    conditions=UserStore.get_conditions(self.user),
                    vitals=UserStore.get_vitals(self.user, limit=None),
                    triage_summaries=UserStore.get_triage_summaries(
                        self.user, limit=None
                    ),
                    document_summaries=UserStore.get_document_summaries(self.user),
                )
                search_profile.conditions = list(
                    dict.fromkeys(search_profile.conditions + [condition])
                )
                search_profile.raw_context += (
                    f"\nExplicit trial topic requested by patient: {condition}"
                )
            else:
                search_profile = TrialSearchProfile(
                    conditions=[condition],
                    symptoms=[],
                    medications=[],
                    age=None,
                    biological_sex="",
                    raw_context=f"Requested condition: {condition}",
                )
            results = find_matching_trials(
                search_profile,
                location_query=location,
                max_results=5,
                query_expander=self.query_expander,
            )
            return {
                "trials": results.get("trials", [])
                if isinstance(results, dict)
                else [],
                "summary": f"Found {len(results.get('trials', []) if isinstance(results, dict) else [])} trials for '{condition}' in {location}.",
            }
        except Exception as exc:
            return {"summary": f"Trial search failed: {exc}", "trials": []}


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


class ClinicalOrchestrator:
    """
    Central workflow engine. Called by RAGEngine._prepare_answer_bundle().
    Returns a bundle dict that is a superset of the original bundle structure.
    """

    def __init__(
        self,
        memory: "MemoryStore",
        pubmed: "PubMedCentralSearcher",
        official_guidance: "OfficialGuidanceEngine",
        llm: "LLMHelper",
        query_expander: "QueryExpander",
        moderation: "ModerationEnsemble",
    ) -> None:
        self.memory = memory
        self.pubmed = pubmed
        self.official_guidance = official_guidance
        self.llm = llm
        self.query_expander = query_expander
        self.moderation = moderation

        self.role_router = RoleRouter()
        self.intent_classifier = IntentRiskClassifier()
        self.decision_support = ClinicalDecisionSupportEngine()
        self.policy_engine = PolicyEngine()
        self.evidence_ranker = EvidenceRanker()

    def prepare_bundle(
        self,
        question: str,
        user: Optional[str],
        user_profile: dict,
        longitudinal_memory_summary: str,
        medications: Optional[List[Dict]] = None,
        triage_summaries: Optional[List[Dict]] = None,
        allergies: Optional[List[Dict]] = None,
        conditions: Optional[List[Dict]] = None,
        vitals: Optional[List[Dict]] = None,
        document_summaries: Optional[List[Dict]] = None,
        context_graph: Optional["ContextGraph"] = None,
        chat_history: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Full clinical orchestration pipeline.
        Returns a dict compatible with RAGEngine._finalize_answer_payload()
        plus new clinical governance and agentic metadata keys.
        """
        normalized_user = (user or "").strip().lower() or None
        current_location = current_location_from_profile(user_profile)

        # -- Step 1: Role resolution (instant) --------------------------------
        clinical_role = user_profile.get("clinical_role") or user_profile.get(
            "role", ""
        )
        role_config = self.role_router.resolve(clinical_role)
        task_mode = decide_task_mode(question, chat_history, role_config.role_key)

        # -- Step 2: Patient history context ----------------------------------
        patient_history: PatientHistoryContext = build_patient_history_context(
            longitudinal_memory=longitudinal_memory_summary,
            medications=medications or [],
            triage_summaries=triage_summaries or [],
            user_profile=user_profile,
            allergies=allergies or [],
            conditions=conditions or [],
            vitals=vitals or [],
        )

        # This is intentionally before intent classification and retrieval.  A
        # model must not be allowed to decide that a reused measurement name
        # means the most common specialty when the structured record already
        # establishes another meaning.
        clinical_context: ClinicalContextDecision = adjudicate_patient_context(
            question=question,
            conditions=conditions or [],
            medications=medications or [],
            vitals=vitals or [],
            allergies=allergies or [],
            triage_summaries=triage_summaries or [],
            document_summaries=document_summaries or [],
            longitudinal_memory=longitudinal_memory_summary,
            chat_summary="\n".join(
                f"{item.get('role', 'user')}: {item.get('content', '')}"
                for item in (chat_history or [])[-4:]
            ),
        )

        # Literal transformations operate on quoted/supplied text and do not
        # assert that embedded clinical content is an active event.
        if (
            not task_mode.literal_transformation
            and self.intent_classifier._crisis_prescreen(
                question, role_key=role_config.role_key
            )
        ):
            return self._build_crisis_bundle(question, normalized_user, role_config)

        # -- Step 4: Moderation -----------------------------------------------
        blocked, category, safe_msg, details = self.moderation.decide(
            question, role_key=role_config.role_key
        )
        if blocked:
            return self._build_moderation_bundle(
                question, normalized_user, safe_msg, category, details, role_config
            )

        if task_mode.is_transformation:
            return self._build_transformation_bundle(
                question=question,
                normalized_user=normalized_user,
                user_profile=user_profile,
                role_config=role_config,
                task_mode=task_mode,
                current_location=current_location,
            )

        # -- Step 5: Intent classification (needed for policy gate) -----------
        history_context = (
            patient_history.as_prompt_block() if not patient_history.is_empty() else ""
        )
        graph_hints: List[str] = (
            list(context_graph.search_hints) if context_graph else []
        )

        try:
            intent = self.intent_classifier.classify(
                question,
                user_profile,
                role_config.role_key,
                patient_history,
                chat_history,
            )
        except Exception as exc:
            print(f"[Orchestrator] Intent classification failed: {exc}")
            intent = IntentClassification()

        selected_skills = select_skills(intent.intent_category, question)

        # -- Step 6: Policy gate (8 hard safety gates) ------------------------
        clinical_decision = self.decision_support.assess(question, intent, role_config)
        intent = self.decision_support.apply_to_intent(intent, clinical_decision)

        policy_decision = self.policy_engine.gate(
            intent, role_config, question, patient_history
        )
        if (
            policy_decision.action == "escalate_only"
            and policy_decision.crisis_response
        ):
            return self._build_crisis_bundle(question, normalized_user, role_config)

        # -- Step 6b: Ambiguity gate -- ask before answering when a term has
        # multiple clinically distinct meanings and patient context doesn't
        # resolve which one applies. Only fires for routine/elevated risk and
        # when no other policy escalation is already in play, so a real safety
        # concern always wins over asking a question.
        if (
            intent.ambiguous_term_detected
            and intent.risk_level not in ("urgent", "crisis")
            and policy_decision.action == "allow"
        ):
            return self._build_clarification_bundle(
                question, normalized_user, role_config, intent
            )

        if (
            clinical_context.requires_clarification
            and intent.risk_level not in ("urgent", "crisis")
            and policy_decision.action == "allow"
        ):
            return self._build_context_clarification_bundle(
                question, normalized_user, role_config, clinical_context
            )

        # -- Step 7: Pathway context ------------------------------------------
        pathway_context = self._get_pathway_context(intent, role_config)

        # -- Step 8: Agentic retrieval loop -----------------------------------
        # Build a compact patient summary for the agent system prompt
        patient_summary = history_context or f"Role: {role_config.role_key}"
        patient_summary += "\n\n" + clinical_context.as_prompt_block()
        if graph_hints:
            patient_summary += "\nRelevant health terms: " + ", ".join(graph_hints[:6])
        retrieval_question = task_mode.retrieval_question(question)
        if clinical_context.query_terms:
            retrieval_question = (
                f"{question}\n\nConfirmed clinical search topic: "
                + "; ".join(clinical_context.query_terms)
            )

        med_names: List[str] = [
            m.get("name", "") for m in (medications or []) if m.get("name")
        ]

        agent_loop = AgenticRetrievalLoop(
            llm=self.llm,
            official_guidance=self.official_guidance,
            pubmed=self.pubmed,
            memory=self.memory,
            user=normalized_user,
            query_expander=self.query_expander,
        )

        try:
            agent_result = agent_loop.run(
                question=retrieval_question,
                patient_summary=patient_summary,
                role_key=role_config.role_key,
                pathway_hint=intent.pathway_hint or "general_triage",
                patient_medications=med_names,
                current_location=current_location,
                selected_skills=selected_skills,
            )
        except Exception as exc:
            print(f"[Orchestrator] Agentic loop failed, using fallback: {exc}")
            agent_result = {
                "collected_sources": [],
                "personal_context": [],
                "trial_results": [],
                "tool_calls_made": [],
            }

        collected_sources: List[Dict] = agent_result.get("collected_sources", [])
        personal_context: List[Dict] = agent_result.get("personal_context", [])
        tool_calls_made: List[Dict] = agent_result.get("tool_calls_made", [])
        hyde_passage: str = agent_result.get("hyde_passage", "")
        query_variants: List[str] = agent_result.get("query_variants", [])

        # Derive expanded_queries from what the agent actually searched
        expanded_queries: List[str] = list(
            dict.fromkeys(
                tc["args"].get("query", tc["args"].get("condition", ""))
                for tc in tool_calls_made
                if tc.get("tool")
                in ("search_nhs_guidance", "search_pubmed", "search_clinical_trials")
                and tc.get("args", {}).get("query")
                or tc.get("args", {}).get("condition")
            )
        ) or [question]

        # -- Step 9: Fallback retrieval if agent returned nothing -------------
        if not collected_sources:
            print(
                "[Orchestrator] Agent found no sources -- falling back to direct retrieval."
            )
            collected_sources, personal_context, expanded_queries = (
                self._run_fallback_retrieval(
                    retrieval_question,
                    history_context,
                    graph_hints,
                    pathway_context,
                    clinical_decision,
                    normalized_user,
                    hyde_passage,
                )
            )

        # -- Step 10: Deduplicate and rank evidence ---------------------------
        raw_sources = self._deduplicate_sources(collected_sources)
        raw_sources, _context_filtered = self._exclude_context_incompatible_sources(
            raw_sources, clinical_context
        )

        combined_sources, evidence_quality_report = (
            self.evidence_ranker.rank_and_tier_with_report(
                sources=raw_sources,
                question=question,
                role_config=role_config,
                intent=intent,
                memory_store=self.memory,
                top_k=6,
                patient_history=patient_history,
                context_graph=context_graph,
            )
        )

        # -- Step 11/11b: Evidence dossier (anti-hallucination layer) + --------
        # reconcile specialty-mismatch exclusions. The dossier's per-article LLM
        # extraction is the only stage that checks for cross-specialty term
        # mismatch (e.g. respiratory vs. urology "peak flow") -- evidence_ranker's
        # quality gate has no concept of this, so a source it accepted as
        # question_aligned/background_only can still be a confirmed mismatch.
        combined_sources, evidence_quality_report, evidence_dossier = (
            self._build_dossier_and_reconcile(
                combined_sources,
                evidence_quality_report,
                retrieval_question,
                user_profile,
                patient_history,
                medications,
                conditions,
            )
        )

        # -- Step 11c: Retry retrieval once if the gate rejected everything ----
        # A quality-gate rejection (sources existed but none passed) and a truly
        # empty retrieval (nothing was ever found) both currently converge here
        # with zero usable sources. Rather than let the answer LLM fall back to
        # unbacked general knowledge, try one more retrieval round seeded with a
        # genuinely different query before giving up -- reusing the next unused
        # HyDE variant (query_variants[0] was already tried above; [1] never was)
        # so this isn't just repeating the same search verbatim.
        retrieval_retry_attempted = False
        if not combined_sources:
            retrieval_retry_attempted = True
            retry_seed = (
                query_variants[1]
                if len(query_variants) > 1 and query_variants[1].strip()
                else None
            )
            if not retry_seed or retry_seed.strip().lower() == retrieval_question.strip().lower():
                retry_seed = None
            retry_collected, retry_personal, retry_queries_used = self._run_fallback_retrieval(
                retry_seed or retrieval_question,
                history_context,
                graph_hints,
                pathway_context,
                clinical_decision,
                normalized_user,
                hyde_passage,
            )
            retry_raw = self._deduplicate_sources(retry_collected)
            retry_raw, _ = self._exclude_context_incompatible_sources(
                retry_raw, clinical_context
            )
            combined_sources, evidence_quality_report = (
                self.evidence_ranker.rank_and_tier_with_report(
                    sources=retry_raw,
                    question=question,
                    role_config=role_config,
                    intent=intent,
                    memory_store=self.memory,
                    top_k=6,
                    patient_history=patient_history,
                    context_graph=context_graph,
                )
            )
            combined_sources, evidence_quality_report, evidence_dossier = (
                self._build_dossier_and_reconcile(
                    combined_sources,
                    evidence_quality_report,
                    retrieval_question,
                    user_profile,
                    patient_history,
                    medications,
                    conditions,
                )
            )
            personal_context = personal_context + retry_personal
            expanded_queries = retry_queries_used

        if retrieval_retry_attempted:
            # combined_sources is only still empty here if the retry also found
            # nothing usable -- the terminal refusal check right below is what
            # handles that case, never a general-knowledge answer.
            retrieval_mode = (
                "evidence_quality_gate_retry_recovered"
                if combined_sources
                else "no_evidence_after_retry"
            )
        else:
            # retrieval_retry_attempted is only False when combined_sources was
            # already non-empty after Step 11/11b (that's the retry's trigger
            # condition), so combined_sources is guaranteed non-empty here.
            retrieval_mode = (
                "agentic_multi_source" if tool_calls_made else "live_multi_source"
            )

        # -- Terminal refusal: never let the answer LLM run on zero evidence ---
        # If, even after the retry above, nothing passed the quality gate, the
        # system must not answer from unbacked general knowledge. Short-circuit
        # here exactly like _build_transformation_bundle/the crisis path already
        # do, before _build_role_context or the answer LLM is ever reached.
        if not combined_sources:
            return self._build_limited_bundle(
                question=question,
                normalized_user=normalized_user,
                personal_context=personal_context,
                retrieval_mode=retrieval_mode,
                expanded_queries=expanded_queries,
                role_config=role_config,
                intent=intent,
                policy_decision=policy_decision,
            )

        # -- Step 12: Build role-aware LLM context ----------------------------
        full_context = self._build_role_context(
            combined_sources=combined_sources,
            personal_context=personal_context,
            policy_decision=policy_decision,
            pathway_context=pathway_context,
            clinical_decision=clinical_decision,
            evidence_quality_report=evidence_quality_report,
            evidence_dossier=evidence_dossier,
            clinical_context=clinical_context,
        )
        completion_block = task_mode.completion_block(
            intent.intent_category, intent.vulnerable_flags
        )
        if completion_block:
            full_context = f"{full_context}\n\n{completion_block}".strip()

        return {
            "kind": "answer",
            # Backward-compatible keys
            "normalized_user": normalized_user,
            "user_profile": user_profile,
            "combined_sources": combined_sources,
            "personal_context": personal_context,
            "longitudinal_memory_summary": longitudinal_memory_summary,
            "expanded_queries": expanded_queries,
            "matches": [],
            "retrieval_mode": retrieval_mode,
            "full_context": full_context,
            "response_completion_guidance": completion_block,
            "evidence_quality_report": evidence_quality_report,
            # Clinical governance
            "role_config": role_config,
            "intent": intent,
            "policy_decision": policy_decision,
            "pathway_context": pathway_context,
            "clinical_decision": clinical_decision,
            "clinical_context": clinical_context,
            # Structured evidence (anti-hallucination layer)
            "evidence_dossier": evidence_dossier,
            # Agentic metadata (new)
            "agentic_tool_calls": tool_calls_made,
            "selected_skills": selected_skills,
            "current_location": current_location,
            "task_mode": task_mode,
        }

    # -- Bundle builders ------------------------------------------------------

    @staticmethod
    def _build_transformation_bundle(
        question: str,
        normalized_user: Optional[str],
        user_profile: dict,
        role_config: RoleConfig,
        task_mode: TaskModeDecision,
        current_location: str,
    ) -> Dict:
        intent = IntentClassification(
            intent_category="administrative",
            risk_level="routine",
            pathway_hint="general_triage",
            confidence=1.0,
        )
        return {
            "kind": "answer",
            "normalized_user": normalized_user,
            "user_profile": user_profile,
            "combined_sources": [],
            "personal_context": [],
            "longitudinal_memory_summary": "",
            "expanded_queries": [],
            "matches": [],
            "retrieval_mode": "controlled_transformation",
            "full_context": task_mode.prompt_block(),
            "response_completion_guidance": "",
            "evidence_quality_report": {},
            "role_config": role_config,
            "intent": intent,
            "policy_decision": PolicyDecision(),
            "pathway_context": None,
            "clinical_decision": None,
            "clinical_context": None,
            "evidence_dossier": None,
            "agentic_tool_calls": [],
            "selected_skills": ["response_validation"],
            "current_location": current_location,
            "task_mode": task_mode,
        }

    def _build_crisis_bundle(
        self,
        question: str,
        normalized_user: Optional[str],
        role_config: RoleConfig,
    ) -> Dict:
        crisis_response = build_crisis_response(role_config.role_key)
        return {
            "kind": "final",
            "payload": {
                "answer_markdown": crisis_response,
                "answer_text": crisis_response,
                "sources": [],
                "personal_context": [],
                "trace": {
                    "trace_id": "trace-crisis",
                    "created_at": _utc_now(),
                    "question": question,
                    "answer_preview": crisis_response[:280],
                    "sources": [],
                    "retrieval_mode": "crisis_escalation",
                    "role_key": role_config.role_key,
                    "intent_category": "crisis",
                    "risk_level": "crisis",
                    "escalation_triggered": True,
                    "crisis_detected": True,
                },
            },
        }

    def _build_moderation_bundle(
        self,
        question: str,
        normalized_user: Optional[str],
        safe_msg: str,
        category: str,
        details: Dict,
        role_config: RoleConfig,
    ) -> Dict:
        return {
            "kind": "final",
            "payload": {
                "answer_markdown": safe_msg,
                "answer_text": safe_msg,
                "sources": [],
                "personal_context": [],
                "trace": {
                    "trace_id": "trace-mod",
                    "created_at": _utc_now(),
                    "question": question,
                    "answer_preview": safe_msg[:280],
                    "sources": [],
                    "retrieval_mode": "moderation_block",
                    "moderation_category": category,
                    "moderation_details": details,
                    "role_key": role_config.role_key,
                },
            },
        }

    def _build_clarification_bundle(
        self,
        question: str,
        normalized_user: Optional[str],
        role_config: RoleConfig,
        intent: IntentClassification,
    ) -> Dict:
        answer = f"## Quick Question\n{intent.ambiguity_clarifying_question}"
        return {
            "kind": "final",
            "payload": {
                "answer_markdown": answer,
                "answer_text": answer,
                "sources": [],
                "personal_context": [],
                "follow_up_questions": intent.ambiguity_reply_options,
                "trace": {
                    "trace_id": "trace-clarify",
                    "created_at": _utc_now(),
                    "question": question,
                    "answer_preview": answer[:280],
                    "sources": [],
                    "retrieval_mode": "clarification_requested",
                    "role_key": role_config.role_key,
                    "intent_category": intent.intent_category,
                    "risk_level": intent.risk_level,
                    "escalation_triggered": False,
                    "ambiguous_term": intent.ambiguous_term,
                },
            },
        }

    def _build_context_clarification_bundle(
        self,
        question: str,
        normalized_user: Optional[str],
        role_config: RoleConfig,
        decision: ClinicalContextDecision,
    ) -> Dict:
        answer = f"## Quick check before I answer\n{decision.clarifying_question}"
        return {
            "kind": "final",
            "payload": {
                "answer_markdown": answer,
                "answer_text": answer,
                "sources": [],
                "personal_context": [],
                "follow_up_questions": decision.clarification_options,
                "trace": {
                    "trace_id": "trace-context-clarify",
                    "created_at": _utc_now(),
                    "question": question,
                    "answer_preview": answer[:280],
                    "sources": [],
                    "retrieval_mode": "clinical_context_clarification",
                    "role_key": role_config.role_key,
                    "clinical_context": decision.as_dict(),
                    "escalation_triggered": False,
                },
            },
        }

    def _build_limited_bundle(
        self,
        question: str,
        normalized_user: Optional[str],
        personal_context: List[Dict],
        retrieval_mode: str,
        expanded_queries: List[str],
        role_config: RoleConfig,
        intent: IntentClassification,
        policy_decision: PolicyDecision,
    ) -> Dict:
        limited_answer = self._build_limited_evidence_response(
            personal_context, role_config
        )
        return {
            "kind": "final",
            "payload": {
                "answer_markdown": limited_answer,
                "answer_text": limited_answer,
                "sources": [],
                "personal_context": personal_context,
                "trace": {
                    "trace_id": "trace-limited",
                    "created_at": _utc_now(),
                    "question": question,
                    "answer_preview": limited_answer[:280],
                    "sources": [],
                    "retrieval_mode": retrieval_mode,
                    "expanded_queries": expanded_queries,
                    "role_key": role_config.role_key,
                    "intent_category": intent.intent_category,
                    "risk_level": intent.risk_level,
                    "escalation_triggered": policy_decision.action != "allow",
                    "policy_gates_applied": policy_decision.gates_as_dicts(),
                },
            },
        }

    # -- Context builders -----------------------------------------------------

    @staticmethod
    def _exclude_mismatched_sources(
        combined_sources: List[Dict],
        evidence_quality_report: Dict,
        excluded_source_ids: List[str],
    ) -> Tuple[List[Dict], Dict]:
        """
        Removes sources the evidence dossier confirmed to concern a different
        specialty/meaning of an ambiguous term from combined_sources (so they can't
        appear in the Sources panel or be cited), and moves the matching entries in
        evidence_quality_report from "accepted" into "excluded" so the quality-gate
        text shown to the answer LLM stops describing them as usable general context.
        """
        excluded_ids = set(excluded_source_ids)
        kept, dropped = [], []
        for source in combined_sources:
            (dropped if source.get("source_id") in excluded_ids else kept).append(
                source
            )
        if not dropped:
            return combined_sources, evidence_quality_report

        report = dict(evidence_quality_report)
        status_counts = dict(report.get("status_counts") or {})
        newly_excluded = []
        for source in dropped:
            status = source.get("evidence_quality_status", "unknown")
            if status_counts.get(status):
                status_counts[status] -= 1
                if status_counts[status] <= 0:
                    del status_counts[status]
            newly_excluded.append(
                {
                    "title": source.get("title", "Retrieved source"),
                    "source_type": source.get("source_type", ""),
                    "provider": source.get("provider", ""),
                    "query": source.get("query", ""),
                    "quality_score": source.get("evidence_quality_score", 0),
                    "reasons": [
                        "Confirmed by evidence extraction to concern a different "
                        "specialty/measurement meaning than this patient's profile."
                    ],
                    "mismatch_flags": ["specialty_mismatch"],
                }
            )

        report["status_counts"] = status_counts
        report["accepted_source_count"] = len(kept)
        report["excluded_source_count"] = report.get("excluded_source_count", 0) + len(
            newly_excluded
        )
        report["excluded_sources"] = (
            newly_excluded + list(report.get("excluded_sources") or [])
        )[:5]

        patient_aligned = status_counts.get("patient_aligned", 0)
        if kept and patient_aligned:
            report["overall_status"] = "patient_aligned_evidence_available"
        elif kept:
            report["overall_status"] = "question_aligned_only"
        elif report["excluded_source_count"]:
            report["overall_status"] = "no_sources_passed_quality_gate"
        else:
            report["overall_status"] = "no_live_evidence"

        return kept, report

    def _build_role_context(
        self,
        combined_sources: List[Dict],
        personal_context: List[Dict],
        policy_decision: PolicyDecision,
        pathway_context,
        clinical_decision: ClinicalDecision,
        evidence_quality_report: Optional[Dict] = None,
        evidence_dossier=None,
        clinical_context: Optional[ClinicalContextDecision] = None,
    ) -> str:
        parts = []

        if clinical_context and clinical_context.status != "insufficient":
            parts.append(clinical_context.as_prompt_block())

        if personal_context:
            personal_lines = "\n".join(
                f"- {item['title']}: {item['snippet']}" for item in personal_context
            )
            parts.append(f"Personal context:\n{personal_lines}")

        if clinical_decision:
            decision_lines = [
                f"- Pathway: {clinical_decision.pathway_label}",
                f"- Urgency: {clinical_decision.urgency_level}",
                f"- Primary action: {clinical_decision.next_step}",
                f"- Summary: {clinical_decision.summary}",
            ]
            decision_lines.extend(
                f"- Immediate action: {item}"
                for item in clinical_decision.immediate_actions[:4]
            )
            decision_lines.extend(
                f"- Monitor now: {item}"
                for item in clinical_decision.monitoring_priorities[:3]
            )
            if clinical_decision.triggered_rules:
                decision_lines.extend(
                    f"- Rule hit: {item.finding}"
                    for item in clinical_decision.triggered_rules
                )
            parts.append(
                "Deterministic clinical decision support output (must not be contradicted):\n"
                + "\n".join(decision_lines)
            )

        if policy_decision.context_notes:
            notes = "\n".join(policy_decision.context_notes)
            parts.append(f"Clinical policy notes (must be followed):\n{notes}")

        if pathway_context and pathway_context.safety_rules:
            rules = "\n".join(f"- {r}" for r in pathway_context.safety_rules)
            parts.append(f"Pathway safety rules:\n{rules}")

        if evidence_quality_report:
            quality_lines = [
                f"- Overall status: {evidence_quality_report.get('overall_status', 'unknown')}",
                f"- Accepted sources: {evidence_quality_report.get('accepted_source_count', 0)}",
                f"- Excluded sources: {evidence_quality_report.get('excluded_source_count', 0)}",
            ]
            profile_facts = evidence_quality_report.get("profile_facts_checked") or []
            if profile_facts:
                quality_lines.append(
                    "- Profile facts checked: "
                    + "; ".join(str(f) for f in profile_facts[:8])
                )
            status_counts = evidence_quality_report.get("status_counts") or {}
            if status_counts:
                counts_text = ", ".join(f"{k}={v}" for k, v in status_counts.items())
                quality_lines.append(f"- Source usability counts: {counts_text}")
            for item in (evidence_quality_report.get("excluded_sources") or [])[:3]:
                reasons = "; ".join(str(r) for r in item.get("reasons", [])[:2])
                quality_lines.append(
                    f"- Filtered out: {item.get('title', 'Source')} ({reasons})"
                )
            quality_lines.append(
                "- Binding rule: use patient_aligned sources for patient-specific guidance; "
                "use question_aligned or background_only sources only for general context."
            )
            parts.append(
                "Private source-use instructions (never mention these labels or this filtering process):\n"
                + "\n".join(quality_lines)
            )

        if evidence_dossier and evidence_dossier.articles:
            parts.append(
                "Structured patient-aligned evidence dossier "
                "(extracted facts matched to this patient -- do not cite facts not present here):\n"
                + evidence_dossier.to_prompt_context()
            )
        elif combined_sources:
            evidence_parts = []
            for source in combined_sources:
                tier = source.get("evidence_tier", 3)
                tier_label = source.get("tier_label", f"Tier {tier}")
                snippet = source.get("detail_snippet") or source.get("snippet", "")
                quality_status = source.get(
                    "evidence_quality_status", "question_aligned"
                )
                use_label = (
                    "patient-specific guidance"
                    if source.get("usable_for_patient_specific_guidance")
                    else "general/background context"
                )
                quality_notes = "; ".join(
                    str(r) for r in source.get("evidence_quality_reasons", [])[:2]
                )
                evidence_parts.append(
                    f"[{tier_label}] {source.get('title', 'Source')} "
                    f"(quality: {quality_status}; use: {use_label}): {snippet}"
                    + (f"\nQuality notes: {quality_notes}" if quality_notes else "")
                )
            parts.append(
                "Biomedical evidence (tiered by source authority):\n"
                + "\n\n".join(evidence_parts)
            )
        # No `elif` for the empty-sources case: the caller (ClinicalOrchestrator's
        # main retrieval method) never reaches this method with combined_sources
        # empty -- it short-circuits to _build_limited_bundle's graceful refusal
        # before _build_role_context is called at all, precisely so the answer
        # LLM never gets a "just answer from general knowledge" instruction.

        return "\n\n".join(parts)

    @staticmethod
    def _exclude_context_incompatible_sources(
        sources: List[Dict], decision: ClinicalContextDecision
    ) -> Tuple[List[Dict], List[Dict]]:
        """Hard-filter evidence that belongs to a different specialty."""
        if not decision.domain:
            return sources, []
        kept: List[Dict] = []
        dropped: List[Dict] = []
        for source in sources:
            title = str(source.get("title", ""))
            content = str(source.get("detail_snippet") or source.get("snippet") or "")
            if source_matches_context(title, content, decision):
                kept.append(source)
            else:
                dropped.append(source)
        return kept, dropped

    # -- Pathway routing ------------------------------------------------------

    def _get_pathway_context(
        self, intent: IntentClassification, role_config: RoleConfig
    ):
        hint = intent.pathway_hint or "general_triage"
        try:
            if hint == "maternity":
                from backend.pathways.maternity import get_pathway_context
            elif hint == "msk":
                from backend.pathways.msk import get_pathway_context
            elif hint == "medications":
                from backend.pathways.medications import get_pathway_context
            elif hint == "chronic_conditions":
                from backend.pathways.chronic_conditions import get_pathway_context
            else:
                from backend.pathways.general_triage import get_pathway_context
            return get_pathway_context(intent, role_config)
        except Exception as exc:
            print(f"[Orchestrator] Pathway load failed ({hint}): {exc}")
            from backend.pathways.general_triage import get_pathway_context

            return get_pathway_context(intent, role_config)

    # -- Fallback query helpers -----------------------------------------------

    def _build_search_queries(
        self,
        question: str,
        patient_history_context: str = "",
        graph_hints: Optional[List[str]] = None,
    ) -> List[str]:
        queries = [question]
        try:
            if patient_history_context:
                queries.extend(
                    self.query_expander.expand_with_patient_context(
                        question, patient_history_context
                    )
                )
            else:
                queries.extend(self.query_expander.expand(question))
        except Exception as exc:
            print(f"[Orchestrator] Query expansion failed: {exc}")
        for hint in graph_hints or []:
            if hint and hint not in queries:
                queries.append(hint)
        return list(dict.fromkeys(q for q in queries if q))[:5]

    def _augment_queries_with_pathway(
        self,
        queries: List[str],
        pathway_context,
        clinical_decision: Optional[ClinicalDecision] = None,
    ) -> List[str]:
        augmented = list(queries)
        if pathway_context and pathway_context.additional_search_terms:
            for term in pathway_context.additional_search_terms[:2]:
                combined = f"{queries[0]} {term}"
                if combined not in augmented:
                    augmented.append(combined)
        if clinical_decision:
            for term in clinical_decision.search_terms[:2]:
                if term not in augmented:
                    augmented.append(term)
        return augmented[:5]

    def _run_fallback_retrieval(
        self,
        seed_question: str,
        history_context: str,
        graph_hints: Optional[List[str]],
        pathway_context,
        clinical_decision: Optional[ClinicalDecision],
        normalized_user: Optional[str],
        hyde_passage: str,
    ) -> Tuple[List[Dict], List[Dict], List[str]]:
        """
        Direct (non-agentic) NHS + PubMed retrieval, seeded with `seed_question`.
        Used both as Step 9's fallback (when the agentic loop found nothing) and
        as Step 11c's retry (when the quality gate rejected everything the agentic
        loop found) -- passing a different `seed_question` the second time is what
        makes the retry a genuine second attempt rather than repeating the same
        search verbatim.
        """
        fallback_queries = self._build_search_queries(
            seed_question, history_context, graph_hints
        )
        search_queries = self._augment_queries_with_pathway(
            fallback_queries, pathway_context, clinical_decision
        )
        collected_sources: List[Dict] = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            preferred = list(dict.fromkeys(pathway_context.preferred_sources or []))
            official_future = executor.submit(
                self.official_guidance.search, search_queries, 1, preferred or None
            )
            pubmed_future = executor.submit(
                self._retrieve_pubmed_for_queries, search_queries, normalized_user
            )
            try:
                collected_sources = official_future.result()
            except Exception as exc:
                print(f"[Orchestrator] Fallback NHS search failed: {exc}")
            try:
                pubmed_future.result()
            except Exception as exc:
                print(f"[Orchestrator] Fallback PubMed search failed: {exc}")

        matches = self.memory.search(
            query=seed_question,
            user=normalized_user,
            hypothetical_passage=hyde_passage,
        )
        personal_context, pubmed_matches = self._split_matches(matches)
        collected_sources.extend(self._build_source_briefings(pubmed_matches))
        return collected_sources, personal_context, fallback_queries

    def _build_dossier_and_reconcile(
        self,
        combined_sources: List[Dict],
        evidence_quality_report: Dict,
        retrieval_question: str,
        user_profile: dict,
        patient_history,
        medications: Optional[List[Dict]],
        conditions: Optional[List[Dict]],
    ) -> Tuple[List[Dict], Dict, Optional["ExtractedEvidenceDossier"]]:
        """
        Runs the evidence dossier (anti-hallucination, per-article extraction) and
        reconciles any specialty-mismatch exclusions it finds back into
        combined_sources/evidence_quality_report. Shared by the initial Step 11/11b
        pass and Step 11c's retry pass so the reconciliation logic isn't duplicated.
        """
        evidence_dossier = None
        if not combined_sources:
            return combined_sources, evidence_quality_report, evidence_dossier

        try:
            from backend.evidence_extractor import build_evidence_dossier

            evidence_dossier = build_evidence_dossier(
                llm=self.llm,
                sources=combined_sources,
                question=retrieval_question,
                user_profile=user_profile,
                patient_history_ctx=patient_history,
                medications=medications or [],
                conditions=conditions or [],
            )
        except Exception as exc:
            print(f"[Orchestrator] Evidence dossier build failed (non-fatal): {exc}")

        if evidence_dossier and evidence_dossier.excluded_source_ids:
            combined_sources, evidence_quality_report = self._exclude_mismatched_sources(
                combined_sources,
                evidence_quality_report,
                evidence_dossier.excluded_source_ids,
            )
        return combined_sources, evidence_quality_report, evidence_dossier

    # -- Source processing helpers --------------------------------------------

    @staticmethod
    def _deduplicate_sources(sources: List[Dict]) -> List[Dict]:
        seen: set = set()
        deduped: List[Dict] = []
        for source in sources:
            key = (
                source.get("url")
                or source.get("pmcid")
                or f"{source.get('title', '')}::{source.get('section', '')}"
            )
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            deduped.append(dict(source))
        for idx, source in enumerate(deduped, start=1):
            source["source_id"] = f"S{idx}"
        return deduped

    def _retrieve_pubmed_for_queries(
        self, queries: List[str], user: Optional[str]
    ) -> None:
        """Fallback: fetch PubMed articles and add to memory store."""
        pending_entries = []
        article_batches = []

        with ThreadPoolExecutor(max_workers=min(3, max(1, len(queries)))) as executor:
            query_futures = {
                executor.submit(self.pubmed.search_article_records, query, 2): query
                for query in queries
            }
            for future, query in query_futures.items():
                try:
                    article_batches.append((query, future.result()))
                except Exception as exc:
                    print(f"[Orchestrator] PubMed search failed for '{query}': {exc}")

        article_records = [
            (query, record) for query, records in article_batches for record in records
        ]

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
            for future, (query, record) in section_futures.items():
                try:
                    sections = future.result()
                except Exception as exc:
                    print(f"[Orchestrator] PubMed section fetch failed: {exc}")
                    sections = {}

                best_name, best_text = self._select_best_pubmed_section(sections)
                if best_text:
                    entry_key = f"{user or 'global'}:pmc:{record['pmcid']}:{best_name}"
                    pending_entries.append(
                        {
                            "text": best_text,
                            "metadata": {
                                "type": "pubmed",
                                "source_type": "pubmed_literature",
                                "pmcid": record["pmcid"],
                                "section": best_name,
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

                abstract = record.get("abstract", "")
                if abstract:
                    entry_key = f"{user or 'global'}:pmc:{record['pmcid']}:abstract"
                    pending_entries.append(
                        {
                            "text": abstract,
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
        seen: set = set()
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
    def _select_best_pubmed_section(sections: Dict[str, str]) -> Tuple[str, str]:
        for key in ("discussion", "conclusion", "introduction"):
            text = (sections.get(key) or "").strip()
            if text:
                return key, text
        return "", ""

    @staticmethod
    def _build_limited_evidence_response(
        personal_context: List[Dict], role_config: RoleConfig
    ) -> str:
        personal_note = ""
        if personal_context:
            personal_note = "\n\n## Available Personal Context\n" + "\n".join(
                f"- {item['title']}: {item['snippet']}" for item in personal_context
            )
        if role_config.role_key in ("doctor", "nurse", "midwife", "physiotherapist", "healthcare_professional"):
            return (
                "## Evidence Retrieval\n"
                "Insufficient live evidence was retrieved for this query. "
                "Please consult current local guidelines, BNF, or NICE CKS directly.\n\n"
                "## Recommended Action\n"
                "Use the relevant local pathway or guideline now, or rephrase the query "
                "with the exact condition, drug, population, or decision point you need."
                + personal_note
            )
        return (
            "## Working Impression\n"
            "I could not retrieve enough reliable live evidence for this question right now.\n\n"
            "## What To Do Now\n"
            "Please narrow the question to a specific symptom, condition, treatment, or population, "
            "or contact a clinician directly if this affects a decision that needs to be made now."
            + personal_note
        )


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
