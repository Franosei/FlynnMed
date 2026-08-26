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
     - search_nhs_guidance    UK and US Tier 1 government guidance
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
import re
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
from backend.conversation_context import render_verbatim
from backend.context_graph import format_relationships_for_user
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


_UNAVAILABLE_DETAIL_PATTERN = re.compile(
    r"\b(?:i\s+(?:do\s+not|don't|dont|cannot|can't|cant)\s+(?:know|remember)|"
    r"i\s+have\s+no\s+(?:idea|further\s+information)|not\s+sure|unknown)\b",
    re.IGNORECASE,
)
_BREATHING_SAFETY_PATTERN = re.compile(
    r"\b(?:breath(?:e|ing|lessness)?|shortness\s+of\s+breath|chok(?:e|es|ed|ing)|"
    r"gasp(?:s|ing)?|respir(?:er|ation)|[ée]touff(?:e|es|ement)|suffocat(?:e|ing|ion))\b",
    re.IGNORECASE,
)
_DISTRESS_SAFETY_PATTERN = re.compile(
    r"\b(?:anxious|anxiety|panic|losing\s+sleep|cannot\s+sleep|can't\s+sleep|"
    r"low\s+mood|depress(?:ed|ion)|postpartum|postnatal|self[- ]harm|suicid(?:e|al))\b",
    re.IGNORECASE,
)

_GUIDELINE_AUTHORITY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "ACOG": ("acog", "american college of obstetricians and gynecologists"),
    "ADA": ("ada", "american diabetes association"),
    "AHA": ("aha", "american heart association"),
    "ASCO": ("asco", "american society of clinical oncology"),
    "NCCN": ("nccn", "national comprehensive cancer network"),
    "NICE": ("nice", "national institute for health and care excellence"),
    "NHS": ("nhs", "national health service"),
    "WHO": ("who", "world health organization", "world health organisation"),
    "CDC": ("cdc", "centers for disease control and prevention"),
    "USPSTF": ("uspstf", "us preventive services task force"),
    "RCOG": ("rcog", "royal college of obstetricians and gynaecologists"),
    "ESC": ("esc", "european society of cardiology"),
    "IDSA": ("idsa", "infectious diseases society of america"),
    "AAP": ("aap", "american academy of pediatrics"),
    "ILCA": ("ilca", "international lactation consultant association"),
    "AACE": (
        "aace",
        "american association of clinical endocrinologists",
        "american association of clinical endocrinology",
    ),
    "SHEA": ("shea", "society for healthcare epidemiology of america"),
    "AAO": ("aao", "american academy of ophthalmology"),
    "ASA": ("asa", "american society of anesthesiologists"),
    "SIGN": ("sign", "scottish intercollegiate guidelines network"),
    "MHRA": ("mhra", "medicines and healthcare products regulatory agency"),
    "BNF": ("bnf", "british national formulary"),
}
# Alias tokens that collide with ordinary English words -- matched case-
# sensitively against the authority's own uppercase key so "a nice, simple
# way" or "warning signs" don't falsely register as a guideline request.
_CASE_SENSITIVE_AUTHORITY_ALIASES = {"who", "nice", "sign"}


# ---------------------------------------------------------------------------
# Agentic retrieval loop
# ---------------------------------------------------------------------------


class AgenticRetrievalLoop:
    """
    LLM-driven tool-calling retrieval agent.

    Given the clinical question and patient context the model decides which
    sources to fetch -- official guidance, PubMed, drug interactions, personal
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
                    "Search trusted UK and US government guidance, including NHS, NICE, "
                    "MedlinePlus, CDC, MyHealthfinder and VA/DoD sources. Use it for clinical "
                    "guidelines, treatment recommendations and patient safety advice. "
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
        question_medications: Optional[List[str]] = None,
        current_location: str = "",
        selected_skills: Optional[List[str]] = None,
        max_iterations: int = 5,
    ) -> Dict:
        """
        Run the agentic retrieval loop.

        Returns a dict with:
          collected_sources   list of official-guidance, PubMed and openFDA sources
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
        medicines_to_retrieve = list(
            dict.fromkeys(
                name.strip()
                for name in (question_medications or [])
                if name and name.strip()
            )
        )
        if medicines_to_retrieve:
            med_hint += (
                "\nExact medicine(s) named in the question: "
                + ", ".join(medicines_to_retrieve[:4])
                + ". Retrieve the exact medicine page, indications, contraindications, "
                "allergy warnings, and patient information; do not search the user's whole "
                "sentence as though it were a guideline title."
            )
        if patient_medications:
            med_hint = (
                med_hint
                + f"\nPatient's current medications: {', '.join(patient_medications[:8])}. "
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

        # Baseline evidence retrieval is mandatory for every question. Official
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

        evidence_search_question = question
        if medicines_to_retrieve:
            evidence_search_question = (
                f"{', '.join(medicines_to_retrieve[:4])} {question} practical advice "
                "food timing daily activities alcohol rest side effects contraindications "
                "allergy warnings official medicine guidance"
            )

        mandatory_tools = (
            ("search_nhs_guidance", self._search_nhs),
            ("search_pubmed", self._search_pubmed),
        )
        hyde_result = None
        with ThreadPoolExecutor(max_workers=3) as executor:
            search_futures = {
                executor.submit(search_fn, evidence_search_question): tool_name
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
                print(
                    f"[AgenticLoop] mandatory {tool_name}({evidence_search_question!r}) "
                    f"-> {len(sources)} source(s)"
                )
                collected_sources.extend(sources)
                tool_calls_made.append(
                    {
                        "tool": tool_name,
                        "args": {"query": evidence_search_question},
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

        # Exact medicine pages are more reliable than asking a general search
        # endpoint to interpret a full chat sentence. This remains additive:
        # failure simply falls back to the searches above.
        if medicines_to_retrieve and hasattr(self.official_guidance, "search_medicine"):
            for medicine_name in medicines_to_retrieve[:3]:
                try:
                    medicine_sources = self.official_guidance.search_medicine(
                        medicine_name, limit=4
                    )
                except Exception as exc:
                    print(
                        f"[AgenticLoop] exact medicine lookup failed for "
                        f"{medicine_name!r}: {exc}"
                    )
                    medicine_sources = []
                collected_sources.extend(medicine_sources)
                tool_calls_made.append(
                    {
                        "tool": "search_medicine_guidance",
                        "args": {"medicine": medicine_name},
                        "iteration": 0,
                        "mandatory": True,
                    }
                )

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
            "summary": f"Found {len(sources)} official guidance sources for '{query}'.",
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
                        "licence": record.get("licence", ""),
                        "licence_status": record.get("licence_status", ""),
                        "licence_url": record.get("licence_url", ""),
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
                            "licence": record.get("licence", ""),
                            "licence_status": record.get("licence_status", ""),
                            "licence_url": record.get("licence_url", ""),
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
                        "licence": record.get("licence", ""),
                        "licence_status": record.get("licence_status", ""),
                        "licence_url": record.get("licence_url", ""),
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
                            "licence": record.get("licence", ""),
                            "licence_status": record.get("licence_status", ""),
                            "licence_url": record.get("licence_url", ""),
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
            from backend.medication_checker import MedicationInteractionChecker

            checker = MedicationInteractionChecker()
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
                            "authority": "US Food and Drug Administration",
                            "jurisdiction": "US",
                            "licence_status": "public_domain_us",
                            "licence_url": (
                                "https://www.fda.gov/about-fda/about-website/website-policies"
                            ),
                            "attribution": "US Food and Drug Administration",
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
        previous_five_chat: Optional[List[Dict]] = None,
        conversation_summary: str = "",
        patient_statement_summary: str = "",
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
        verbatim_recent = list(previous_five_chat or (chat_history or [])[-5:])

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
            chat_summary=(
                "Whole earlier conversation summary:\n"
                + (conversation_summary or "No earlier conversation.")
                + "\n\nPrevious five messages verbatim:\n"
                + render_verbatim(verbatim_recent)
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

        if task_mode.mode == "chart_lookup":
            return self._build_chart_lookup_bundle(
                question=question,
                normalized_user=normalized_user,
                user_profile=user_profile,
                role_config=role_config,
                task_mode=task_mode,
                current_location=current_location,
                patient_history=patient_history,
                longitudinal_memory_summary=longitudinal_memory_summary,
                context_graph=context_graph,
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
                verbatim_recent,
                conversation_summary,
            )
        except Exception as exc:
            print(f"[Orchestrator] Intent classification failed: {exc}")
            intent = IntentClassification()

        selected_skills = select_skills(intent.intent_category, question)

        # Carry the exact medicine name through retrieval and fallback handling.
        # The user's full sentence is usually a poor medicine-site search query.
        question_medications: List[str] = []
        medicine_reference = bool(
            re.search(
                r"\b(?:the\s+)?(?:medication|medicine|antibiotics?|prescription|"
                r"dose|capsules?|tablets?)\b",
                question,
                re.IGNORECASE,
            )
        )
        if (
            intent.intent_category == "medication_query"
            or "medication_safety" in selected_skills
            or (
                medicine_reference
                and bool((chat_history or []) or patient_history.known_medications)
            )
        ):
            medication_extraction_text = "\n".join(
                [
                    str(item.get("content", "")).strip()
                    for item in verbatim_recent
                    if item.get("role") == "user"
                    and str(item.get("content", "")).strip()
                ]
                + (
                    ["Summary of all prior patient statements:\n" + patient_statement_summary]
                    if patient_statement_summary
                    and patient_statement_summary != "No earlier conversation."
                    else []
                )
                + (
                    ["Saved current medicines: " + ", ".join(patient_history.known_medications)]
                    if patient_history.known_medications
                    else []
                )
                + [question]
            )
            try:
                question_medications = self.llm.extract_medication_mentions(
                    medication_extraction_text
                )
            except Exception as exc:
                print(f"[Orchestrator] Medication-name extraction failed: {exc}")
            question_medications = list(
                dict.fromkeys(
                    str(name).strip()
                    for name in question_medications
                    if str(name).strip()
                )
            )[:6]
            if (
                not question_medications
                and medicine_reference
                and len(patient_history.known_medications) == 1
            ):
                question_medications = patient_history.known_medications[:1]
            # A short continuation such as "When taking the medication..." can
            # be misclassified as general information even though the medicine
            # is explicit in the preceding user turn.  Once we have resolved an
            # exact medicine, keep the whole request on the medication pathway.
            if question_medications and medicine_reference and intent.intent_category in {
                "general_info",
                "symptom_triage",
                "maternity",
                "msk",
                "chronic_condition",
            }:
                intent.intent_category = "medication_query"
                intent.pathway_hint = "medications"
                selected_skills = select_skills(intent.intent_category, question)

        # -- Step 6: Policy gate (8 hard safety gates) ------------------------
        # Deterministic pathways may only assert findings reported by the user.
        # Exclude assistant turns so an earlier generated warning cannot become
        # clinical evidence on the next message.
        user_case_text = "\n".join(
            [
                str(item.get("content", "")).strip()
                for item in verbatim_recent
                if item.get("role") == "user"
                and str(item.get("content", "")).strip()
            ]
            + [question]
        )
        clinical_decision = self.decision_support.assess(
            question,
            intent,
            role_config,
            case_text=user_case_text,
        )
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
            and not self._clarification_detail_unavailable(question)
            and not self._requests_multiple_interpretations(question)
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

        # A patient-specific answer is only useful when the facts that change
        # the decision are known. Ask those exact questions before retrieval
        # and report generation; do not fill the gap with a generic answer.
        if (
            intent.clarification_required
            and intent.clarifying_questions
            and intent.risk_level == "routine"
            and intent.intent_category not in {
                "general_info", "administrative", "mental_health"
            }
            and policy_decision.action == "allow"
        ):
            return self._build_information_clarification_bundle(
                question=question,
                normalized_user=normalized_user,
                role_config=role_config,
                intent=intent,
                patient_history=patient_history,
                question_medications=question_medications,
                context_graph=context_graph,
            )

        # -- Step 7: Pathway context ------------------------------------------
        pathway_context = self._get_pathway_context(intent, role_config)

        # -- Step 8: Agentic retrieval loop -----------------------------------
        # Build a compact patient summary for the agent system prompt.
        # Recent user-supplied conversation goes first:
        # query_expander.expand_with_hyde() only
        # reads the first 500 chars of this string, and a bare follow-up like
        # "can the antibiotics clear this lump?" carries no searchable clinical
        # terms of its own -- the drug/condition names live in the prior turns,
        # so without this the mandatory evidence search runs contextless and
        # can come back empty even though the conversation already established
        # what's being asked about.
        # Assistant prose is not evidence that the patient confirmed a
        # diagnosis or treatment indication, so keep it out of retrieval.
        recent_turns = [
            item
            for item in verbatim_recent
            if item.get("role") == "user"
        ]
        recent_conversation = " | ".join(
            f"{item.get('role', 'user')}: {str(item.get('content', ''))[:150]}"
            for item in recent_turns
            if item.get("content")
        )[:500]

        patient_summary = history_context or f"Role: {role_config.role_key}"
        if patient_statement_summary and patient_statement_summary != "No earlier conversation.":
            patient_summary = (
                "Summary of all prior patient-authored chat (user statements only):\n"
                + patient_statement_summary
                + "\n\n"
                + patient_summary
            )
        if recent_conversation:
            patient_summary = (
                "Recent conversation (resolve references like 'it'/'this'/'the antibiotics' "
                f"against what was actually discussed here): {recent_conversation}\n\n"
                + patient_summary
            )
        patient_summary += "\n\n" + clinical_context.as_prompt_block()
        relationship_block = (
            context_graph.relationship_prompt_block() if context_graph else ""
        )
        if relationship_block:
            patient_summary += "\n\n" + relationship_block
        if graph_hints:
            patient_summary += "\nRelevant health terms: " + ", ".join(graph_hints[:6])
        retrieval_question = task_mode.retrieval_question(question)
        if question_medications:
            recent_user_text = " ".join(
                str(item.get("content") or "")
                for item in verbatim_recent
                if item.get("role") == "user"
            ).lower() + " " + patient_statement_summary.lower()
            known_condition_names = [
                re.sub(r"\s+\([^)]*\)$", "", item).strip()
                for item in patient_history.known_conditions
                if str(item).strip()
            ]
            relevant_conditions = [
                item for item in known_condition_names
                if item.lower() in recent_user_text
            ]
            if not relevant_conditions and len(known_condition_names) == 1:
                relevant_conditions = known_condition_names
            retrieval_question = (
                f"{question}\n\nExact medicine in this conversation: "
                + ", ".join(question_medications)
                + ". Answer the user's current practical question about this exact medicine."
            )
            if relevant_conditions:
                retrieval_question += (
                    "\nEstablished condition from the patient's own record or prior statements: "
                    + "; ".join(relevant_conditions[:3])
                    + ". Treat the current message as a follow-up in that context."
                )
        if clinical_context.query_terms:
            retrieval_question = (
                f"{retrieval_question}\n\nConfirmed clinical search topic: "
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
                question_medications=question_medications,
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
                question=retrieval_question,
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
                    question=retrieval_question,
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

        requested_guideline_authorities = self._requested_guideline_authorities(
            question
        )
        missing_requested_guideline_authorities = (
            self._missing_requested_guideline_authorities(
                requested_guideline_authorities, combined_sources
            )
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
                patient_history=patient_history,
                question_medications=question_medications,
                context_graph=context_graph,
                missing_requested_guideline_authorities=(
                    missing_requested_guideline_authorities
                ),
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
            context_graph=context_graph,
        )
        completion_block = task_mode.completion_block(
            intent.intent_category, intent.vulnerable_flags
        )
        if completion_block:
            full_context = f"{full_context}\n\n{completion_block}".strip()
        if missing_requested_guideline_authorities:
            missing_names = ", ".join(missing_requested_guideline_authorities)
            authority_instruction = (
                "REQUESTED GUIDANCE SOURCE CHECK: No retrieved source from "
                f"{missing_names} was verified. State this explicitly. Do not attribute "
                "the answer to those organisations and do not present another source as "
                "a substitute for their guidance."
            )
            full_context = f"{full_context}\n\n{authority_instruction}".strip()
            completion_block = (
                f"{completion_block}\n\n{authority_instruction}".strip()
            )

        return {
            "kind": "answer",
            # Backward-compatible keys
            "normalized_user": normalized_user,
            "user_profile": user_profile,
            "combined_sources": combined_sources,
            "personal_context": personal_context,
            "longitudinal_memory_summary": longitudinal_memory_summary,
            "patient_history_context": history_context,
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
            "question_medications": question_medications,
            "requested_guideline_authorities": requested_guideline_authorities,
            "missing_requested_guideline_authorities": (
                missing_requested_guideline_authorities
            ),
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

    @staticmethod
    def _build_chart_lookup_bundle(
        question: str,
        normalized_user: Optional[str],
        user_profile: dict,
        role_config: RoleConfig,
        task_mode: TaskModeDecision,
        current_location: str,
        patient_history: PatientHistoryContext,
        longitudinal_memory_summary: str,
        context_graph: Optional["ContextGraph"] = None,
    ) -> Dict:
        """
        Same shape as _build_transformation_bundle (no retrieval, one
        answer_question call) with one deliberate difference: unlike
        documentation/translation, this must NOT drop the patient's chart
        data -- the whole point of chart_lookup mode is answering directly
        from it. patient_history.as_prompt_block() and
        longitudinal_memory_summary are already fully computed by this point
        in prepare_bundle (Step 2, before this branch), so no new
        chart-rendering code is needed here.
        """
        intent = IntentClassification(
            intent_category="administrative",
            risk_level="routine",
            pathway_hint="general_triage",
            confidence=1.0,
        )
        chart_text = "\n\n".join(
            part
            for part in [
                patient_history.as_prompt_block(),
                longitudinal_memory_summary,
                context_graph.relationship_prompt_block() if context_graph else "",
            ]
            if part
        )
        full_context = "\n\n".join(
            part for part in [task_mode.prompt_block(), chart_text] if part
        )
        return {
            "kind": "answer",
            "normalized_user": normalized_user,
            "user_profile": user_profile,
            "combined_sources": [],
            "personal_context": [],
            "longitudinal_memory_summary": longitudinal_memory_summary,
            "expanded_queries": [],
            "matches": [],
            "retrieval_mode": "chart_lookup",
            "full_context": full_context,
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
        patient_history: PatientHistoryContext,
        question_medications: Optional[List[str]] = None,
        context_graph: Optional["ContextGraph"] = None,
        missing_requested_guideline_authorities: Optional[List[str]] = None,
    ) -> Dict:
        limited_answer = self._build_limited_evidence_response(
            question=question,
            personal_context=personal_context,
            role_config=role_config,
            intent=intent,
            patient_history=patient_history,
            question_medications=question_medications or [],
            context_graph=context_graph,
        )
        if missing_requested_guideline_authorities:
            missing_names = ", ".join(missing_requested_guideline_authorities)
            limited_answer = (
                "> **Requested guidance unavailable:** I could not verify a source from "
                f"{missing_names} in the evidence available for this response. Related "
                "evidence must not be treated as a substitute.\n\n"
                + limited_answer
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
                    "missing_requested_guideline_authorities": (
                        missing_requested_guideline_authorities or []
                    ),
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
        context_graph: Optional["ContextGraph"] = None,
    ) -> str:
        parts = []

        if clinical_context and clinical_context.status != "insufficient":
            parts.append(clinical_context.as_prompt_block())

        relationship_block = (
            context_graph.relationship_prompt_block() if context_graph else ""
        )
        if relationship_block:
            parts.append(relationship_block)

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
                    f"(authority: {source.get('authority') or source.get('provider', 'unknown')}; "
                    f"jurisdiction: {source.get('jurisdiction', 'unspecified')}; "
                    f"quality: {quality_status}; use: {use_label}): {snippet}"
                    + (f"\nQuality notes: {quality_notes}" if quality_notes else "")
                )
            parts.append(
                "Biomedical evidence (tiered by source authority). Apply jurisdiction-specific "
                "recommendations only when they match the user's setting, and describe differences "
                "rather than silently combining conflicting UK and US guidance:\n"
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
                    "licence": metadata.get("licence", ""),
                    "licence_status": metadata.get("licence_status", ""),
                    "licence_url": metadata.get("licence_url", ""),
                }
            )
        return sources

    @staticmethod
    def _clarification_detail_unavailable(question: str) -> bool:
        """Return true when the user has already said the requested detail is unknown."""
        return bool(_UNAVAILABLE_DETAIL_PATTERN.search(question or ""))

    @staticmethod
    def _requests_multiple_interpretations(question: str) -> bool:
        """Do not force a choice when the user explicitly asks for a comparison."""
        return bool(
            re.search(
                r"\b(?:both|compare|comparison|versus|vs\.?|each)\b",
                question or "",
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _requested_guideline_authorities(question: str) -> List[str]:
        text = question or ""
        requested: List[str] = []
        for authority, aliases in _GUIDELINE_AUTHORITY_ALIASES.items():
            for alias in aliases:
                is_case_sensitive = alias in _CASE_SENSITIVE_AUTHORITY_ALIASES
                flags = 0 if is_case_sensitive else re.IGNORECASE
                search_term = authority if is_case_sensitive else alias
                if re.search(rf"\b{re.escape(search_term)}\b", text, flags):
                    requested.append(authority)
                    break
        return requested

    @staticmethod
    def _missing_requested_guideline_authorities(
        requested: List[str], sources: List[Dict]
    ) -> List[str]:
        missing: List[str] = []
        for authority in requested:
            aliases = _GUIDELINE_AUTHORITY_ALIASES.get(authority, (authority,))
            found = False
            for source in sources:
                if source.get("source_type") != "official_guidance":
                    continue
                source_text = " ".join(
                    str(source.get(field, ""))
                    for field in (
                        "provider",
                        "title",
                        "url",
                    )
                )
                if any(
                    re.search(
                        rf"\b{re.escape(alias)}\b", source_text, re.IGNORECASE
                    )
                    for alias in aliases
                ):
                    found = True
                    break
            if not found:
                missing.append(authority)
        return missing

    @staticmethod
    def _select_best_pubmed_section(sections: Dict[str, str]) -> Tuple[str, str]:
        for key in ("discussion", "conclusion", "introduction"):
            text = (sections.get(key) or "").strip()
            if text:
                return key, text
        return "", ""

    def _build_information_clarification_bundle(
        self,
        question: str,
        normalized_user: Optional[str],
        role_config: RoleConfig,
        intent: IntentClassification,
        patient_history: PatientHistoryContext,
        question_medications: Optional[List[str]] = None,
        context_graph: Optional["ContextGraph"] = None,
    ) -> Dict:
        clinical_roles = {
            "doctor", "nurse", "midwife", "physiotherapist", "healthcare_professional"
        }
        is_clinical = role_config.role_key in clinical_roles
        known_facts: List[str] = []
        if question_medications:
            known_facts.append(
                "The medicine named is " + ", ".join(question_medications[:3]) + "."
            )
        if (
            intent.intent_category == "medication_query"
            and patient_history.known_allergies
        ):
            known_facts.append(
                "The record lists this allergy: "
                + "; ".join(patient_history.known_allergies[:5])
                + "."
            )
        if (
            intent.intent_category == "chronic_condition"
            and patient_history.known_conditions
        ):
            known_facts.append(
                "The recorded conditions are "
                + "; ".join(patient_history.known_conditions[:4])
                + "."
            )
        if (
            intent.intent_category == "medication_query"
            and patient_history.known_medications
        ):
            known_facts.append(
                "I already have the current medicine list, so you do not need to repeat it."
            )
        if context_graph and context_graph.edges:
            relationship_context = format_relationships_for_user(
                context_graph.edges, max_edges=3
            )
            if relationship_context:
                known_facts.append(relationship_context)

        allergy_is_decision_relevant = bool(
            patient_history.known_allergies
            and any(
                "allerg" in item.lower()
                for item in intent.clarifying_questions
            )
        )

        safety_line = ""
        if allergy_is_decision_relevant:
            if is_clinical:
                safety_line = (
                    "Reconcile the recorded allergy against the exact product before administration."
                )
            else:
                safety_line = (
                    "Before taking the first or next dose, ask the prescribing clinic or a pharmacist "
                    "to confirm that this exact medicine was checked against your recorded allergy."
                )

        if is_clinical:
            opening = "I need these facts before giving a patient-specific recommendation."
        else:
            opening = "I need these answers before I can tell you what this means for you."

        substantive_line = ""
        if intent.intent_category == "medication_query" and not safety_line:
            substantive_line = (
                "Until this is clear, do not start, stop, double, or otherwise change a "
                "medicine based on this chat. Check the label and contact the prescriber "
                "or a pharmacist if a dose is due."
            )
        elif not safety_line:
            substantive_line = (
                "I can give conditional information now, but the missing detail may change "
                "what applies. Avoid relying on one interpretation until it is confirmed."
            )

        # A pure question gate with no safety-net leaves a genuinely worried
        # patient with nothing to act on while they gather the answers. For
        # symptom-adjacent routine categories, always give a concrete
        # threshold for acting sooner instead of waiting for the reply.
        safety_net_line = ""
        if intent.intent_category in {
            "symptom_triage", "chronic_condition", "maternity", "msk"
        }:
            if is_clinical:
                safety_net_line = (
                    "If a red flag emerges or the presentation changes while waiting for "
                    "this detail, escalate through the local pathway now rather than waiting "
                    "for a reply."
                )
            else:
                safety_net_line = (
                    "You do not need to wait for these answers if things get worse, feel "
                    "wrong, or you're worried -- seek care now instead."
                )

        parts = [item for item in (safety_line, substantive_line, opening) if item]
        if known_facts:
            parts.append(" ".join(known_facts))
        parts.append(
            "\n".join(
                f"{index}. {item}"
                for index, item in enumerate(intent.clarifying_questions[:3], start=1)
            )
        )
        if safety_net_line:
            parts.append(safety_net_line)
        parts.append("Reply with the numbered answers, and I'll give you a direct answer.")
        answer = "\n\n".join(parts)

        return {
            "kind": "final",
            "payload": {
                "answer_markdown": answer,
                "answer_text": answer,
                "sources": [],
                "personal_context": [],
                "follow_up_questions": [],
                "trace": {
                    "trace_id": "trace-information-clarify",
                    "created_at": _utc_now(),
                    "question": question,
                    "answer_preview": answer[:280],
                    "sources": [],
                    "retrieval_mode": "information_clarification_requested",
                    "role_key": role_config.role_key,
                    "intent_category": intent.intent_category,
                    "risk_level": intent.risk_level,
                    "clarification_reason": intent.clarification_reason,
                    "clarifying_questions": intent.clarifying_questions[:3],
                    "escalation_triggered": False,
                },
            },
        }

    def _build_limited_evidence_response(
        self,
        question: str,
        personal_context: List[Dict],
        role_config: RoleConfig,
        intent: IntentClassification,
        patient_history: PatientHistoryContext,
        question_medications: List[str],
        context_graph: Optional["ContextGraph"] = None,
    ) -> str:
        clinical_roles = {
            "doctor", "nurse", "midwife", "physiotherapist", "healthcare_professional"
        }
        is_clinical = role_config.role_key in clinical_roles
        allergy_text = "; ".join(patient_history.known_allergies[:5])
        medicine_text = ", ".join(question_medications[:3]) or "the medicine"
        condition_text = "; ".join(patient_history.known_conditions[:5])
        def include_recorded_relationships(answer: str) -> str:
            relationship_context = format_relationships_for_user(
                context_graph.edges if context_graph else [], max_edges=3
            )
            if not relationship_context:
                return answer
            return answer + "\n\n" + relationship_context

        if intent.risk_level in {"elevated", "urgent"} and _BREATHING_SAFETY_PATTERN.search(
            question or ""
        ):
            if is_clinical:
                safety_opening = (
                    "If the patient is currently struggling to breathe, choking, unable to "
                    "speak, becoming cyanosed or unresponsive, or rapidly deteriorating, "
                    "activate the local emergency pathway now. If the episode has stopped but "
                    "is recurring, arrange prompt clinical assessment."
                )
            else:
                safety_opening = (
                    "If the person is currently struggling to breathe, choking, unable to "
                    "speak, turning pale, blue or grey, becoming unresponsive, or rapidly "
                    "worsening, call local emergency services now. If the episode has stopped "
                    "but is recurring, arrange prompt clinical assessment."
                )
            return include_recorded_relationships(safety_opening)

        if _DISTRESS_SAFETY_PATTERN.search(question or ""):
            if is_clinical:
                return include_recorded_relationships(
                    "Acknowledge the distress and assess current safety, including thoughts of "
                    "self-harm, inability to care for self or a baby, psychotic symptoms, and "
                    "rapid deterioration. Use the urgent local mental-health or maternity pathway "
                    "if any are present; otherwise arrange timely clinical follow-up rather than "
                    "returning only clarification questions."
                )
            return include_recorded_relationships(
                "This sounds distressing, especially when it is disrupting sleep. If there are "
                "thoughts of self-harm, immediate danger, severe confusion, or an inability to "
                "care safely for yourself or a baby, seek emergency help now. Otherwise, arrange "
                "a timely review with a GP or maternity team and tell someone you trust today."
            )

        if intent.intent_category == "medication_query":
            if is_clinical:
                opening = (
                    f"I can't safely confirm current prescribing information for {medicine_text} "
                    "here, so I would not use this chat response to make the prescribing decision."
                )
                if allergy_text:
                    opening += (
                        f" The patient record lists {allergy_text}; reconcile that allergy "
                        "against the exact product before administration."
                    )
                return include_recorded_relationships(
                    opening
                    + " Please check the current BNF or local formulary, then tell me the indication, "
                    "patient group, and decision you need help with for a focused review."
                )

            opening = f"I want to help you check {medicine_text} safely."
            if allergy_text:
                opening += (
                    f" Your health record lists this allergy: {allergy_text}. Before taking "
                    "the medicine, ask the prescribing clinic or a pharmacist to check the exact "
                    "product against that allergy."
                )
            else:
                opening += (
                    " I can't safely confirm its current medicine information here, so please "
                    "check the label with the prescribing clinic or a pharmacist before changing "
                    "how you take it."
                )
            if condition_text:
                return include_recorded_relationships(
                    opening
                    + f" I understand this is a follow-up in the context of {condition_text}; "
                    "you do not need to repeat the condition. I could not verify enough current "
                    "guidance just now, so I will not invent a recovery time or claim that exercise "
                    "makes the medicine work faster. Tell me only whether you have started it, how "
                    "long you have taken it, whether the problem is improving or worsening, and "
                    "whether you have fever or feel generally unwell."
                )
            return include_recorded_relationships(
                opening
                + " What did the doctor say it was treating, and have you taken any doses yet? "
                "With that, I can focus on the exact use and explain it in plain language."
            )

        if intent.intent_category in {"general_info", "administrative"}:
            return include_recorded_relationships(
                "I could not verify a reliable current source for the specific guidance or code, "
                "so I will not invent one. Check the current official coding manual or the named "
                "guideline body for your jurisdiction. If you share the coding system, edition, "
                "and country, I can retry a focused source check. If the decision is time-sensitive, "
                "consult a clinician or coding specialist directly."
            )

        if is_clinical:
            return include_recorded_relationships(
                "I can't safely confirm enough current guidance to support a clinical decision from this "
                "answer. Please use the relevant local pathway now. If you send the exact condition, "
                "patient group, and decision point, I can provide a more focused evidence review."
            )

        if intent.intent_category in {
            "symptom_triage", "maternity", "msk", "mental_health", "chronic_condition"
        }:
            if condition_text:
                return include_recorded_relationships(
                    f"I understand this as a follow-up about {condition_text}; you do not need to "
                    "repeat where the problem is or name the condition again. I could not verify "
                    "enough current guidance just now, so I will not guess at a recovery time. "
                    "Tell me only what has changed since the last message, whether it is improving "
                    "or worsening, and whether any new fever or severe symptoms have appeared."
                )
            return include_recorded_relationships(
                "I can help, but I need a little more detail to give you a useful answer. "
                "Tell me where the problem is, when it started, how severe it is, and whether it is "
                "getting worse. If you already have a diagnosis or treatment, include its exact name."
            )

        return include_recorded_relationships(
            "I can help with this, but I need the specific health topic or decision you want to make. "
            "Tell me the symptom, condition, medicine, or report wording, and who the question is about. "
            "If this affects a decision that must be made now, consult a clinician directly."
        )


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
