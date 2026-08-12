import json
import os
from typing import Generator, Optional, TYPE_CHECKING

from dotenv import load_dotenv
from openai import OpenAI

from backend.product_config import PRODUCT_NAME
from backend.user_store import compute_current_age
from backend.agentic_health_contract import operating_contract_prompt

if TYPE_CHECKING:
    from backend.role_router import RoleConfig

load_dotenv()


class LLMHelper:
    """
    Wrapper around OpenAI's Chat Completions API for question answering and summarization.
    """

    # gpt-4o-mini for all generation -- both final answers and auxiliary calls.
    ANSWER_MODEL = "gpt-4o-mini"
    AUX_MODEL = "gpt-4o-mini"
    REQUEST_TIMEOUT_SECONDS = 120.0

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in .env")
        self.client = OpenAI(
            api_key=api_key,
            timeout=self.REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )

    def answer_question(
        self,
        question: str,
        context: str,
        chat_history: Optional[list[dict]] = None,
        stream: bool = False,
        user_profile: Optional[dict] = None,
        source_briefings: Optional[list[dict]] = None,
        longitudinal_memory: Optional[str] = None,
        role_config: Optional["RoleConfig"] = None,
        escalation_banner: str = "",
        policy_context_note: str = "",
        clinical_context: str = "",
        selected_skills: Optional[list[str]] = None,
        current_location: str = "",
        task_mode=None,
        response_completion_guidance: str = "",
        is_patient_scoped: bool = False,
    ) -> str | Generator[str, None, None]:
        """
        Creates a role-aware, evidence-grounded response using the supplied evidence dossier.
        Uses gpt-4o-mini for answer quality. Inline source citations like [S1] are mandatory.
        """
        if role_config:
            from backend.response_templates import get_persona_block

            persona = get_persona_block(role_config.role_key)
        else:
            persona = (
                f"You are {PRODUCT_NAME}, a safe and competent clinical information assistant supporting "
                "individual health users, caregivers, and healthcare teams. "
                "You provide decisive, evidence-grounded guidance with a clear next-step plan."
            )

        # A clinician asking about a specific patient's chart is not the patient --
        # without an explicit instruction the model defaults to the dominant
        # "your medications/your BP" patient-voice framing baked into the
        # SPECIFICITY REQUIREMENTS example below, even though role_config already
        # says "doctor". This is most visible on short chart-lookup questions
        # ("what was the recent medication"), which carry no other framing cue.
        clinician_patient_scoped = bool(
            role_config
            and role_config.role_key
            in ("doctor", "nurse", "midwife", "physiotherapist", "healthcare_professional")
            and is_patient_scoped
        )
        voice_instruction = ""
        if clinician_patient_scoped:
            voice_instruction = (
                "VOICE: You are a clinician reviewing a specific patient's chart -- you are not the "
                "patient. Refer to the patient in the THIRD PERSON by name or as 'the patient' (e.g. "
                "'Jane Whitfield is currently taking Metformin 500 mg twice daily'). Never address the "
                "patient directly as 'you'/'your'. This applies throughout the whole answer, including "
                "any monitoring, action, or follow-up section.\n\n"
            )
            specificity_example = (
                "- Quote the patient's actual recorded values only where relevant. "
                "Never write 'the patient's blood pressure appears elevated' without the number; "
                "write \"Jane's last recorded BP of X/Y mmHg on [date] is Stage 2 hypertension.\"\n"
            )
        else:
            specificity_example = (
                "- Quote the patient's actual recorded values only where relevant. "
                "Never write 'your blood pressure appears elevated' when you have the number; "
                "write 'your last recorded BP of X/Y mmHg on [date] is Stage 2 hypertension'.\n"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    f"{persona}\n\n"
                    f"{operating_contract_prompt(selected_skills or ['response_validation'], current_location)}\n\n"
                    f"{task_mode.prompt_block() if task_mode else ''}\n\n"
                    f"{voice_instruction}"
                    "CORE RULES:\n"
                    "0. Follow the CONTROLLED TASK MODE when one is supplied. For literal documentation, "
                    "translation, or a chart-data lookup, its output constraints override clinical headings, "
                    "evidence, citation, and patient-advice instructions below.\n"
                    "1. Use only the supplied evidence dossier and conversation context.\n"
                    "2. Use concise markdown with the role-appropriate section headings provided.\n"
                    "3. MANDATORY CITATIONS: every specific clinical claim -- a mechanism, causal explanation, "
                    "named diagnosis or condition-specific fact, statistic, or actionable recommendation (a "
                    "treatment, drug, dose, monitoring interval, or timeframe) -- that comes from the evidence "
                    "dossier MUST carry an inline [S#] marker at the exact sentence it supports, e.g. 'Weekly "
                    "talking-therapy sessions are recommended for mild postpartum depression [S2].' This applies "
                    "to every role, including patients -- a patient-facing answer is not exempt from citing its "
                    "sources. If you cannot point to a specific source for a specific claim, do not state it as "
                    "established fact: either omit it, or say plainly that it is general clinical knowledge/common "
                    "practice rather than something the reviewed evidence confirms. Do not force citations onto "
                    "conversational guidance, questions, or clearly labelled uncertainty.\n"
                    "4. Do not claim diagnostic certainty that the supplied facts cannot support. For a clinician, give one "
                    "prioritized provisional impression or must-not-miss syndrome; never give an unranked list of possibilities.\n"
                    "5. Surface emergency or urgent action first only when supported by the supplied facts.\n"
                    "6. Synthesize across sources -- do not copy any single source.\n"
                    "7. Prioritize Tier 1 (formal guidance) first, then Tier 2/3 for nuance.\n"
                    "8. Review longitudinal memory but mention only facts that materially change this answer. "
                    "Explain relevant connections in plain language and ignore unrelated history.\n"
                    "9. Source-use labels are private instructions. Never repeat their names or describe retrieval, "
                    "filtering, evidence checks, internal review, or whether evidence passed. Use patient-aligned "
                    "sources for patient-specific claims and other sources only for general context.\n"
                    "10. Do not infer age, sex, medicines, diagnoses, allergies, pregnancy status, or test results. "
                    "Mention a missing fact only when it is necessary for the requested decision; do not imply a "
                    "record exists or was reviewed unless actual personal context was supplied.\n"
                    "11. Do NOT add a disclaimer footer -- one is appended automatically.\n"
                    "12. If a clinical-context adjudication is supplied, it is binding. Do not reinterpret "
                    "a measurement or test as another specialty, even if the user's wording is commonly "
                    "used elsewhere.\n\n"
                    "SPECIFICITY REQUIREMENTS:\n"
                    f"{specificity_example}"
                    "- Name each relevant medication, condition, result, or vital by its recorded name; "
                    "do not force unrelated history into the response.\n"
                    "- Give concrete timeframes and thresholds only when the supplied evidence or "
                    "deterministic safety route supports them. Never invent a target, range, or deadline.\n"
                    "- Make monitoring points measurable when the record and evidence provide a measure; "
                    "otherwise say exactly what is still unknown.\n"
                    "- For clinical users: include specific investigation targets, drug doses where the "
                    "evidence explicitly supports them, and escalation criteria.\n\n"
                    "HEALTHCARE-PROFESSIONAL DECISION FORMAT (doctor, nurse, midwife, physiotherapist, or other clinician only):\n"
                    "- Put the decision in the first sentence: one leading working impression or one must-not-miss syndrome, plus disposition.\n"
                    "- When essential facts are missing, do not guess a diagnosis. State 'Diagnosis not established' and ask at most one "
                    "compact discriminator question (it may contain tightly related items such as onset, weakness, speech, or facial signs).\n"
                    "- Mention at most two alternative diagnoses, and only if they would change the immediate action. Rank them.\n"
                    "- Keep routine clinician answers to about 120 words unless complexity or an emergency requires more.\n"
                    "- Cite the decision and recommended action inline. Prefer a direct formal-guidance link represented by the supplied [S#] source.\n\n"
                    "Write naturally and directly. Avoid filler, repeated warnings, and generic lists. "
                    "Answer in the user's language unless they request another language. "
                    "If the next step depends on a clinician confirming the test or diagnosis, say that "
                    "plainly and explain exactly what information the patient should bring."
                ),
            }
        ]

        if role_config:
            from backend.response_templates import get_section_headings_text

            headings_text = get_section_headings_text(role_config.role_key)
        else:
            headings_text = (
                "Use only the few headings that materially help answer this request."
            )

        is_transformation = bool(task_mode and task_mode.is_transformation)
        is_chart_lookup = bool(task_mode and task_mode.mode == "chart_lookup")
        if is_transformation:
            response_instructions = (
                "Follow the controlled transformation mode exactly. Do not use role-oriented health headings, "
                "do not add citations, and do not append clinical commentary or follow-up questions."
            )
        elif is_chart_lookup:
            response_instructions = (
                "Answer directly and specifically from the chart data given above -- this is a record lookup, "
                "not an evidence review, so do not use role-oriented health headings and do not add [S#] "
                "citations (there are no external sources for this answer). A short, direct answer is "
                "correct; do not pad it with an emergency, differential, evidence, or monitoring section "
                "that wasn't asked for. If the specific fact requested isn't present in the data above, say "
                "so plainly rather than guessing."
            )
        else:
            response_instructions = (
                f"Available role-appropriate headings:\n{headings_text}\n"
                "Use only helpful sections; for a simple request, answer in one or two short paragraphs. "
                "Do not force an emergency, differential, evidence, disclaimer, or monitoring section.\n\n"
                "Cite claims drawn from evidence; omit a citation if direct support is unavailable and narrow "
                "or omit the claim instead.\n"
                "Where multiple sources agree, synthesize into one statement with combined citations.\n"
                "Label evidence tier (Tier 1 / Tier 2 / Tier 3) when it helps assess recommendation strength.\n"
                "Give specific routes, thresholds, and timeframes only when supported."
            )

            if role_config and role_config.role_key in (
                "doctor", "nurse", "midwife", "physiotherapist", "healthcare_professional"
            ):
                response_instructions += (
                    "\nFor this clinician-facing answer, use no more than three short sections. "
                    "The first line must contain the prioritized decision and disposition. "
                    "Do not repeat the same escalation criteria in multiple sections or add a generic differential list."
                )

        policy_block = ""
        if policy_context_note:
            policy_block = f"Clinical policy instructions (must be followed):\n{policy_context_note}\n\n"

        banner_instruction = ""
        if escalation_banner:
            banner_instruction = (
                "IMPORTANT: An escalation notice will be shown automatically directly above your "
                "response -- do NOT write, repeat, or paraphrase it or its wording anywhere in your "
                "answer. Do not open your answer with your own escalation/important notice; start "
                "directly with the substantive content. For any 'Escalate Now If' or 'Get Urgent "
                "Help If' heading, write only: 'See escalation notice above.' Do not list the same "
                "triggers again.\n\n"
            )

        memory_text = self._render_longitudinal_memory(longitudinal_memory)
        has_patient_data = (
            memory_text != "No durable patient-specific memory recorded yet."
        )

        messages.append(
            {
                "role": "user",
                "content": (
                    f"User profile:\n{self._render_profile_summary(user_profile)}\n\n"
                    f"Longitudinal patient memory (use these specific values in your answer):\n{memory_text}\n\n"
                    f"Recent conversation:\n{self._render_chat_history(chat_history)}\n\n"
                    f"Evidence dossier:\n{self._render_evidence_dossier(source_briefings, context)}\n\n"
                    f"Clinical context gate:\n{clinical_context or 'No cross-specialty context decision was needed.'}\n\n"
                    + (
                        "Controlled response requirements:\n"
                        f"{response_completion_guidance}\n\n"
                        if response_completion_guidance
                        else ""
                    )
                    + f"{policy_block}"
                    f"Current question:\n{question}\n\n"
                    f"{banner_instruction}"
                    f"{response_instructions}\n\n"
                    + (
                        "The longitudinal memory contains patient data. Use actual names and values only when they "
                        "are relevant to the current question; otherwise leave them out.\n\n"
                        if has_patient_data
                        else ""
                    )
                ),
            }
        )

        return (
            self._stream_response(messages, model=self.ANSWER_MODEL)
            if stream
            else self._complete_response(messages, model=self.ANSWER_MODEL)
        )

    def refresh_longitudinal_memory(
        self,
        existing_memory: str,
        new_information: str,
        user_profile: Optional[dict] = None,
        source_label: str = "conversation",
    ) -> str:
        """
        Merges new patient-specific facts into a durable longitudinal memory summary.
        Generic education, hypotheticals, and unsupported assistant inferences should
        not be written into the memory.
        """
        cleaned_new_information = (new_information or "").strip()
        if not cleaned_new_information:
            return (existing_memory or "").strip()

        messages = [
            {
                "role": "system",
                "content": (
                    "You maintain a longitudinal patient memory for a health assistant. "
                    "Update the memory using only durable patient-specific facts that are explicitly stated "
                    "in the supplied new information or clearly present in provided record summaries.\n\n"
                    "Rules:\n"
                    "1. Keep existing confirmed facts unless the new information clearly supersedes them.\n"
                    "2. Do not add generic medical education, hypothetical examples, or assistant speculation.\n"
                    "3. If the new information is not about the specific patient, leave the memory unchanged.\n"
                    "4. Keep the output concise, de-duplicated, and clinically useful.\n"
                    "5. Use the exact headings below.\n"
                    "6. If a section has no reliable facts, write `None noted`.\n"
                    "7. If there is no durable patient-specific information at all, return the existing memory unchanged.\n"
                    "8. Never invent medications, diagnoses, allergies, dates, or test results.\n"
                    "9. Write in plain text only -- no markdown, no asterisks, no bold, no bullet dashes, "
                    "no hyphens as list markers. Each fact should be a short plain sentence or phrase."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User profile:\n{self._render_profile_summary(user_profile)}\n\n"
                    f"Existing longitudinal memory:\n{self._render_longitudinal_memory(existing_memory)}\n\n"
                    f"New information source: {source_label}\n"
                    f"New information:\n{cleaned_new_information}\n\n"
                    "Return the refreshed longitudinal memory using exactly this structure:\n"
                    "Patient Summary:\n"
                    "Conditions and history:\n"
                    "Current treatments and medicines:\n"
                    "Recent symptoms or active concerns:\n"
                    "Investigations or notable results:\n"
                    "Risks, allergies, or safety flags:\n"
                    "Care plan and follow-up:\n"
                    "Open questions or uncertainties:\n"
                ),
            },
        ]
        return self._complete_response(messages)

    def summarize_user_health_record(self, record_text: str) -> str:
        """
        Summarizes an anonymized health document into a retrieval-friendly clinical overview.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are preparing an intake summary from an anonymized health document. "
                    "Capture diagnoses, therapies, abnormal findings, timelines, and care priorities "
                    "that would help a medical evidence system retrieve relevant literature."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Health document:\n{record_text}\n\n"
                    "Produce one short clinical paragraph followed by a compact plain-text list."
                ),
            },
        ]
        return self._complete_response(messages)

    def extract_medication_mentions(self, text: str) -> list[str]:
        cleaned_text = (text or "").strip()
        if not cleaned_text:
            return []

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract only direct medication names from the user's message. "
                        "Do not infer diagnoses, drug classes, supplements, or vague categories. "
                        "Return a JSON object with one key: medications."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Message:\n{cleaned_text}\n\n"
                        "Return JSON in this shape only:\n"
                        '{"medications": ["drug name"]}'
                    ),
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        payload = json.loads(raw)
        medications = payload.get("medications", [])
        if not isinstance(medications, list):
            return []
        return [str(item).strip() for item in medications if str(item).strip()][:6]

    def build_structured_triage(
        self,
        question: str,
        answer_markdown: str,
        fallback_triage: dict,
        intent_summary: str = "",
    ) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You produce a compact structured triage summary for a health assistant. "
                        "Use only the supplied answer and fallback safety route. "
                        "Never lower the acuity below the fallback next step. "
                        "Return a JSON object with these exact keys: urgency_level, next_step, what_to_monitor, rationale. "
                        "The next_step must be one of: Self-care, Primary-care clinician, "
                        "Local urgent-care service, Same-day review, Immediate review, "
                        "Local emergency services. Do not invent a national service or number. "
                        "what_to_monitor must be an array of up to 3 short phrases."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Intent summary:\n{intent_summary or 'Not available'}\n\n"
                        f"Fallback triage (minimum safe acuity):\n{json.dumps(fallback_triage)}\n\n"
                        f"Assistant answer:\n{answer_markdown}\n\n"
                        "Return only valid JSON."
                    ),
                },
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content.strip())

    def check_claim_source_alignment(
        self,
        answer_markdown: str,
        source_briefings: list[dict],
    ) -> list[dict]:
        """
        Reviews the answer and checks whether each factual claim is backed by
        a retrieved source. Returns a list of dicts:
          {"claim": "...", "status": "supported"|"general_knowledge",
           "requires_evidence": bool, "source_ids": [...]}
        Only the top 5 claims are checked to keep latency low. requires_evidence
        distinguishes claims a reader would expect to be evidence-backed (a
        mechanism, a statistic, a named causal relationship) from generic
        safety-netting or self-care language that doesn't need a citation --
        callers should only act on unsupported claims where this is true.
        """
        if not answer_markdown or not source_briefings:
            return []

        source_block = "\n".join(
            f"[{s['source_id']}] {s.get('title', '')} -- "
            f"{(s.get('detail_snippet') or s.get('snippet', ''))[:600]}"
            for s in source_briefings[:8]
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You check whether factual claims in a clinical answer are backed by the listed sources. "
                        "Extract up to 5 specific factual or clinical claims from the answer -- prioritise claims "
                        "that assert a specific mechanism, causal explanation, statistic, dosing detail, or "
                        "diagnostic/prognostic fact. Skip purely conversational lines, generic safety-netting "
                        "('seek care if symptoms worsen'), or requests for more information.\n"
                        "For each claim, decide: "
                        "'supported' if a listed source's content reasonably backs or is consistent with this "
                        "claim -- it does not need to be a verbatim match, the same guidance/finding in different "
                        "words still counts, and every listed source has already passed a relevance/quality "
                        "filter before reaching you, so give it a genuine chance to support the claim rather than "
                        "defaulting to 'general_knowledge' whenever the wording isn't identical; otherwise "
                        "'general_knowledge' (plausible but not actually consistent with any listed source).\n"
                        "Also set requires_evidence: true for any specific management or treatment recommendation "
                        "(e.g. 'start weekly talk-therapy sessions', 'take X twice daily'), a named diagnosis or "
                        "condition-specific fact, a mechanism, a causal explanation, a statistic, or a specific "
                        "frequency/dose/timeframe -- these are the claims a reader relies on as clinically "
                        "specific guidance, and being phrased gently ('consider', 'you might', 'often') does NOT "
                        "make them generic; a soft tone is not the same as a generic claim. Set requires_evidence: "
                        "false ONLY for content with no clinical specificity at all -- pure emotional/conversational "
                        "framing ('this can feel overwhelming'), a request for more information, or truly generic "
                        "safety-netting that names no specific action, frequency, or treatment ('seek care if "
                        "symptoms worsen', 'stay hydrated', 'monitor your mood').\n"
                        "source_ids must only be populated for a 'supported' claim, and must use the exact "
                        "bracketed id shown before each source above (e.g. 'S1', never '1' or a bare number) -- "
                        "leave source_ids empty for a 'general_knowledge' claim.\n"
                        "Return a JSON object with one key: claims. "
                        'Each claim is: {"claim": str, "status": str, "requires_evidence": bool, "source_ids": [str]}.'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Sources:\n{source_block}\n\n"
                        f"Answer (first 1200 chars):\n{answer_markdown[:1200]}\n\n"
                        "Return only valid JSON."
                    ),
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        payload = json.loads(raw)
        items = payload.get("claims", [])
        if not isinstance(items, list):
            return []
        result = []
        for item in items[:5]:
            if isinstance(item, dict) and item.get("claim"):
                result.append(
                    {
                        "claim": str(item.get("claim", "")).strip(),
                        "status": str(item.get("status", "general_knowledge")).strip(),
                        "requires_evidence": bool(item.get("requires_evidence", False)),
                        "source_ids": [str(s) for s in item.get("source_ids", [])],
                    }
                )
        return result

    def extract_medication_proposal(
        self,
        answer_markdown: str,
        source_briefings: list[dict],
    ) -> dict:
        """
        Extracts ONE specific candidate medication name + dose/frequency from
        answer_markdown -- but ONLY if the answer explicitly names a specific
        drug and dosing/frequency. Never infers or invents a candidate that
        isn't actually stated. answer_markdown is expected to have already
        passed check_claim_source_alignment/apply_claim_corrections (i.e. it's
        the cited, corrected answer_markdown from a real retrieval-backed
        response), so this is a constrained-extraction step over already-
        grounded text, not a fresh clinical judgment call of its own.

        Returns {} if the answer doesn't name a specific drug + dose/frequency
        -- callers must treat that as "no candidate," not retry or guess.
        Otherwise returns {"medication_name": str, "dose_frequency": str,
        "source_ids": [str]}.
        """
        if not answer_markdown or not source_briefings:
            return {}

        source_block = "\n".join(
            f"[{s['source_id']}] {s.get('title', '')}" for s in source_briefings[:8]
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract a single candidate medication proposal from a clinical answer, "
                        "if one is actually present. Look for a SPECIFIC named drug (generic or brand "
                        "name) together with a SPECIFIC dose or frequency (e.g. '500mg twice daily', "
                        "'one tablet at bedtime'). Do not infer a drug class, do not guess a dose that "
                        "isn't stated, and do not invent a candidate the answer doesn't actually name -- "
                        "if the answer only discusses non-pharmacological options, defers to a "
                        "clinician without naming a specific drug, or names a drug with no dose/"
                        "frequency given, there is no candidate.\n"
                        "If a candidate is present, set source_ids to the [S#] ids (exact bracketed "
                        "form, e.g. 'S1') from the sources below that support this specific "
                        "medication/dose -- leave empty if none directly support it.\n"
                        "Return a JSON object: {\"has_candidate\": bool, \"medication_name\": str, "
                        '"dose_frequency": str, "source_ids": [str]}. If has_candidate is false, '
                        "the other fields should be empty."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Sources:\n{source_block}\n\n"
                        f"Answer:\n{answer_markdown[:2000]}\n\n"
                        "Return only valid JSON."
                    ),
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        payload = json.loads(raw)
        if not payload.get("has_candidate"):
            return {}
        medication_name = str(payload.get("medication_name", "")).strip()
        dose_frequency = str(payload.get("dose_frequency", "")).strip()
        if not medication_name or not dose_frequency:
            return {}
        return {
            "medication_name": medication_name,
            "dose_frequency": dose_frequency,
            "source_ids": [str(s) for s in payload.get("source_ids", [])],
        }

    def apply_claim_corrections(
        self,
        answer_markdown: str,
        unsupported_claims: list[dict],
        source_briefings: list[dict],
        uncited_supported_claims: Optional[list[dict]] = None,
    ) -> str:
        """
        Two corrections in one pass (kept as a single call to avoid a second
        latency round-trip per correction type):

        1. Claims check_claim_source_alignment confirmed are NOT backed by any
           retrieved source (status=general_knowledge, requires_evidence=True)
           must stop reading as evidence-backed fact. Adding a hedge word like
           "often" or "may" is not enough on its own -- most flagged claims are
           already phrased that softly, so hedging alone leaves the actual
           problem (no source backs this) untouched. The sentence must instead
           make clear it is general clinical knowledge, not something the
           specific sources reviewed for this answer confirm -- and for a
           specific/actionable claim (a treatment, dose, or management step),
           point the reader to confirm it with a clinician rather than
           presenting it as established.
        2. Claims check_claim_source_alignment confirmed ARE backed by a
           specific source, but whose [S#] marker never made it into the
           text, get that citation inserted at the claim, unchanged otherwise.

        Nothing else in the answer should change -- structure, other claims,
        other citations, banners. Falls back to the original answer if the
        call fails or returns nothing usable, so a broken correction never
        blocks delivery.
        """
        uncited_supported_claims = uncited_supported_claims or []
        if (not unsupported_claims and not uncited_supported_claims) or not answer_markdown:
            return answer_markdown

        source_block = "\n".join(
            f"[{s['source_id']}] {s.get('title', '')}" for s in source_briefings[:8]
        )

        instruction_sections = []
        if unsupported_claims:
            claims_block = "\n".join(f'- "{c["claim"]}"' for c in unsupported_claims)
            instruction_sections.append(
                "CORRECTION A -- these claims are NOT backed by any of the listed "
                "retrieved sources; they are general knowledge at best:\n"
                f"{claims_block}\n"
                "Rewrite each so it clearly reads as general clinical knowledge, not "
                "something the specific sources reviewed for this answer confirm. Do "
                "not just add a hedge word ('often', 'may', 'can be') -- the sentence "
                "must stop implying it came from the cited evidence. For a specific, "
                "actionable claim (a treatment, dose, or management step), say it "
                "should be confirmed with a clinician rather than presenting it as "
                "established."
            )
        if uncited_supported_claims:
            claims_block = "\n".join(
                f'- "{c["claim"]}" -- insert {"".join(f"[{sid}]" for sid in c.get("source_ids", []))} '
                "immediately after this claim"
                for c in uncited_supported_claims
                if c.get("source_ids")
            )
            if claims_block:
                instruction_sections.append(
                    "CORRECTION B -- these claims ARE backed by the source(s) shown, but "
                    "the citation marker is missing from the text. Insert exactly the "
                    "marker shown, at the claim, without changing the claim's wording:\n"
                    f"{claims_block}"
                )

        if not instruction_sections:
            return answer_markdown

        prompt = (
            "You are correcting citation accuracy in a clinical answer, based on a "
            "claim-by-claim source-alignment check that has already been run.\n\n"
            + "\n\n".join(instruction_sections) + "\n\n"
            f"Sources available:\n{source_block}\n\n"
            "Make ONLY the corrections described above. Do not change anything else: "
            "keep the same structure, headings, other claims, other citations, "
            "banners, and tone. Do not add new claims or remove content unrelated to "
            "the claims listed above.\n\n"
            f"Original answer:\n{answer_markdown}\n\n"
            "Return only the revised answer text, nothing else."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.AUX_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            revised = (response.choices[0].message.content or "").strip()
            return revised or answer_markdown
        except Exception as exc:
            print(f"[LLMHelper] apply_claim_corrections failed, keeping original: {exc}")
            return answer_markdown

    def generate_follow_up_questions(
        self,
        question: str,
        answer: str,
        chat_history: Optional[list[dict]] = None,
        user_profile: Optional[dict] = None,
        patient_context: Optional[str] = None,
        role_key: str = "patient",
        is_patient_scoped: bool = False,
    ) -> list[str]:
        """
        Three distinct chip styles depending on who's asking and about whom:
        - patient (and caregiver -- not in the clinician tuple below, same as
          this codebase's other role checks): first-person "confirm this
          about yourself" chips, unchanged from the original design.
        - clinician, general/patient-agnostic chat (Evidence Review, or
          image/document analysis there -- is_patient_scoped=False): chips
          about terminology/mechanisms/differentials/evidence quality, never
          in any patient's voice.
        - clinician, viewing a specific patient's chart (previsit_chat --
          is_patient_scoped=True): chips phrased as clinical actions an
          examining clinician would check next, third-person about the
          patient, never first-person.
        """
        is_clinician = role_key in (
            "doctor", "nurse", "midwife", "physiotherapist", "healthcare_professional"
        )
        if not is_clinician:
            context = "patient"
        elif is_patient_scoped:
            context = "clinician_patient_scoped"
        else:
            context = "clinician_general"

        profile_text = self._render_profile_summary(user_profile)
        patient_data = (
            patient_context or ""
        ).strip() or "No structured patient data available."
        # Last 3 conversation turns -- full content, no truncation
        recent_turns = ""
        if chat_history:
            turns = [
                m
                for m in chat_history
                if m.get("role") in ("user", "assistant")
                and m.get("content", "").strip()
            ][-6:]  # last 3 pairs = 6 messages
            if turns:
                recent_turns = "\n".join(
                    f"{m['role'].title()}: {m['content'].strip()}" for m in turns
                )

        if context == "patient":
            system_prompt = (
                "You generate clickable follow-up chips shown after a clinical answer.\n\n"
                "A chip is a SHORT STATEMENT in the patient's voice -- something they might "
                "want to CONFIRM as true about themselves to refine the answer. "
                "Clicking a chip means the patient is saying 'yes, I have / experience this.'\n\n"
                "STRICT RULES:\n"
                "1. Chips must come from what the EVIDENCE AND ANSWER raised -- risk factors, "
                "associated symptoms, red flags, lifestyle triggers, or family history the "
                "research identified as relevant. Do NOT invent generic health questions.\n"
                "2. Never ask something the patient already described in their question.\n"
                "3. Never include source counts, numbers of papers, or any metadata from the "
                "answer -- chips are about the PATIENT, not the evidence database.\n"
                "4. Each chip 'display' must be a short first-person statement, max 8 words:\n"
                "   GOOD: 'I also have a fever', 'My dad had a heart attack', "
                "'Pain spreads to my jaw', 'I smoke about 10 a day'\n"
                "   BAD: 'How long have you had symptoms?', '3 sources reviewed', "
                "'Have you noticed any other symptoms?'\n"
                "5. Each chip 'prompt' is what gets sent to the model -- it must:\n"
                "   a) Start by identifying what the original question was about\n"
                "   b) State the confirmation as a fact the patient is adding\n"
                "   c) Ask how this changes or refines the answer\n"
                "   Example: 'Regarding my sore throat question -- I also have a fever of "
                "around 38.5°C. Does this change whether I need antibiotics or a GP visit?'\n\n"
                'Return JSON: {\'questions\': [{"display": str, "prompt": str}, ...]}, '
                "up to 5 items."
            )
        elif context == "clinician_general":
            system_prompt = (
                "You generate clickable follow-up chips shown after a clinical evidence-review "
                "answer for a clinician doing general medical research -- this question is NOT "
                "about a specific patient.\n\n"
                "A chip is a SHORT, THIRD-PERSON research question that deepens or extends the "
                "topic just discussed -- medical terminology, disease mechanisms, differential "
                "considerations, or evidence quality/strength. NEVER phrase a chip as a patient "
                "describing their own symptoms, and never address 'you' as if the clinician were "
                "the patient.\n\n"
                "STRICT RULES:\n"
                "1. Chips must come from a term, mechanism, finding, or evidence gap the ANSWER "
                "actually raised. Do NOT invent generic questions.\n"
                "2. Never ask something already answered in the question or answer.\n"
                "3. Each chip 'display' must be a short research question, max 10 words:\n"
                "   GOOD: 'Explain the mechanism of this interaction', 'What are the differential "
                "considerations?', 'How strong is this evidence?', 'Compare with an alternative "
                "treatment'\n"
                "   BAD: 'I also have a fever', 'Have you noticed other symptoms?' (patient-voice "
                "or patient-directed phrasing)\n"
                "4. Each chip 'prompt' is what gets sent to the model -- reference the original "
                "topic and ask the follow-up research question directly.\n\n"
                'Return JSON: {\'questions\': [{"display": str, "prompt": str}, ...]}, '
                "up to 5 items."
            )
        else:  # clinician_patient_scoped
            system_prompt = (
                "You generate clickable follow-up chips shown after a clinician's chat answer "
                "about a SPECIFIC patient's chart/record.\n\n"
                "A chip is a SHORT, THIRD-PERSON clinical action or consideration an examining "
                "clinician would want to check next about THIS patient -- never in the patient's "
                "own voice, never first-person ('I have...'), always about the patient in third "
                "person ('her', 'his', 'the patient').\n\n"
                "STRICT RULES:\n"
                "1. Chips must come from what the ANSWER or PATIENT RECORD raised -- a value to "
                "check, a risk factor, an interaction to screen for, or a trend to review. Do NOT "
                "invent generic questions.\n"
                "2. Never ask something already answered.\n"
                "3. Each chip 'display' must be a short third-person clinical action, max 10 words:\n"
                "   GOOD: 'Check her renal function before dose escalation', 'Review his last "
                "three HbA1c readings', 'Screen for interactions with her NSAID use'\n"
                "   BAD: 'I also have a fever', 'Have you noticed...' (never first or second "
                "person)\n"
                "4. Each chip 'prompt' is what gets sent to the model -- reference the patient "
                "and the original topic, phrased as a clinician's follow-up request.\n\n"
                'Return JSON: {\'questions\': [{"display": str, "prompt": str}, ...]}, '
                "up to 5 items."
            )

        if context == "clinician_general":
            # The acting clinician's OWN profile/health-record data is irrelevant (and
            # potentially confusing) for a patient-agnostic research question -- omit
            # that block entirely rather than injecting the clinician's own vitals/meds.
            user_content = (
                f"Role: {role_key}\n"
                + (
                    f"Recent conversation:\n{recent_turns}\n\n"
                    if recent_turns
                    else ""
                )
                + f"Original question:\n{question}\n\n"
                f"Answer given (read this to find a term, mechanism, or evidence gap "
                f"raised but not yet explored):\n{answer}\n\n"
                "Generate up to 5 chips. Each must be grounded in a specific point from "
                "the answer above. The 'prompt' must reference the original topic so the "
                "model knows exactly what conversation it is continuing. "
                "Return only valid JSON."
            )
        else:
            user_content = (
                f"Role: {role_key}\n"
                f"Patient profile:\n{profile_text}\n\n"
                f"Patient health record:\n{patient_data}\n\n"
                + (
                    f"Recent conversation:\n{recent_turns}\n\n"
                    if recent_turns
                    else ""
                )
                + f"Original question:\n{question}\n\n"
                f"Answer given (read this to find what risk factors, red flags, "
                f"or associated findings were raised but not yet confirmed):\n{answer}\n\n"
                "Generate up to 5 chips. Each must be grounded in a specific finding from "
                "the answer above. The 'prompt' must reference the original topic so the "
                "model knows exactly what conversation it is continuing. "
                "Return only valid JSON."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.AUX_MODEL,
                messages=messages,
                temperature=0.4,
                response_format={"type": "json_object"},
            )
            raw = json.loads(response.choices[0].message.content.strip())
            items = raw.get("questions", [])
            if not isinstance(items, list):
                return []
            result = []
            for item in items[:5]:
                if isinstance(item, dict):
                    display = str(item.get("display", "")).strip()
                    prompt = str(item.get("prompt", display)).strip()
                    if display:
                        result.append({"display": display, "prompt": prompt})
                elif isinstance(item, str) and item.strip():
                    result.append({"display": item.strip(), "prompt": item.strip()})
            return result
        except Exception as exc:
            print(f"Follow-up question generation failed: {exc}")
            return []

    def _complete_response(self, messages, model: Optional[str] = None) -> str:
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def _stream_response(
        self, messages, model: Optional[str] = None
    ) -> Generator[str, None, None]:
        stream = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=0.2,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    @staticmethod
    def _render_chat_history(chat_history: Optional[list[dict]]) -> str:
        if not chat_history:
            return "No prior conversation."

        lines = []
        for message in chat_history[-6:]:
            role = message.get("role", "user").title()
            content = message.get("content", "").strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else "No prior conversation."

    @staticmethod
    def _render_profile_summary(user_profile: Optional[dict]) -> str:
        if not user_profile:
            return "No additional user profile available."

        fragments = []

        # Demographics -- listed first as they modify almost every clinical guideline
        dob = (user_profile.get("date_of_birth") or "").strip()
        age = compute_current_age(dob)
        if age is not None:
            fragments.append(f"Age: {age} years")
        sex = (user_profile.get("biological_sex") or "").strip()
        if sex and sex != "Prefer not to say":
            fragments.append(f"Biological sex: {sex}")

        for field in (
            "display_name",
            "role",
            "care_context",
            "organization",
            "follow_up_preferences",
        ):
            value = (user_profile.get(field) or "").strip()
            if value:
                fragments.append(f"{field.replace('_', ' ').title()}: {value}")
        return (
            "\n".join(fragments)
            if fragments
            else "No additional user profile available."
        )

    @staticmethod
    def _render_longitudinal_memory(longitudinal_memory: Optional[str]) -> str:
        cleaned = (longitudinal_memory or "").strip()
        return cleaned or "No durable patient-specific memory recorded yet."

    @staticmethod
    def _render_evidence_dossier(
        source_briefings: Optional[list[dict]], fallback_context: str
    ) -> str:
        if source_briefings:
            blocks = []
            for source in source_briefings:
                tier_label = source.get("tier_label", "")
                tier_str = f" | {tier_label}" if tier_label else ""
                blocks.append(
                    "\n".join(
                        [
                            f"[{source['source_id']}] {source.get('title', 'Untitled article')}{tier_str}",
                            f"Source type: {source.get('source_type', 'evidence source')}",
                            f"Provider: {source.get('provider', source.get('journal', 'Unknown provider'))}",
                            f"Journal: {source.get('journal', 'Unknown journal')}",
                            f"Year: {source.get('year', 'Unknown year')}",
                            f"Section: {source.get('section', 'Retrieved text')}",
                            f"Relevance: {source.get('relevance', source.get('similarity', 'n/a'))}",
                            f"Quality status: {source.get('evidence_quality_status', 'question_aligned')}",
                            f"Question alignment: {source.get('question_alignment_score', 'n/a')}",
                            f"Patient alignment: {source.get('patient_alignment_score', 'n/a')}",
                            (
                                "Patient-specific use: yes"
                                if source.get("usable_for_patient_specific_guidance")
                                else "Patient-specific use: no - general/background context only"
                            ),
                            (
                                "Matched profile facts: "
                                + ", ".join(
                                    source.get("patient_alignment_facts", [])[:5]
                                )
                                if source.get("patient_alignment_facts")
                                else "Matched profile facts: none"
                            ),
                            (
                                "Quality notes: "
                                + "; ".join(
                                    source.get("evidence_quality_reasons", [])[:3]
                                )
                                if source.get("evidence_quality_reasons")
                                else "Quality notes: none"
                            ),
                            f"Evidence: {source.get('detail_snippet', source.get('snippet', source.get('evidence', '')))}",
                        ]
                    )
                )
            return "\n\n".join(blocks)

        return fallback_context or "No biomedical evidence was retrieved."
