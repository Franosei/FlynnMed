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

        messages = [
            {
                "role": "system",
                "content": (
                    f"{persona}\n\n"
                    f"{operating_contract_prompt(selected_skills or ['response_validation'], current_location)}\n\n"
                    f"{task_mode.prompt_block() if task_mode else ''}\n\n"
                    "CORE RULES:\n"
                    "0. Follow the CONTROLLED TASK MODE when one is supplied. For literal documentation or "
                    "translation, its output constraints override clinical headings, evidence, citation, and "
                    "patient-advice instructions below.\n"
                    "1. Use only the supplied evidence dossier and conversation context.\n"
                    "2. Use concise markdown with the role-appropriate section headings provided.\n"
                    "3. Cite clinical claims that rely on the supplied sources inline with [S1], [S1][S2], etc. "
                    "Do not force citations onto conversational guidance, questions, or clearly labelled uncertainty.\n"
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
                    "- Quote the patient's actual recorded values only where relevant. "
                    "Never write 'your blood pressure appears elevated' when you have the number; "
                    "write 'your last recorded BP of X/Y mmHg on [date] is Stage 2 hypertension'.\n"
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
        if is_transformation:
            response_instructions = (
                "Follow the controlled transformation mode exactly. Do not use role-oriented health headings, "
                "do not add citations, and do not append clinical commentary or follow-up questions."
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
                f"IMPORTANT: Begin your response with this escalation notice verbatim:\n"
                f"{escalation_banner}\n\n"
                "Because the escalation notice is already at the top, do NOT repeat or paraphrase "
                "its criteria anywhere in the body sections. For any 'Escalate Now If' or "
                "'Get Urgent Help If' heading, write only: 'See escalation notice above.' "
                "Do not list the same triggers again.\n\n"
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
            f"[{s['source_id']}] {s.get('title', '')} -- {s.get('snippet', '')[:200]}"
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
                        "'supported' (a listed source clearly backs it), or "
                        "'general_knowledge' (plausible but not directly in any listed source).\n"
                        "Also set requires_evidence: true if this is the kind of specific clinical claim a "
                        "reader would reasonably expect to be evidence-backed; false if it's generic safety "
                        "advice, common-sense self-care, or too vague to need a citation.\n"
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

    def rewrite_unsupported_claims(
        self,
        answer_markdown: str,
        unsupported_claims: list[dict],
        source_briefings: list[dict],
    ) -> str:
        """
        Revises an answer so that claims flagged by check_claim_source_alignment
        as evidence-requiring but unsupported read as appropriately hedged
        general knowledge instead of unqualified fact. Nothing else in the
        answer should change -- structure, other claims, citations, banners.
        Falls back to the original answer if the rewrite call fails or returns
        nothing usable, so a broken correction never blocks delivery.
        """
        if not unsupported_claims or not answer_markdown:
            return answer_markdown

        claims_block = "\n".join(f'- "{c["claim"]}"' for c in unsupported_claims)
        source_block = "\n".join(
            f"[{s['source_id']}] {s.get('title', '')}" for s in source_briefings[:8]
        )

        prompt = (
            "You are revising a clinical answer. The following specific claims in it were "
            "checked and are NOT backed by any of the listed retrieved sources -- they are "
            "general knowledge at best, not something the retrieved evidence confirms:\n\n"
            f"{claims_block}\n\n"
            f"Sources available:\n{source_block}\n\n"
            "Rewrite the answer below so that ONLY these specific claims are softened with "
            "appropriate hedging (e.g. 'often', 'can be', 'it's possible that' -- or a note "
            "that it would need clinical confirmation) so they no longer read as confirmed "
            "fact. Do not change anything else: keep the same structure, headings, other "
            "claims, citations, banners, and tone. Do not add new claims or remove content "
            "unrelated to the flagged claims.\n\n"
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
            print(f"[LLMHelper] rewrite_unsupported_claims failed, keeping original: {exc}")
            return answer_markdown

    def generate_follow_up_questions(
        self,
        question: str,
        answer: str,
        chat_history: Optional[list[dict]] = None,
        user_profile: Optional[dict] = None,
        patient_context: Optional[str] = None,
        role_key: str = "patient",
    ) -> list[str]:
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

        messages = [
            {
                "role": "system",
                "content": (
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
                ),
            },
            {
                "role": "user",
                "content": (
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
                ),
            },
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
