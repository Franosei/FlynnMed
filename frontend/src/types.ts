export type Dict<T = unknown> = Record<string, T>;

export type ProductConfig = {
  product_name: string;
  product_tagline: string;
  product_subtitle: string;
  founder_name: string;
  support_email: string;
  terms_version: string;
  role_options: string[];
  role_terms: Record<string, RoleTerms>;
  privacy_notice_points: string[];
};

export type RoleTerms = {
  title: string;
  summary: string;
  bullets: string[];
  acknowledgement: string;
};

export type Profile = {
  username: string;
  display_name?: string;
  email?: string;
  care_context?: string;
  role?: string;
  clinical_role?: string;
  organization?: string;
  follow_up_preferences?: string;
  date_of_birth?: string;
  biological_sex?: string;
  created_at?: string;
  last_login?: string;
};

export type Message = {
  role: "user" | "assistant" | string;
  content: string;
  timestamp?: string;
  sources?: Source[];
  trace_id?: string;
  metadata?: Dict<any>;
  message_id?: string;
};

export type Source = {
  source_id?: string;
  title?: string;
  journal?: string;
  year?: string | number;
  url?: string;
  snippet?: string;
  evidence_tier?: number;
  tier_label?: string;
  tier_description?: string;
  evidence_quality_status?: string;
  evidence_quality_score?: number;
  question_alignment_score?: number;
  patient_alignment_score?: number;
  patient_alignment_facts?: string[];
  evidence_quality_reasons?: string[];
  usable_for_patient_specific_guidance?: boolean;
  // Evidence Ledger Phase 1 -- present only when the source was successfully
  // persisted with a verified passage; absent (not just empty) for a source
  // where persistence failed or no exact passage was extracted, so the UI
  // can distinguish "no passage available" from "not checked yet".
  exact_passage?: string;
  passage_locator?: string;
  source_version?: string;
  retrieved_at?: string;
};

// Evidence Ledger v2 (#11): the answer -> claim -> passage -> source /
// patient-fact lineage, fetched lazily via GET /api/evidence/trace/{trace_id}.
export type EvidenceTraceSource = {
  title: string;
  url: string;
  source_version: string;
  retrieved_at: string;
  is_full_document: boolean;
};

export type EvidenceTraceClaim = {
  claim_text: string;
  study_design: string;
  certainty: string;
  risk_of_bias: string;
  passage: {
    exact_text: string;
    locator: string;
    source: EvidenceTraceSource | null;
  } | null;
};

export type EvidenceTracePatientFact = {
  label: string;
  value: string;
  status: string;
  source: string;
  previous_fact_id: string | null;
};

export type EvidenceTraceContradiction = {
  topic: string;
  claim_a: string;
  claim_b: string;
  description: string;
  source_a: EvidenceTraceSource | null;
  source_b: EvidenceTraceSource | null;
};

export type AnswerClaimTrace = {
  claim_text: string;
  status: string;
  requires_evidence: boolean;
  module: string;
  llm_only_support: boolean;
  evidence_claims: EvidenceTraceClaim[];
  patient_facts: EvidenceTracePatientFact[];
};

export type EvidenceTrace = {
  trace_id: string;
  claims: AnswerClaimTrace[];
  contradictions: EvidenceTraceContradiction[];
};

export type ClinicalNote = {
  note_id: string;
  created_at: string;
  updated_at: string;
  username: string;
  display_name?: string;
  trace_id?: string;
  question?: string;
  subjective: string;
  objective: string;
  assessment: string;
  plan: string;
  urgency_level: string;
  requires_gp_visit: boolean;
  gp_visit_reason?: string;
  generated_by: string;
  edited_by?: string | null;
  email_sent: boolean;
  email_sent_at?: string | null;
};

export type Snapshot = {
  product: {
    name: string;
    tagline: string;
    subtitle: string;
    support_email: string;
  };
  user: string;
  profile: Profile;
  metrics: Record<string, number>;
  latest_triage: Dict<any>;
  chat_history: Message[];
  uploads: Dict<any>[];
  document_summaries: Dict<any>[];
  symptom_logs: Dict<any>[];
  medications: Dict<any>[];
  allergies: Dict<any>[];
  conditions: Dict<any>[];
  clinical_relationships?: Dict<any>[];
  vitals: Dict<any>[];
  triage_summaries: Dict<any>[];
  traces: Dict<any>[];
  audit: Dict<any>[];
  memory: Dict<any>;
  trial_search_result?: TrialSearchResult | null;
  clinical_notes: ClinicalNote[];
  safety_reviews?: SafetyReview[];
};

export type SafetyReview = {
  review_id: string;
  rule_id: string;
  priority: "emergency" | "urgent" | "review";
  category: string;
  status: "detected" | "patient_confirmed" | "follow_up_recorded";
  what_changed: string;
  why_it_matters: string;
  uncertainty: string;
  proposed_action: string;
  patient_facts: Array<{
    record_type: string;
    record_id: string;
    label: string;
    value: string;
    recorded_on: string;
  }>;
  evidence: Array<{
    claim: string;
    source_title: string;
    source_url: string;
    passage: string;
  }>;
  approver: string;
  outcome: {
    action_happened: boolean | null;
    patient_improved: boolean | null;
    note: string;
    updated_at: string;
  };
  writeback: { status: string; message: string };
};

export type AuthResponse = {
  token: string;
  profile: Profile;
  snapshot: Snapshot;
};

export type AccessGrant = {
  grant_id: string;
  patient_id: string;
  patient_name: string;
  clinician_name: string;
  clinician_role: string;
  organization: string;
  status: "pending" | "active" | "denied" | "revoked" | "expired";
  scopes: string[];
  request_reason: string;
  requested_at: string;
  decided_at: string;
  expires_at: string;
};

export type AccessOverview = {
  account_kind: "patient" | "clinician";
  patient_id?: string;
  requests: AccessGrant[];
  active_count: number;
  pending_count: number;
};

export type ClinicianPatientSummary = {
  patient: {
    patient_id: string;
    display_name: string;
    date_of_birth: string;
    biological_sex: string;
  };
  grant: AccessGrant;
  conditions: Dict<any>[];
  medications: Dict<any>[];
  allergies: Dict<any>[];
  vitals: Dict<any>[];
  symptoms: Dict<any>[];
  triage: Dict<any>[];
  care_plans: Dict<any>[];
  clinical_notes: Dict<any>[];
  clinical_relationships: Dict<any>[];
  chat_history: Dict<any>[];
  chat_history_authorized: boolean;
  previsit_summaries: PreVisitSummary[];
  proposed_medications: ProposedMedication[];
};

// ── Pre-visit summary + patient-scoped clinician chat ───────────────────────

export type PreVisitSummary = {
  id: string;
  status: "draft" | "released";
  generation_trigger: "ai_generated" | "clinician_edited" | "released";
  summary_text: string;
  authored_by_display_name: string;
  authored_by_clinical_role: string;
  authored_by_organization: string;
  released_at: string;
  released_by_display_name: string;
  released_by_clinical_role: string;
  created_at: string;
};

export type PreVisitChatMessage = {
  id: string;
  role: "clinician" | "assistant";
  content: string;
  authored_by_display_name: string;
  authored_by_clinical_role: string;
  sources?: Source[];
  // Live-turn only -- not persisted (PreVisitChatMessage has no DB column for
  // this yet), so absent when a message is loaded from chat history on reload.
  follow_up_questions?: Array<string | { display: string; prompt: string }>;
  created_at: string;
};

export type PrevisitChatStreamEvent =
  | { type: "user_message"; message: { role: "clinician"; content: string; timestamp: string } }
  | {
      type: "assistant_message";
      message: {
        role: "assistant";
        content: string;
        sources?: Source[];
        follow_up_questions?: Array<string | { display: string; prompt: string }>;
      };
    }
  | { type: "status"; message: string }
  | { type: "token"; delta: string }
  | { type: "error"; message: string }
  | { type: "done" };

// ── Medication proposals ─────────────────────────────────────────────────────

export type MedicationSafetyFlag = {
  severity?: string;
  summary?: string;
  [key: string]: any;
};

export type MedicationSafetyCheck = {
  allergy_flags: MedicationSafetyFlag[];
  interaction_flags: MedicationSafetyFlag[];
  unresolved_medications: string[];
  checked_at: string;
};

export type ProposedMedication = {
  id: string;
  status: "draft" | "released";
  generation_trigger: "ai_generated" | "clinician_edited" | "released";
  clinical_situation_text: string;
  candidate_medication_name: string;
  candidate_dose_frequency: string;
  rationale_text: string;
  citations: Source[];
  safety_check: MedicationSafetyCheck;
  override_reason: string;
  authored_by_display_name: string;
  authored_by_clinical_role: string;
  authored_by_organization: string;
  released_at: string;
  released_by_display_name: string;
  released_by_clinical_role: string;
  created_at: string;
};

export type TrialSearchResult = {
  searched_at?: string;
  trials: Dict<any>[];
  condition_terms: string[];
  medication_terms: string[];
  location: string;
  error?: string;
  context_status?: string;
  clinical_context?: Dict<any>;
};

export type ChatStreamEvent =
  | { type: "user_message"; message: Message }
  | { type: "assistant_message"; message: Message }
  | { type: "status"; message: string }
  | { type: "token"; delta: string }
  | { type: "snapshot"; snapshot: Snapshot }
  | { type: "error"; message: string; assistant_message?: Message }
  | { type: "done" };

export type FeedbackRating = "thumbs_up" | "thumbs_down";

// ── Care Plans ──────────────────────────────────────────────────────────────

export type CarePlanGoal = {
  id: string;
  text: string;
  metric?: string;
  target_months?: number;
  achieved?: boolean;
};

export type CarePlanTask = {
  id: string;
  text: string;
  time_of_day?: "morning" | "afternoon" | "evening" | "bedtime" | "any";
  rationale?: string;
  completed_dates?: string[];
};

export type MedReminder = {
  id: string;
  medication: string;
  dose?: string;
  timing?: string;
  notes?: string;
};

export type LabReminder = {
  id: string;
  test: string;
  frequency_months?: number;
  notes?: string;
  target_value?: string;
  last_done?: string | null;
  next_due?: string | null;
};

export type EscalationThreshold = {
  id: string;
  symptom: string;
  threshold?: string;
  action: string;
  urgency?: "call_999" | "a_and_e" | "gp_same_day" | "gp_routine" | "self_monitor";
};

export type CarePlanLifestyle = {
  diet?: string;
  exercise?: string;
  sleep?: string;
  weight?: string;
  mental_health?: string;
  smoking?: string;
  alcohol?: string;
  other?: string;
};

export type MissedCareItem = {
  id: string;
  item: string;
  frequency_months?: number;
  notes?: string;
  last_done?: string | null;
  overdue?: boolean;
};

export type CarePlan = {
  id: string;
  condition: string;
  title: string;
  status: "active" | "completed" | "paused";
  created_at: string;
  updated_at: string;
  goals: CarePlanGoal[];
  daily_tasks: CarePlanTask[];
  weekly_tasks: CarePlanTask[];
  medication_reminders: MedReminder[];
  lab_reminders: LabReminder[];
  escalation_thresholds: EscalationThreshold[];
  lifestyle: CarePlanLifestyle;
  missed_care_checklist: MissedCareItem[];
  evidence_summary: string;
  safety_notes?: string;
  clinical_context?: Dict<any>;
  validation?: { status?: string; violations?: string[] };
  gp_prep_summary?: string | null;
  after_visit_notes?: { text: string; date: string }[];
};

export type FeedbackResponse = {
  ok: boolean;
  already_rated: boolean;
  rating: FeedbackRating;
  saved: boolean;
  snapshot: Snapshot;
};
