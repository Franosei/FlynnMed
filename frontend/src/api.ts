import type { AccessGrant, AccessOverview, AuthResponse, CarePlan, ChatStreamEvent, ClinicalNote, ClinicianPatientSummary, FeedbackRating, FeedbackResponse, PreVisitChatMessage, PreVisitSummary, PrevisitChatStreamEvent, ProductConfig, ProposedMedication, SafetyReview, Snapshot } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";
const TOKEN_KEY = "flynnmed_token";

export function getStoredToken(): string {
  return localStorage.getItem(TOKEN_KEY) ?? "";
}

export function setStoredToken(token: string): void {
  if (token) {
    localStorage.setItem(TOKEN_KEY, token);
  } else {
    localStorage.removeItem(TOKEN_KEY);
  }
}

type RequestOptions = RequestInit & {
  auth?: boolean;
};

async function readError(response: Response): Promise<string> {
  try {
    const payload = await response.json();
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
    return JSON.stringify(payload.detail ?? payload);
  } catch {
    return response.statusText || "Request failed.";
  }
}

export async function apiRequest<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const headers = new Headers(options.headers);
  const hasFormData = options.body instanceof FormData;
  if (!hasFormData && options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (options.auth !== false) {
    const token = getStoredToken();
    if (token) {
      headers.set("Authorization", `Bearer ${token}`);
    }
  }

  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers
  });

  if (!response.ok) {
    throw new Error(await readError(response));
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
}

export function getConfig(): Promise<ProductConfig> {
  return apiRequest<ProductConfig>("/api/config", { auth: false });
}

export function login(identifier: string, password: string): Promise<AuthResponse> {
  return apiRequest<AuthResponse>("/api/auth/login", {
    auth: false,
    method: "POST",
    body: JSON.stringify({ identifier, password })
  });
}

export function signup(payload: Record<string, unknown>): Promise<AuthResponse> {
  return apiRequest<AuthResponse>("/api/auth/signup", {
    auth: false,
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function fetchSnapshot(): Promise<Snapshot> {
  return apiRequest<Snapshot>("/api/snapshot");
}

export function updateSafetyReview(
  reviewId: string,
  payload: {
    status: "detected" | "patient_confirmed" | "follow_up_recorded";
    action_happened?: boolean;
    patient_improved?: boolean;
    note?: string;
  }
): Promise<{ review: SafetyReview; snapshot: Snapshot }> {
  return apiRequest(`/api/safety-reviews/${encodeURIComponent(reviewId)}`, {
    method: "PATCH",
    body: JSON.stringify(payload)
  });
}

export function fetchAccessOverview(): Promise<AccessOverview> {
  return apiRequest<AccessOverview>("/api/access");
}

export function requestPatientAccess(payload: {
  patient_id: string;
  reason: string;
  include_chat_history: boolean;
}): Promise<AccessGrant> {
  return apiRequest<AccessGrant>("/api/access/requests", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function decidePatientAccess(
  grantId: string,
  approve: boolean,
  decisionNote = ""
): Promise<AccessGrant> {
  return apiRequest<AccessGrant>(`/api/access/requests/${grantId}/decision`, {
    method: "POST",
    body: JSON.stringify({ approve, decision_note: decisionNote })
  });
}

export function revokePatientAccess(grantId: string): Promise<AccessGrant> {
  return apiRequest<AccessGrant>(`/api/access/requests/${grantId}`, {
    method: "DELETE"
  });
}

export function fetchClinicianPatient(patientId: string): Promise<ClinicianPatientSummary> {
  return apiRequest<ClinicianPatientSummary>(
    `/api/clinician/patients/${encodeURIComponent(patientId)}`
  );
}

// ── Pre-visit summary + patient-scoped clinician chat ───────────────────────

export function generatePrevisitSummary(patientId: string): Promise<PreVisitSummary> {
  return apiRequest<PreVisitSummary>(
    `/api/clinician/patients/${encodeURIComponent(patientId)}/summary`,
    { method: "POST" }
  );
}

export function savePrevisitSummaryDraft(patientId: string, summaryText: string): Promise<PreVisitSummary> {
  return apiRequest<PreVisitSummary>(
    `/api/clinician/patients/${encodeURIComponent(patientId)}/summary/draft`,
    { method: "PATCH", body: JSON.stringify({ summary_text: summaryText }) }
  );
}

export function releasePrevisitSummary(patientId: string, summaryText: string): Promise<PreVisitSummary> {
  return apiRequest<PreVisitSummary>(
    `/api/clinician/patients/${encodeURIComponent(patientId)}/summary/release`,
    { method: "POST", body: JSON.stringify({ summary_text: summaryText }) }
  );
}

export function fetchPrevisitChatHistory(patientId: string): Promise<{ messages: PreVisitChatMessage[] }> {
  return apiRequest(`/api/clinician/patients/${encodeURIComponent(patientId)}/chat`);
}

export async function streamPrevisitChat(
  patientId: string,
  message: string,
  onEvent: (event: PrevisitChatStreamEvent) => void
): Promise<void> {
  const token = getStoredToken();
  const response = await fetch(
    `${API_BASE}/api/clinician/patients/${encodeURIComponent(patientId)}/chat`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`
      },
      body: JSON.stringify({ message })
    }
  );
  return consumeNdjsonStream<PrevisitChatStreamEvent>(response, onEvent);
}

export function fetchMyPrevisitSummaries(): Promise<{ summaries: PreVisitSummary[] }> {
  return apiRequest("/api/previsit-summaries");
}

// ── Medication proposals ─────────────────────────────────────────────────────

export function generateMedicationProposal(
  patientId: string,
  clinicalSituation: string
): Promise<ProposedMedication> {
  return apiRequest<ProposedMedication>(
    `/api/clinician/patients/${encodeURIComponent(patientId)}/medication-proposals`,
    { method: "POST", body: JSON.stringify({ clinical_situation: clinicalSituation }) }
  );
}

export function saveMedicationProposalDraft(
  patientId: string,
  payload: {
    clinical_situation_text: string;
    candidate_medication_name: string;
    candidate_dose_frequency: string;
    rationale_text: string;
    citations: unknown[];
  }
): Promise<ProposedMedication> {
  return apiRequest<ProposedMedication>(
    `/api/clinician/patients/${encodeURIComponent(patientId)}/medication-proposals/draft`,
    { method: "PATCH", body: JSON.stringify(payload) }
  );
}

export function releaseMedicationProposal(
  patientId: string,
  payload: {
    clinical_situation_text: string;
    candidate_medication_name: string;
    candidate_dose_frequency: string;
    rationale_text: string;
    citations: unknown[];
    override_reason: string;
  }
): Promise<ProposedMedication> {
  return apiRequest<ProposedMedication>(
    `/api/clinician/patients/${encodeURIComponent(patientId)}/medication-proposals/release`,
    { method: "POST", body: JSON.stringify(payload) }
  );
}

export function fetchMyMedicationProposals(): Promise<{ proposals: ProposedMedication[] }> {
  return apiRequest("/api/my-medication-proposals");
}

/**
 * Shared ndjson stream consumer -- one JSON event per line, tolerant of
 * partial flushes/heartbeats (unparseable lines are dropped rather than
 * thrown). Used by every streaming endpoint in this file so the buffering
 * logic lives in exactly one place.
 */
async function consumeNdjsonStream<TEvent>(
  response: Response,
  onEvent: (event: TEvent) => void
): Promise<void> {
  if (!response.ok || !response.body) {
    throw new Error(await readError(response));
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const processLine = (line: string) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    try {
      onEvent(JSON.parse(trimmed) as TEvent);
    } catch {
      // Ignore unparseable lines (partial flushes, heartbeats)
    }
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) processLine(line);
    }
    if (buffer.trim()) processLine(buffer);
  } catch (err) {
    // The connection dropped mid-stream -- surface a clean error event
    const message = err instanceof Error ? err.message : "The connection was interrupted.";
    onEvent({ type: "error", message: `Stream interrupted: ${message}` } as TEvent);
  }
}

export async function streamChat(
  message: string,
  onEvent: (event: ChatStreamEvent) => void
): Promise<void> {
  const token = getStoredToken();
  const response = await fetch(`${API_BASE}/api/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`
    },
    body: JSON.stringify({ message })
  });
  return consumeNdjsonStream<ChatStreamEvent>(response, onEvent);
}

export async function streamImageAnalysis(
  message: string,
  image: File,
  onEvent: (event: ChatStreamEvent) => void
): Promise<void> {
  const token = getStoredToken();
  const form = new FormData();
  form.append("message", message);
  form.append("image", image);

  const response = await fetch(`${API_BASE}/api/chat/image/stream`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`
    },
    body: form
  });
  return consumeNdjsonStream<ChatStreamEvent>(response, onEvent);
}

export async function streamDocumentAnalysis(
  message: string,
  document: File,
  onEvent: (event: ChatStreamEvent) => void
): Promise<void> {
  const token = getStoredToken();
  const form = new FormData();
  form.append("message", message);
  form.append("document", document);

  const response = await fetch(`${API_BASE}/api/chat/document/stream`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`
    },
    body: form
  });
  return consumeNdjsonStream<ChatStreamEvent>(response, onEvent);
}

export interface UploadExtracted {
  vitals: unknown[];
  medications: unknown[];
  allergies: unknown[];
  conditions: unknown[];
  extraction_errors?: string[];
  extraction_method?: string;
}

export function uploadDocuments(files: File[], processUnverified: boolean): Promise<{
  processed: { file: string; extracted?: UploadExtracted }[];
  pending: unknown[];
  duplicates: { file: string; message: string }[];
  rejected: { file: string; message: string }[];
  snapshot: Snapshot;
}> {
  const form = new FormData();
  files.forEach((file) => form.append("files", file));
  form.append("process_unverified", String(processUnverified));
  return apiRequest("/api/uploads", {
    method: "POST",
    body: form
  });
}

export function transcribeAudio(file: File): Promise<{ text: string }> {
  const form = new FormData();
  form.append("audio", file);
  return apiRequest("/api/voice/transcribe", {
    method: "POST",
    body: form
  });
}

export function rateResponse(payload: {
  trace_id: string;
  message_id?: string;
  rating: FeedbackRating;
}): Promise<FeedbackResponse> {
  return apiRequest<FeedbackResponse>("/api/feedback", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

// ── Clinical notes ─────────────────────────────────────────────────────────

export function generateNote(payload: {
  question?: string;
  conversation_summary?: string;
  trace_id?: string;
}): Promise<{ note: ClinicalNote; snapshot: Snapshot }> {
  return apiRequest("/api/notes", { method: "POST", body: JSON.stringify(payload) });
}

export function updateNote(
  noteId: string,
  updates: Partial<Pick<ClinicalNote, "subjective" | "objective" | "assessment" | "plan" | "urgency_level" | "requires_gp_visit" | "gp_visit_reason">>
): Promise<{ note: ClinicalNote; snapshot: Snapshot }> {
  return apiRequest(`/api/notes/${noteId}`, { method: "PUT", body: JSON.stringify(updates) });
}

export function deleteNote(noteId: string): Promise<void> {
  return apiRequest(`/api/notes/${noteId}`, { method: "DELETE" });
}

export function emailNote(noteId: string): Promise<{ ok: boolean; sent_to: string; snapshot: Snapshot }> {
  return apiRequest(`/api/notes/${noteId}/email`, { method: "POST" });
}

export function sendUrgentAlert(reason: string, urgencyLevel: string): Promise<{ ok: boolean; sent_to: string }> {
  return apiRequest("/api/email/urgent", {
    method: "POST",
    body: JSON.stringify({ reason, urgency_level: urgencyLevel })
  });
}

export async function downloadProtectedFile(path: string, filename: string): Promise<void> {
  const token = getStoredToken();
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      Authorization: `Bearer ${token}`
    }
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

// ── Care Plans ───────────────────────────────────────────────────────────────

export function listCarePlans(): Promise<CarePlan[]> {
  return apiRequest<CarePlan[]>("/api/care-plans");
}

export function deleteCarePlan(planId: string): Promise<{ ok: boolean }> {
  return apiRequest(`/api/care-plans/${planId}`, { method: "DELETE" });
}

export function toggleCarePlanTask(
  planId: string,
  taskId: string,
  done: boolean
): Promise<CarePlan> {
  return apiRequest<CarePlan>(`/api/care-plans/${planId}/tasks/${taskId}`, {
    method: "PATCH",
    body: JSON.stringify({ done })
  });
}

export function addAfterVisitNote(planId: string, note: string): Promise<CarePlan> {
  return apiRequest<CarePlan>(`/api/care-plans/${planId}/after-visit`, {
    method: "POST",
    body: JSON.stringify({ note })
  });
}

export function generateGpPrep(planId: string): Promise<{ gp_prep_summary: string; plan: CarePlan }> {
  return apiRequest(`/api/care-plans/${planId}/gp-prep`, { method: "POST" });
}

export interface ClarifyOption {
  display: string;
  prompt: string;
}

export async function generateCarePlan(
  condition: string,
  chatSummary: string,
  onProgress: (msg: string) => void,
  onClarify?: (question: string, options: ClarifyOption[]) => void,
  clarification?: { question: string; answer: string }
): Promise<CarePlan | null> {
  const token = getStoredToken();
  const response = await fetch(`${API_BASE}/api/care-plans/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`
    },
    body: JSON.stringify({
      condition,
      chat_summary: chatSummary,
      clarification_question: clarification?.question ?? "",
      clarification_answer: clarification?.answer ?? ""
    })
  });

  if (!response.ok || !response.body) {
    throw new Error(await readError(response));
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalPlan: CarePlan | null = null;
  let clarified = false;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      let event: { type?: string; message?: string; plan?: CarePlan; question?: string; options?: ClarifyOption[] };
      try {
        event = JSON.parse(trimmed);
      } catch {
        continue; // malformed/incomplete line -- skip, don't let it mask real events
      }
      if (event.type === "progress") onProgress(event.message as string);
      else if (event.type === "done") finalPlan = event.plan as CarePlan;
      else if (event.type === "clarify") {
        clarified = true;
        onClarify?.(event.question as string, (event.options ?? []) as ClarifyOption[]);
      } else if (event.type === "error") throw new Error(event.message as string);
    }
  }

  if (!finalPlan && !clarified) throw new Error("Care plan generation did not complete.");
  return finalPlan;
}
