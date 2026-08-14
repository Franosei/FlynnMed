import { Component, FormEvent, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import {
  Activity,
  AlertTriangle,
  BadgePlus,
  CalendarClock,
  CheckCircle2,
  ClipboardList,
  Download,
  FileDown,
  FileText,
  FlaskConical,
  HeartPulse,
  Home,
  Image as ImageIcon,
  ListChecks,
  LogOut,
  MessageSquare,
  Mic,
  PanelLeft,
  PanelRight,
  Pill,
  Plus,
  RefreshCw,
  Search,
  Send,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  StopCircle,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  Upload,
  UserCheck,
  Users,
  X
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import {
  addAfterVisitNote,
  apiRequest,
  deleteCarePlan,
  deleteNote,
  downloadProtectedFile,
  emailNote,
  fetchAccessOverview,
  fetchClinicianPatient,
  fetchMyMedicationProposals,
  fetchMyPrevisitSummaries,
  fetchPrevisitChatHistory,
  fetchSnapshot,
  generateCarePlan,
  generateGpPrep,
  generateMedicationProposal,
  generateNote,
  generatePrevisitSummary,
  getConfig,
  getStoredToken,
  listCarePlans,
  login,
  rateResponse,
  releaseMedicationProposal,
  releasePrevisitSummary,
  requestPatientAccess,
  revokePatientAccess,
  saveMedicationProposalDraft,
  savePrevisitSummaryDraft,
  sendUrgentAlert,
  setStoredToken,
  signup,
  streamChat,
  streamDocumentAnalysis,
  streamImageAnalysis,
  streamPrevisitChat,
  toggleCarePlanTask,
  transcribeAudio,
  decidePatientAccess,
  updateNote,
  updateSafetyReview,
  uploadDocuments
} from "./api";
import type { AccessOverview, AuthResponse, CarePlan, CarePlanTask, ChatStreamEvent, ClinicalNote, ClinicianPatientSummary, Dict, EscalationThreshold, FeedbackRating, LabReminder, MedReminder, Message, MissedCareItem, PreVisitChatMessage, PreVisitSummary, PrevisitChatStreamEvent, ProductConfig, ProposedMedication, SafetyReview, Snapshot, TrialSearchResult } from "./types";
import type { ClarifyOption, UploadExtracted } from "./api";
import {
  buildSeries,
  buildSymptomSeries,
  buildTimelineEvents,
  buildTrendInsights,
  clean,
  formatDate,
  formatTimestamp,
  isClinicianRole,
  isPatientRole,
  parseMemorySections,
  unique,
  vitalLabel
} from "./utils";

type View = "workspace" | "safety" | "chat" | "timeline" | "trials" | "care-plans" | "patients" | "access";

class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  render() {
    if (this.state.error) {
      const msg = (this.state.error as Error).message;
      return (
        <main style={{ padding: "2rem", fontFamily: "sans-serif", color: "#122227", background: "#f7f9f7", minHeight: "100vh" }}>
          <h2 style={{ color: "#c0392b" }}>Something went wrong</h2>
          <p style={{ color: "#627174" }}>Please refresh the page. If this keeps happening, contact support.</p>
          <pre style={{ background: "#fff", border: "1px solid #d8e2df", borderRadius: 6, padding: "0.8rem", fontSize: "0.8rem", overflowX: "auto" }}>{msg}</pre>
        </main>
      );
    }
    return this.props.children;
  }
}

const PATIENT_URGENCY: Record<string, { label: string; color: string; bg: string }> = {
  routine:  { label: "Routine",             color: "var(--green)",  bg: "#edf8f4" },
  elevated: { label: "Worth monitoring",    color: "var(--gold)",   bg: "#fff8e5" },
  high:     { label: "See your GP soon",    color: "var(--accent)", bg: "#fff6f2" },
  urgent:   { label: "Seek urgent care",    color: "var(--danger)", bg: "#fff0ed" },
  crisis:   { label: "Call 999 / 911 now", color: "var(--danger)", bg: "#fff0ed" },
};

const STARTER_PROMPTS = [
  "What symptoms would make chest pain an urgent medical review issue?",
  "Summarize the most important themes from my uploaded records in plain language.",
  "What does the recent evidence say about hypertension treatment in older adults?"
];

const COUNTRIES = [
  "United Kingdom",
  "United States",
  "Ghana",
  "Nigeria",
  "Canada",
  "Australia",
  "Germany",
  "France",
  "Netherlands",
  "Ireland",
  "Italy",
  "Spain",
  "India",
  "South Africa",
  "Kenya",
  "Brazil",
  "Mexico",
  "Japan",
  "Singapore",
  "United Arab Emirates"
];

const VITAL_OPTIONS = [
  ["blood_pressure", "Blood pressure", "mmHg", "120/80"],
  ["heart_rate", "Heart rate", "bpm", "72"],
  ["weight", "Weight", "kg", "72"],
  ["blood_glucose", "Blood glucose", "mmol/L", "5.4"],
  ["oxygen_saturation", "Oxygen saturation", "%", "98"],
  ["temperature", "Temperature", "C", "37.0"],
  ["hba1c", "HbA1c", "mmol/mol", "48"],
  ["egfr", "eGFR", "mL/min/1.73m2", "65"]
];

const MAX_IMAGE_UPLOAD_BYTES = 5 * 1024 * 1024;
const MAX_DOCUMENT_UPLOAD_BYTES = 15 * 1024 * 1024;

function App() {
  const [config, setConfig] = useState<ProductConfig | null>(null);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [view, setView] = useState<View>("workspace");
  const [loading, setLoading] = useState(true);
  const [notice, setNotice] = useState("");

  useEffect(() => {
    let mounted = true;
    async function boot() {
      try {
        const loadedConfig = await getConfig();
        if (!mounted) {
          return;
        }
        setConfig(loadedConfig);
        if (getStoredToken()) {
          const loadedSnapshot = await fetchSnapshot();
          if (mounted) {
            setSnapshot(loadedSnapshot);
          }
        }
      } catch (error) {
        setNotice(error instanceof Error ? error.message : "Could not load the app.");
        setStoredToken("");
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    }
    boot();
    return () => {
      mounted = false;
    };
  }, []);

  const handleAuth = (response: AuthResponse) => {
    setStoredToken(response.token);
    setSnapshot(response.snapshot);
    setView("workspace");
    setNotice("");
  };

  const signOut = () => {
    setStoredToken("");
    setSnapshot(null);
    setView("workspace");
  };

  if (loading) {
    return (
      <main className="boot-screen">
        <img src="/assistant.png" alt="" />
        <span>Opening FlynnMed</span>
      </main>
    );
  }

  if (!config) {
    return <main className="boot-screen">Configuration is unavailable.</main>;
  }

  if (!snapshot) {
    return <AuthScreen config={config} onSuccess={handleAuth} />;
  }

  const role = snapshot.profile.clinical_role || snapshot.profile.role;
  const clinician = isClinicianRole(role);

  return (
    <ErrorBoundary>
      <Shell snapshot={snapshot} view={view} setView={setView} signOut={signOut} notice={notice}>
        {view === "workspace" && (
          clinician
            ? <ClinicianWorkspace snapshot={snapshot} setView={setView} setNotice={setNotice} />
            : <WorkspaceView snapshot={snapshot} setView={setView} setSnapshot={setSnapshot} />
        )}
        {view === "chat" && <ChatView snapshot={snapshot} setSnapshot={setSnapshot} setNotice={setNotice} />}
        {!clinician && view === "safety" && <SafetyReviewView snapshot={snapshot} setSnapshot={setSnapshot} setNotice={setNotice} />}
        {!clinician && view === "timeline" && <TimelineView snapshot={snapshot} />}
        {!clinician && view === "trials" && <TrialsView snapshot={snapshot} setSnapshot={setSnapshot} setNotice={setNotice} />}
        {!clinician && view === "care-plans" && <CarePlanScreen snapshot={snapshot} />}
        {!clinician && view === "access" && <PatientAccessView setNotice={setNotice} />}
        {clinician && view === "patients" && <ClinicianPatientsView setNotice={setNotice} />}
      </Shell>
    </ErrorBoundary>
  );
}

function Req() {
  return <span className="req" aria-label="required">*</span>;
}

function AuthScreen({ config, onSuccess }: { config: ProductConfig; onSuccess: (response: AuthResponse) => void }) {
  const [mode, setMode] = useState<"Sign in" | "Create account">("Sign in");
  const [role, setRole] = useState(config.role_options[0]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [loginForm, setLoginForm] = useState({ identifier: "", password: "" });
  const [signupForm, setSignupForm] = useState({
    full_name: "",
    email: "",
    username: "",
    organization: "",
    date_of_birth: "",
    biological_sex: "",
    password: "",
    confirm_password: "",
    accept_role_terms: false,
    accept_privacy: false
  });

  const terms = config.role_terms[role] ?? config.role_terms[config.role_options[0]];
  const isClinician = role !== "Patient" && role !== "Individual";

  function validateSignup(): string {
    const f = signupForm;
    if (!f.full_name.trim()) return "Full name is required.";
    if (f.full_name.trim().split(/\s+/).filter(Boolean).length < 2) return "Enter your first and last name.";
    if (!f.email.trim()) return "Email address is required.";
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(f.email.trim())) return "Enter a valid email address.";
    if (!f.username.trim()) return "Username is required.";
    if (f.username.trim().length < 3) return "Username must be at least 3 characters.";
    if (!f.password) return "Password is required.";
    if (f.password.length < 8) return "Password must be at least 8 characters.";
    if (f.password !== f.confirm_password) return "Passwords do not match.";
    if (!f.accept_role_terms) return "You must accept the role terms to continue.";
    if (!f.accept_privacy) return "You must accept the privacy notice to continue.";
    return "";
  }

  async function submitLogin(event: FormEvent) {
    event.preventDefault();
    const identifier = loginForm.identifier.trim().toLowerCase();
    if (!identifier || !loginForm.password) {
      setError("Enter your email or username and password.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      onSuccess(await login(identifier, loginForm.password));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Sign in failed.");
    } finally {
      setBusy(false);
    }
  }

  async function submitSignup(event: FormEvent) {
    event.preventDefault();
    const validationError = validateSignup();
    if (validationError) { setError(validationError); return; }
    setBusy(true);
    setError("");
    try {
      onSuccess(await signup({ ...signupForm, role }) as AuthResponse);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Account creation failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="auth-page">
      <section className="auth-intro">
        <div className="eyebrow">Secure health workspace</div>
        <h1>{config.product_name}</h1>
        <p>{config.product_tagline}</p>
        <ul className="auth-highlights">
          <li><ShieldCheck size={18} /> Secure, role-aware sign-in</li>
          <li><ClipboardList size={18} /> Terms tailored to your account role</li>
          <li><HeartPulse size={18} /> Conversation history saved across visits</li>
          <li><Sparkles size={18} /> Evidence-based support, every time</li>
        </ul>
      </section>

      <section className="auth-panel" aria-label="Account access">
        <img src="/logo.png" alt={config.product_name} className="auth-panel-logo" />
        <div className="segmented">
          {(["Sign in", "Create account"] as const).map((item) => (
            <button key={item} className={mode === item ? "active" : ""} onClick={() => { setMode(item); setError(""); }} type="button">
              {item}
            </button>
          ))}
        </div>

        {error && <div className="notice error">{error}</div>}

        {mode === "Sign in" ? (
          <form onSubmit={submitLogin} className="stack">
            <label>
              Email or username <Req />
              <input
                value={loginForm.identifier}
                onChange={(event) => setLoginForm({ ...loginForm, identifier: event.target.value })}
                autoComplete="username"
                placeholder="your@email.com or username"
              />
            </label>
            <label>
              Password <Req />
              <input
                type="password"
                value={loginForm.password}
                onChange={(event) => setLoginForm({ ...loginForm, password: event.target.value })}
                autoComplete="current-password"
                placeholder="Min. 8 characters"
              />
            </label>
            <button className="primary full" disabled={busy} type="submit">
              <ShieldCheck size={18} />
              {busy ? "Signing in..." : "Sign in"}
            </button>
            <p className="field-hint center">Not case-sensitive. Support: {config.support_email}</p>
          </form>
        ) : (
          <form onSubmit={submitSignup} className="stack">
            <label>
              Account role <Req />
              <select value={role} onChange={(event) => setRole(event.target.value)}>
                {config.role_options.map((option) => (
                  <option key={option}>{option}</option>
                ))}
              </select>
            </label>
            <div className="terms-panel">
              <strong>{terms.title}</strong>
              <p>{terms.summary}</p>
              <ul>
                {terms.bullets.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
            <p className="field-hint"><Req /> Required fields</p>
            <div className="form-grid">
              <label>
                Full name <Req />
                <input
                  value={signupForm.full_name}
                  onChange={(event) => setSignupForm({ ...signupForm, full_name: event.target.value })}
                  autoComplete="name"
                  placeholder="First and last name"
                />
              </label>
              <label>
                Email address <Req />
                <input
                  type="email"
                  value={signupForm.email}
                  onChange={(event) => setSignupForm({ ...signupForm, email: event.target.value })}
                  autoComplete="email"
                  placeholder="your@email.com"
                />
              </label>
              <label>
                Username <Req />
                <input
                  value={signupForm.username}
                  onChange={(event) => setSignupForm({ ...signupForm, username: event.target.value })}
                  autoComplete="username"
                  placeholder="At least 3 characters"
                />
              </label>
              <label>
                {isClinician ? "Organisation" : "Organisation"} <span className="optional-tag">Optional</span>
                <input
                  value={signupForm.organization}
                  onChange={(event) => setSignupForm({ ...signupForm, organization: event.target.value })}
                  placeholder={isClinician ? "Hospital or clinic name" : "Optional"}
                />
              </label>
              <label>
                Date of birth <span className="optional-tag">Optional</span>
                <input
                  type="date"
                  value={signupForm.date_of_birth}
                  onChange={(event) => setSignupForm({ ...signupForm, date_of_birth: event.target.value })}
                />
              </label>
              <label>
                Biological sex <span className="optional-tag">Optional</span>
                <select
                  value={signupForm.biological_sex}
                  onChange={(event) => setSignupForm({ ...signupForm, biological_sex: event.target.value })}
                >
                  <option value="">Prefer not to say</option>
                  <option>Male</option>
                  <option>Female</option>
                  <option>Other</option>
                </select>
              </label>
              <label>
                Password <Req />
                <input
                  type="password"
                  value={signupForm.password}
                  onChange={(event) => setSignupForm({ ...signupForm, password: event.target.value })}
                  autoComplete="new-password"
                  placeholder="Min. 8 characters"
                />
              </label>
              <label>
                Confirm password <Req />
                <input
                  type="password"
                  value={signupForm.confirm_password}
                  onChange={(event) => setSignupForm({ ...signupForm, confirm_password: event.target.value })}
                  autoComplete="new-password"
                  placeholder="Repeat your password"
                />
              </label>
            </div>
            <label className="check-row">
              <input
                type="checkbox"
                checked={signupForm.accept_role_terms}
                onChange={(event) => setSignupForm({ ...signupForm, accept_role_terms: event.target.checked })}
              />
              <span>{terms.acknowledgement} <Req /></span>
            </label>
            <label className="check-row">
              <input
                type="checkbox"
                checked={signupForm.accept_privacy}
                onChange={(event) => setSignupForm({ ...signupForm, accept_privacy: event.target.checked })}
              />
              <span>I have read the privacy notice and understand account support is available at {config.support_email}. <Req /></span>
            </label>
            <button className="primary full" disabled={busy} type="submit">
              <BadgePlus size={18} />
              {busy ? "Creating account..." : "Create account"}
            </button>
            <p className="field-hint center">A verification code will be emailed to you.</p>
          </form>
        )}
      </section>
    </main>
  );
}

function Shell({
  children,
  snapshot,
  view,
  setView,
  signOut,
  notice
}: {
  children: React.ReactNode;
  snapshot: Snapshot;
  view: View;
  setView: (view: View) => void;
  signOut: () => void;
  notice: string;
}) {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const role = snapshot.profile.clinical_role || snapshot.profile.role;
  const clinician = isClinicianRole(role);
  const nav: Array<{ id: View; label: string; short: string; icon: LucideIcon }> = clinician
    ? [
        { id: "workspace", label: "Clinical Home", short: "Home", icon: Home },
        { id: "patients", label: "My Patients", short: "Patients", icon: Users },
        { id: "chat", label: "Evidence Review", short: "Evidence", icon: MessageSquare }
      ]
    : [
        { id: "workspace", label: "Home", short: "Home", icon: Home },
        { id: "safety", label: "Safety Review", short: "Safety", icon: ShieldCheck },
        { id: "chat", label: "Health Chat", short: "Chat", icon: MessageSquare },
        { id: "care-plans", label: "My Care Plans", short: "Plans", icon: ListChecks },
        { id: "timeline", label: "My Timeline", short: "Timeline", icon: CalendarClock },
        { id: "trials", label: "Find Clinical Trials", short: "Trials", icon: FlaskConical },
        { id: "access", label: "Clinician Access", short: "Access", icon: UserCheck }
      ];
  const name = clean(snapshot.profile.display_name, snapshot.user);

  return (
    <div className={`app-shell${sidebarOpen ? "" : " sidebar-collapsed"}`}>
      <aside className="sidebar">
        <div className="brand-block">
          <img src="/assistant.png" alt="" />
          <div>
            <strong>{snapshot.product.name}</strong>
            <span>{clean(snapshot.profile.clinical_role || snapshot.profile.role, "Patient / Individual")}</span>
          </div>
        </div>
        <nav>
          {nav.map((item) => {
            const Icon = item.icon;
            return (
              <button key={item.id} className={view === item.id ? "active" : ""} onClick={() => setView(item.id)}>
                <Icon size={19} />
                {item.label}
              </button>
            );
          })}
        </nav>
        <div className="sidebar-footer">
          <button className="ghost sidebar-collapse-btn" onClick={() => setSidebarOpen(false)} title="Collapse sidebar">
            <PanelLeft size={18} />
            Collapse
          </button>
          <button className="ghost sidebar-signout" onClick={signOut}>
            <LogOut size={18} />
            Sign out
          </button>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            {!sidebarOpen && (
              <button className="sidebar-toggle" onClick={() => setSidebarOpen(true)} title="Expand sidebar" aria-label="Expand sidebar">
                <PanelLeft size={20} />
              </button>
            )}
            <div className="topbar-title">
              <span className="eyebrow">FlynnMed</span>
              <h1>{name}</h1>
            </div>
          </div>
          <button className="icon-button" onClick={signOut} title="Sign out" aria-label="Sign out">
            <LogOut size={20} />
          </button>
        </header>
        {notice && <div className="notice">{notice}</div>}
        {children}
      </main>

      <nav className="mobile-nav" aria-label="Primary navigation">
        {nav.map((item) => {
          const Icon = item.icon;
          return (
            <button key={item.id} className={view === item.id ? "active" : ""} onClick={() => setView(item.id)} aria-current={view === item.id ? "page" : undefined}>
              <Icon size={19} />
              <span>{item.short}</span>
            </button>
          );
        })}
      </nav>
    </div>
  );
}

function WorkspaceView({
  snapshot,
  setView,
  setSnapshot
}: {
  snapshot: Snapshot;
  setView: (view: View) => void;
  setSnapshot: (snapshot: Snapshot) => void;
}) {
  const metrics: Array<[string, number, LucideIcon]> = [
    ["Documents", snapshot.metrics.documents, FileText],
    ["Symptoms", snapshot.metrics.symptoms, Activity],
    ["Conditions", snapshot.metrics.conditions, Stethoscope],
    ["Medications", snapshot.metrics.medications, Pill],
    ["Vitals/labs", snapshot.metrics.vitals, HeartPulse]
  ];

  return (
    <div className="view-stack">
      <section className="workspace-band">
        <div>
          <span className="eyebrow">Workspace</span>
          <h2>Choose where to work today.</h2>
          <p>Chat, review the longitudinal record, or search recruiting studies with saved account context.</p>
        </div>
        {snapshot.latest_triage?.next_step && (
          <div className="triage-strip">
            <strong>{clean(snapshot.latest_triage.urgency_level, "Routine")}</strong>
            <span>{clean(snapshot.latest_triage.next_step, "Self-care")}</span>
          </div>
        )}
      </section>

      <section className="metric-grid">
        {metrics.map(([label, value, Icon]) => (
          <div className="metric-card" key={label}>
            <Icon size={20} />
            <strong>{value}</strong>
            <span>{label}</span>
          </div>
        ))}
      </section>

      <section className="action-grid">
        <ActionButton
          icon={<ShieldCheck size={21} />}
          title="Safety review"
          body="Review important changes in results, symptoms and medicines, with the facts and guidance behind each proposed next step."
          onClick={() => setView("safety")}
        />
        <ActionButton
          icon={<MessageSquare size={21} />}
          title="Chat"
          body="Ask a health question, clarify missing information, or add a new record."
          onClick={() => setView("chat")}
        />
        <ActionButton
          icon={<CalendarClock size={21} />}
          title="Health Timeline"
          body="Review conditions, medications, allergies, readings, uploaded records, and trend cards."
          onClick={() => setView("timeline")}
        />
      </section>

      <RecordPanel snapshot={snapshot} setSnapshot={setSnapshot} compact />
    </div>
  );
}

function SafetyReviewView({
  snapshot,
  setSnapshot,
  setNotice
}: {
  snapshot: Snapshot;
  setSnapshot: (snapshot: Snapshot) => void;
  setNotice: (notice: string) => void;
}) {
  const [busy, setBusy] = useState("");
  const [followUp, setFollowUp] = useState<Record<string, { happened: boolean | null; improved: boolean | null; note: string }>>({});
  const reviews = snapshot.safety_reviews ?? [];
  const emergencyCount = reviews.filter((item) => item.priority === "emergency").length;

  async function save(review: SafetyReview, payload: Parameters<typeof updateSafetyReview>[1]) {
    setBusy(review.review_id);
    try {
      const response = await updateSafetyReview(review.review_id, payload);
      setSnapshot(response.snapshot);
      setNotice(payload.status === "patient_confirmed"
        ? "Your confirmation is saved. A qualified clinician still needs to review the proposal."
        : "Follow-up saved.");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not update the safety review.");
    } finally {
      setBusy("");
    }
  }

  return (
    <div className="view-stack safety-review-page">
      <section className="workspace-band safety-heading">
        <div>
          <span className="eyebrow">Evidence to action</span>
          <h2>Safety review</h2>
          <p>Important changes found across your results, symptoms, medicines and allergies.</p>
        </div>
        <div className={`safety-count${emergencyCount ? " emergency" : ""}`}>
          <strong>{reviews.length}</strong>
          <span>{reviews.length === 1 ? "active review" : "active reviews"}</span>
        </div>
      </section>

      {emergencyCount > 0 && (
        <section className="emergency-banner" role="alert">
          <AlertTriangle size={24} />
          <div>
            <strong>Emergency action comes first</strong>
            <p>Follow the emergency instruction below now. Do not wait for confirmation or clinician approval in FlynnMed.</p>
          </div>
        </section>
      )}

      {reviews.length === 0 ? (
        <section className="safety-empty">
          <CheckCircle2 size={26} />
          <div>
            <h3>No supported safety conflict is active</h3>
            <p>FlynnMed only shows a finding when a locked rule can link it to exact record facts and guidance. This is not a complete clinical review.</p>
          </div>
        </section>
      ) : reviews.map((review) => {
        const draft = followUp[review.review_id] ?? {
          happened: review.outcome.action_happened,
          improved: review.outcome.patient_improved,
          note: review.outcome.note || ""
        };
        return (
          <article className={`safety-review safety-${review.priority}`} key={review.review_id}>
            <header className="safety-review-header">
              <div>
                <span className="safety-priority">{review.priority}</span>
                <h3>{review.category}</h3>
              </div>
              <span className="review-status">{review.status.replace(/_/g, " ")}</span>
            </header>

            <div className="safety-flow">
              <section>
                <span>What changed</span>
                <p>{review.what_changed}</p>
              </section>
              <section>
                <span>Why it matters</span>
                <p>{review.why_it_matters}</p>
              </section>
              <section className="proposed-action">
                <span>Proposed next step</span>
                <p>{review.proposed_action}</p>
              </section>
            </div>

            <div className="traceability-grid">
              <section>
                <h4>Patient facts used</h4>
                {review.patient_facts.map((fact) => (
                  <div className="trace-row" key={`${fact.record_type}-${fact.record_id}`}>
                    <strong>{fact.label}</strong>
                    <span>{fact.value}{fact.recorded_on ? `, ${formatDate(fact.recorded_on)}` : ""}</span>
                  </div>
                ))}
              </section>
              <section>
                <h4>Guidance used</h4>
                {review.evidence.map((source) => (
                  <div className="trace-row" key={source.source_url}>
                    <a href={source.source_url} target="_blank" rel="noreferrer">{source.source_title}</a>
                    <span>{source.passage}</span>
                  </div>
                ))}
              </section>
            </div>

            <details className="uncertainty-panel">
              <summary>Uncertainty and limits</summary>
              <p>{review.uncertainty}</p>
            </details>

            <section className="approval-strip">
              <div>
                <span>Approver</span>
                <strong>{review.approver}</strong>
                <small>{review.writeback.message}</small>
              </div>
              {review.status === "detected" && review.priority !== "emergency" && (
                <button className="primary" disabled={busy === review.review_id} onClick={() => save(review, { status: "patient_confirmed" })}>
                  <CheckCircle2 size={17} />
                  Confirm for review
                </button>
              )}
            </section>

            {(review.status !== "detected" || review.priority === "emergency") && (
              <section className="follow-up-panel">
                <h4>Follow-up</h4>
                <div className="follow-up-fields">
                  <fieldset>
                    <legend>Did the action happen?</legend>
                    <div className="segmented compact-segmented">
                      <button type="button" className={draft.happened === true ? "active" : ""} onClick={() => setFollowUp({ ...followUp, [review.review_id]: { ...draft, happened: true } })}>Yes</button>
                      <button type="button" className={draft.happened === false ? "active" : ""} onClick={() => setFollowUp({ ...followUp, [review.review_id]: { ...draft, happened: false } })}>No</button>
                    </div>
                  </fieldset>
                  <fieldset>
                    <legend>Are you feeling better?</legend>
                    <div className="segmented compact-segmented">
                      <button type="button" className={draft.improved === true ? "active" : ""} onClick={() => setFollowUp({ ...followUp, [review.review_id]: { ...draft, improved: true } })}>Yes</button>
                      <button type="button" className={draft.improved === false ? "active" : ""} onClick={() => setFollowUp({ ...followUp, [review.review_id]: { ...draft, improved: false } })}>No</button>
                    </div>
                  </fieldset>
                </div>
                <label>
                  Note for the clinician <span className="optional-tag">Optional</span>
                  <textarea rows={3} value={draft.note} maxLength={500} onChange={(event) => setFollowUp({ ...followUp, [review.review_id]: { ...draft, note: event.target.value } })} />
                </label>
                <button className="ghost" disabled={busy === review.review_id || draft.happened === null} onClick={() => save(review, {
                  status: "follow_up_recorded",
                  action_happened: draft.happened ?? undefined,
                  patient_improved: draft.improved ?? undefined,
                  note: draft.note
                })}>
                  Save follow-up
                </button>
              </section>
            )}
          </article>
        );
      })}
    </div>
  );
}

function ClinicianWorkspace({
  snapshot,
  setView,
  setNotice
}: {
  snapshot: Snapshot;
  setView: (view: View) => void;
  setNotice: (notice: string) => void;
}) {
  const [access, setAccess] = useState<AccessOverview | null>(null);

  useEffect(() => {
    fetchAccessOverview()
      .then(setAccess)
      .catch((error) => setNotice(error instanceof Error ? error.message : "Could not load patient access."));
  }, [setNotice]);

  return (
    <div className="view-stack clinician-dashboard">
      <section className="workspace-band clinician-band">
        <div>
          <span className="eyebrow">Clinical workspace</span>
          <h2>Welcome, {clean(snapshot.profile.display_name, "clinician")}</h2>
          <p>
            Review consented patient records and use evidence support within your professional scope.
            Patient data is never opened without an active access grant.
          </p>
        </div>
        <div className="role-badge">
          <Stethoscope size={18} />
          {clean(snapshot.profile.clinical_role || snapshot.profile.role, "Clinician")}
        </div>
      </section>

      <section className="metric-grid">
        <div className="metric-card">
          <Users size={20} />
          <strong>{access?.active_count ?? 0}</strong>
          <span>Consented patients</span>
        </div>
        <div className="metric-card">
          <UserCheck size={20} />
          <strong>{access?.pending_count ?? 0}</strong>
          <span>Pending requests</span>
        </div>
        <div className="metric-card">
          <ShieldCheck size={20} />
          <strong>Scoped</strong>
          <span>Consent-controlled access</span>
        </div>
      </section>

      <section className="action-grid clinician-actions">
        <ActionButton
          icon={<Users size={21} />}
          title="Open patient list"
          body="Request access by MRN or review records for patients who have approved access."
          onClick={() => setView("patients")}
        />
        <ActionButton
          icon={<MessageSquare size={21} />}
          title="Clinical evidence review"
          body="Ask a professional evidence question without treating your own clinician account as a patient record."
          onClick={() => setView("chat")}
        />
      </section>

      <section className="surface-card boundary-card">
        <ShieldCheck size={22} />
        <div>
          <h3>Professional boundary</h3>
          <p>
            Personal care plans, personal timelines, and patient trial matching belong to patient
            workspaces. In the clinician workspace, those records appear only inside an approved
            patient chart.
          </p>
        </div>
      </section>
    </div>
  );
}

function PatientAccessView({ setNotice }: { setNotice: (notice: string) => void }) {
  const [overview, setOverview] = useState<AccessOverview | null>(null);
  const [busy, setBusy] = useState("");
  const [loadError, setLoadError] = useState("");

  async function refresh() {
    try {
      setOverview(await fetchAccessOverview());
      setLoadError("");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not load access requests.";
      setLoadError(message);
      setNotice(message);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function decide(grantId: string, approve: boolean) {
    setBusy(grantId);
    try {
      await decidePatientAccess(grantId, approve);
      setNotice(approve ? "Clinician access approved for 90 days." : "Access request denied.");
      await refresh();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not update access.");
    } finally {
      setBusy("");
    }
  }

  async function revoke(grantId: string) {
    setBusy(grantId);
    try {
      await revokePatientAccess(grantId);
      setNotice("Clinician access revoked.");
      await refresh();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not revoke access.");
    } finally {
      setBusy("");
    }
  }

  return (
    <div className="view-stack">
      <section className="workspace-band">
        <div>
          <span className="eyebrow">Privacy and consent</span>
          <h2>Clinician access</h2>
          <p>Only approve a request from a clinician you recognise and expect to review your record.</p>
        </div>
        <div className="mrn-card">
          <span>Your MRN</span>
          <strong>{loadError ? "Database not connected" : overview?.patient_id || "Loading..."}</strong>
          <small>{loadError ? "Your MRN will appear when clinician access is available." : "Share this privately with your clinician."}</small>
        </div>
      </section>

      {loadError && (
        <section className="access-unavailable" role="alert">
          <AlertTriangle size={20} />
          <div>
            <strong>Clinician access is temporarily unavailable</strong>
            <p>{loadError}</p>
          </div>
        </section>
      )}

      <section className="surface-card">
        <h3>Pending requests</h3>
        {!loadError && !overview?.requests.some((item) => item.status === "pending") && (
          <p className="muted">No clinician is waiting for access.</p>
        )}
        <div className="access-list">
          {overview?.requests.filter((item) => item.status === "pending").map((item) => (
            <article className="access-row" key={item.grant_id}>
              <div>
                <strong>{item.clinician_name}</strong>
                <span>{item.clinician_role}{item.organization ? ` · ${item.organization}` : ""}</span>
                <p>{item.request_reason || "No reason supplied."}</p>
                <small>
                  Requested: record summary
                  {item.scopes.includes("chat_history") ? " and chat history" : ""}
                </small>
              </div>
              <div className="row-actions">
                <button className="primary" disabled={busy === item.grant_id} onClick={() => decide(item.grant_id, true)}>
                  Approve
                </button>
                <button className="ghost" disabled={busy === item.grant_id} onClick={() => decide(item.grant_id, false)}>
                  Deny
                </button>
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="surface-card">
        <h3>Active access</h3>
        {!loadError && !overview?.requests.some((item) => item.status === "active") && (
          <p className="muted">No clinician currently has access.</p>
        )}
        <div className="access-list">
          {overview?.requests.filter((item) => item.status === "active").map((item) => (
            <article className="access-row" key={item.grant_id}>
              <div>
                <strong>{item.clinician_name}</strong>
                <span>{item.clinician_role}{item.organization ? ` · ${item.organization}` : ""}</span>
                <small>Expires {formatDate(item.expires_at)}</small>
              </div>
              <button className="ghost danger-button" disabled={busy === item.grant_id} onClick={() => revoke(item.grant_id)}>
                Revoke
              </button>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

function ClinicianPatientsView({ setNotice }: { setNotice: (notice: string) => void }) {
  const [overview, setOverview] = useState<AccessOverview | null>(null);
  const [selected, setSelected] = useState<ClinicianPatientSummary | null>(null);
  const [mrn, setMrn] = useState("");
  const [reason, setReason] = useState("");
  const [includeChat, setIncludeChat] = useState(false);
  const [busy, setBusy] = useState(false);

  async function refresh() {
    try {
      setOverview(await fetchAccessOverview());
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not load patient access.");
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function submitRequest(event: FormEvent) {
    event.preventDefault();
    if (!mrn.trim()) {
      setNotice("Enter the patient MRN.");
      return;
    }
    if (!reason.trim()) {
      setNotice("Enter the clinical reason for requesting access.");
      return;
    }
    setBusy(true);
    try {
      await requestPatientAccess({
        patient_id: mrn.trim(),
        reason: reason.trim(),
        include_chat_history: includeChat
      });
      setNotice("Access request sent. The patient must approve it before any record is shown.");
      setMrn("");
      setReason("");
      setIncludeChat(false);
      await refresh();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not request access.");
    } finally {
      setBusy(false);
    }
  }

  async function openPatient(patientId: string) {
    setBusy(true);
    try {
      setSelected(await fetchClinicianPatient(patientId));
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not open patient record.");
    } finally {
      setBusy(false);
    }
  }

  async function revoke(grantId: string) {
    setBusy(true);
    try {
      await revokePatientAccess(grantId);
      setSelected(null);
      setNotice("Patient access released.");
      await refresh();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not release access.");
    } finally {
      setBusy(false);
    }
  }

  if (selected) {
    return (
      <ClinicianPatientChart
        summary={selected}
        busy={busy}
        onBack={() => setSelected(null)}
        onRevoke={() => revoke(selected.grant.grant_id)}
      />
    );
  }

  const active = overview?.requests.filter((item) => item.status === "active") ?? [];
  const pending = overview?.requests.filter((item) => item.status === "pending") ?? [];
  return (
    <div className="view-stack">
      <section className="workspace-band clinician-band">
        <div>
          <span className="eyebrow">Consented care</span>
          <h2>My patients</h2>
          <p>Open approved records or request access using the MRN provided by the patient.</p>
        </div>
      </section>

      <section className="surface-card">
        <h3>Approved patients</h3>
        {!active.length && <p className="muted">No patient has approved access yet.</p>}
        <div className="patient-grid">
          {active.map((item) => (
            <button className="patient-card" key={item.grant_id} onClick={() => openPatient(item.patient_id)} disabled={busy}>
              <UserCheck size={22} />
              <strong>{item.patient_name}</strong>
              <span>{item.patient_id}</span>
              <small>Access until {formatDate(item.expires_at)}</small>
            </button>
          ))}
        </div>
      </section>

      <section className="surface-card access-request-card">
        <h3>Request patient access</h3>
        <form className="stack" onSubmit={submitRequest}>
          <label>
            Patient MRN
            <input value={mrn} onChange={(event) => setMrn(event.target.value)} placeholder="FM-XXXX-XXXX" />
          </label>
          <label>
            Clinical reason
            <textarea value={reason} onChange={(event) => setReason(event.target.value)} placeholder="Reason for reviewing this record" rows={3} />
          </label>
          <label className="check-row">
            <input type="checkbox" checked={includeChat} onChange={(event) => setIncludeChat(event.target.checked)} />
            <span>Also request access to conversation history</span>
          </label>
          <button className="primary" type="submit" disabled={busy}>Send consent request</button>
        </form>
      </section>

      {!!pending.length && (
        <section className="surface-card">
          <h3>Waiting for patient approval</h3>
          <div className="access-list">
            {pending.map((item) => (
              <article className="access-row" key={item.grant_id}>
                <div>
                  <strong>{item.patient_name}</strong>
                  <span>{item.patient_id}</span>
                  <small>Requested {formatDate(item.requested_at)}</small>
                </div>
              </article>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function ClinicianPatientChart({
  summary,
  busy,
  onBack,
  onRevoke
}: {
  summary: ClinicianPatientSummary;
  busy: boolean;
  onBack: () => void;
  onRevoke: () => void;
}) {
  return (
    <div className="view-stack patient-chart">
      <section className="workspace-band clinician-band">
        <div>
          <button className="link-btn" onClick={onBack}>← Back to patients</button>
          <span className="eyebrow">Consented patient record</span>
          <h2>{summary.patient.display_name}</h2>
          <p>
            {summary.patient.patient_id}
            {summary.patient.date_of_birth ? ` · DOB ${formatDate(summary.patient.date_of_birth)}` : ""}
            {summary.patient.biological_sex ? ` · ${summary.patient.biological_sex}` : ""}
          </p>
        </div>
        <button className="ghost danger-button" onClick={onRevoke} disabled={busy}>Release access</button>
      </section>

      <PrevisitWorkspace
        patientId={summary.patient.patient_id}
        initialSummaries={summary.previsit_summaries}
      />

      <MedicationProposalWorkspace
        patientId={summary.patient.patient_id}
        initialProposals={summary.proposed_medications}
      />

      <section className="chart-grid">
        <ClinicalDataList title="Active conditions" items={summary.conditions} primary="name" secondary="status" />
        <ClinicalDataList title="Current medications" items={summary.medications} primary="name" secondary="dose" />
        <ClinicalDataList title="Allergies" items={summary.allergies} primary="name" secondary="reaction" />
        <ClinicalDataList title="Recent observations" items={summary.vitals} primary="type" secondary="value" suffix="unit" />
        <ClinicalDataList title="Recent symptoms" items={summary.symptoms} primary="symptom" secondary="severity" />
        <ClinicalDataList title="Triage and disposition" items={summary.triage} primary="urgency_level" secondary="next_step" />
        <ClinicalDataList title="Patient care plans" items={summary.care_plans} primary="title" secondary="status" />
        <ClinicalDataList title="Clinical notes" items={summary.clinical_notes} primary="assessment" secondary="plan" />
        <ClinicalDataList
          title="Clinical relationships"
          items={(summary.clinical_relationships ?? []).map((item: Dict<any>) => ({
            ...item,
            relationship: `${item.relation ?? "associated with"} ${item.target_name ?? ""}`,
          }))}
          primary="source_name"
          secondary="relationship"
        />
      </section>

      <section className="surface-card">
        <h3>Conversation history</h3>
        {!summary.chat_history_authorized ? (
          <p className="muted">The patient did not grant access to conversation history.</p>
        ) : (
          <ClinicalDataList title="" items={summary.chat_history} primary="role" secondary="content" secondaryAsMarkdown />
        )}
      </section>
    </div>
  );
}

/**
 * One consolidated pre-visit workspace: the AI-suggested chart summary, an
 * inline chat scoped to this same patient so the clinician never has to
 * leave the page to ask a follow-up question, and the draft-edit/release
 * controls -- the chat is a tool for building the note, not a separate
 * destination. Local state here (not lifted to the parent) since every
 * action mutates server state the parent's already-fetched summary prop
 * doesn't reflect; re-fetching the whole chart after every keystroke/turn
 * would be wasteful.
 */
function PrevisitWorkspace({
  patientId,
  initialSummaries
}: {
  patientId: string;
  initialSummaries: PreVisitSummary[];
}) {
  const [summaries, setSummaries] = useState<PreVisitSummary[]>(initialSummaries);
  const [draftText, setDraftText] = useState(() => summaries.find((s) => s.status === "draft")?.summary_text ?? "");
  const [chatMessages, setChatMessages] = useState<PreVisitChatMessage[]>([]);
  const [chatDraft, setChatDraft] = useState("");
  const [chatStatus, setChatStatus] = useState("");
  const [chatStreamText, setChatStreamText] = useState("");
  const [chatBusy, setChatBusy] = useState(false);
  const [summaryBusy, setSummaryBusy] = useState(false);
  const [releaseBusy, setReleaseBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [historyOpen, setHistoryOpen] = useState(false);
  const chatListRef = useRef<HTMLDivElement>(null);

  const latestDraft = summaries.find((s) => s.status === "draft") ?? null;
  const releasedSummaries = summaries.filter((s) => s.status === "released");

  useEffect(() => {
    fetchPrevisitChatHistory(patientId)
      .then((res) => setChatMessages(res.messages))
      .catch(() => {
        // No transcript yet, or a transient load failure -- the chat starts
        // empty either way, nothing to surface to the clinician here.
      });
  }, [patientId]);

  useEffect(() => {
    chatListRef.current?.scrollTo({ top: chatListRef.current.scrollHeight, behavior: "smooth" });
  }, [chatMessages, chatStreamText, chatStatus]);

  async function handleGenerate() {
    setSummaryBusy(true);
    setNotice("");
    try {
      const result = await generatePrevisitSummary(patientId);
      setSummaries((current) => [result, ...current]);
      setDraftText(result.summary_text);
    } catch (err) {
      setNotice(err instanceof Error ? err.message : "Could not generate the summary.");
    } finally {
      setSummaryBusy(false);
    }
  }

  async function handleSaveDraft() {
    setSummaryBusy(true);
    setNotice("");
    try {
      const result = await savePrevisitSummaryDraft(patientId, draftText);
      setSummaries((current) => [result, ...current]);
    } catch (err) {
      setNotice(err instanceof Error ? err.message : "Could not save the draft.");
    } finally {
      setSummaryBusy(false);
    }
  }

  async function handleRelease() {
    if (!draftText.trim()) {
      setNotice("Write or generate a summary before releasing it to the patient.");
      return;
    }
    const confirmed = window.confirm(
      "Release this summary to the patient's portal now? This cannot be undone -- the patient will see it immediately."
    );
    if (!confirmed) return;

    setReleaseBusy(true);
    setNotice("");
    try {
      const result = await releasePrevisitSummary(patientId, draftText);
      setSummaries((current) => [result, ...current]);
      setNotice("Summary released to the patient's portal.");
    } catch (err) {
      setNotice(err instanceof Error ? err.message : "Could not release the summary.");
    } finally {
      setReleaseBusy(false);
    }
  }

  async function handleSendChat(overrideText?: string) {
    const message = (overrideText ?? chatDraft).trim();
    if (!message || chatBusy) return;
    setChatDraft("");
    setChatBusy(true);
    setChatStatus("Reviewing this patient's record...");
    setChatStreamText("");
    try {
      await streamPrevisitChat(patientId, message, (event: PrevisitChatStreamEvent) => {
        if (event.type === "user_message") {
          setChatMessages((current) => [
            ...current,
            {
              id: `local-${Date.now()}`,
              role: "clinician",
              content: event.message.content,
              authored_by_display_name: "You",
              authored_by_clinical_role: "",
              created_at: event.message.timestamp
            }
          ]);
        }
        if (event.type === "status") setChatStatus(event.message);
        if (event.type === "token") setChatStreamText((current) => current + event.delta);
        if (event.type === "assistant_message") {
          setChatMessages((current) => [
            ...current,
            {
              id: `local-${Date.now()}-a`,
              role: "assistant",
              content: event.message.content,
              authored_by_display_name: "FlynnMed",
              authored_by_clinical_role: "",
              sources: event.message.sources,
              follow_up_questions: event.message.follow_up_questions,
              created_at: new Date().toISOString()
            }
          ]);
          setChatStreamText("");
          setChatStatus("");
        }
        if (event.type === "error") {
          setNotice(event.message);
          setChatStreamText("");
          setChatStatus("");
        }
      });
    } catch (err) {
      setNotice(err instanceof Error ? err.message : "Message failed.");
    } finally {
      setChatBusy(false);
      setChatStatus("");
    }
  }

  return (
    <section className="surface-card previsit-workspace">
      <h3>Pre-visit summary</h3>
      {notice && <p className="notice warn">{notice}</p>}

      {!latestDraft && !draftText ? (
        <div className="previsit-empty">
          <p className="muted">No summary drafted yet for this patient.</p>
          <button className="primary" onClick={handleGenerate} disabled={summaryBusy}>
            <Sparkles size={16} /> {summaryBusy ? "Generating..." : "Generate summary"}
          </button>
        </div>
      ) : (
        <>
          <label className="col-label">Preview</label>
          <div className="markdown previsit-draft-preview">
            <ReactMarkdown>{draftText || "*Nothing to preview yet.*"}</ReactMarkdown>
          </div>
          <label className="col-label">Edit summary (markdown source)</label>
          <textarea
            className="previsit-draft"
            rows={10}
            value={draftText}
            onChange={(event) => setDraftText(event.target.value)}
            placeholder="The AI-suggested summary will appear here -- edit freely before releasing."
          />
          <div className="previsit-actions">
            <button className="ghost" onClick={handleGenerate} disabled={summaryBusy}>
              <RefreshCw size={16} /> {summaryBusy ? "Working..." : "Regenerate suggested summary"}
            </button>
            <button className="ghost" onClick={handleSaveDraft} disabled={summaryBusy || !draftText.trim()}>
              Save draft
            </button>
            <button className="primary" onClick={handleRelease} disabled={releaseBusy || !draftText.trim()}>
              <ShieldCheck size={16} /> {releaseBusy ? "Releasing..." : "Release to patient"}
            </button>
          </div>
        </>
      )}

      <div className="previsit-chat">
        <p className="col-label">Ask about this patient</p>
        <div className="previsit-chat-list" ref={chatListRef}>
          {!chatMessages.length && !chatStreamText && (
            <p className="muted">
              Ask a follow-up question about this patient's record -- e.g. "what were the last three BP readings?"
            </p>
          )}
          {chatMessages.map((message) => (
            <div key={message.id} className={`previsit-turn previsit-turn-${message.role}`}>
              <span className="turn-label">{message.role === "clinician" ? "You" : "FlynnMed"}</span>
              <div className="markdown">
                <ReactMarkdown>{message.content}</ReactMarkdown>
              </div>
              {message.role === "assistant" && !!message.follow_up_questions?.length && (
                <div className="follow-ups">
                  {message.follow_up_questions.slice(0, 3).map((q, i) => {
                    const display = typeof q === "string" ? q : q.display;
                    const prompt = typeof q === "string" ? q : q.prompt;
                    return (
                      <button key={i} className="follow-up-btn" onClick={() => handleSendChat(prompt)}>
                        {display}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          ))}
          {chatStreamText && (
            <div className="previsit-turn previsit-turn-assistant">
              <span className="turn-label">FlynnMed</span>
              <div className="markdown">
                <ReactMarkdown>{chatStreamText}</ReactMarkdown>
              </div>
            </div>
          )}
          {chatStatus && <p className="muted previsit-status">{chatStatus}</p>}
        </div>
        <div className="previsit-chat-input">
          <input
            value={chatDraft}
            onChange={(event) => setChatDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                handleSendChat();
              }
            }}
            placeholder="Ask a question about this patient..."
            disabled={chatBusy}
          />
          <button className="primary" onClick={() => handleSendChat()} disabled={chatBusy || !chatDraft.trim()}>
            <Send size={16} />
          </button>
        </div>
      </div>

      {!!releasedSummaries.length && (
        <div className="previsit-history">
          <button className="link-btn" onClick={() => setHistoryOpen((open) => !open)}>
            {historyOpen ? "Hide" : "Show"} {releasedSummaries.length} past released summary
            {releasedSummaries.length === 1 ? "" : "ies"}
          </button>
          {historyOpen &&
            releasedSummaries.map((item) => (
              <article key={item.id} className="previsit-history-item">
                <p className="muted">
                  Released {formatTimestamp(item.released_at)} by {item.authored_by_display_name}
                  {item.authored_by_clinical_role ? ` (${item.authored_by_clinical_role})` : ""}
                </p>
                <div className="markdown">
                  <ReactMarkdown>{item.summary_text}</ReactMarkdown>
                </div>
              </article>
            ))}
        </div>
      )}
    </section>
  );
}

function medicationProposalHasFlags(proposal: Pick<ProposedMedication, "safety_check">): boolean {
  const check = proposal.safety_check;
  if (!check) return false;
  if (check.allergy_flags?.length) return true;
  return (check.interaction_flags ?? []).some(
    (flag) => flag.severity === "high" || flag.severity === "monitor"
  );
}

/**
 * Clinician-only medication-proposal workspace: describe a clinical
 * situation, get a real evidence-grounded candidate drug + dose back
 * (never the model's own general knowledge -- see backend/medication_proposal.py),
 * deterministically checked against this patient's actual allergies/current
 * medications. Nothing here is visible to the patient until an explicit
 * release, same gate as PrevisitWorkspace above.
 */
function MedicationProposalWorkspace({
  patientId,
  initialProposals
}: {
  patientId: string;
  initialProposals: ProposedMedication[];
}) {
  const [proposals, setProposals] = useState<ProposedMedication[]>(initialProposals);
  const latestDraft = proposals.find((p) => p.status === "draft") ?? null;
  const releasedProposals = proposals.filter((p) => p.status === "released");

  const [situation, setSituation] = useState("");
  const [medicationName, setMedicationName] = useState(latestDraft?.candidate_medication_name ?? "");
  const [doseFrequency, setDoseFrequency] = useState(latestDraft?.candidate_dose_frequency ?? "");
  const [rationale, setRationale] = useState(latestDraft?.rationale_text ?? "");
  const [overrideReason, setOverrideReason] = useState("");
  const [generateBusy, setGenerateBusy] = useState(false);
  const [saveBusy, setSaveBusy] = useState(false);
  const [releaseBusy, setReleaseBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [historyOpen, setHistoryOpen] = useState(false);

  const activeDraft = proposals.find((p) => p.status === "draft") ?? null;
  const hasFlags = activeDraft ? medicationProposalHasFlags(activeDraft) : false;

  async function handleGenerate() {
    if (!situation.trim()) {
      setNotice("Describe the clinical situation before generating a proposal.");
      return;
    }
    setGenerateBusy(true);
    setNotice("");
    try {
      const result = await generateMedicationProposal(patientId, situation.trim());
      setProposals((current) => [result, ...current]);
      setMedicationName(result.candidate_medication_name);
      setDoseFrequency(result.candidate_dose_frequency);
      setRationale(result.rationale_text);
      setOverrideReason("");
    } catch (err) {
      setNotice(err instanceof Error ? err.message : "Could not generate a medication proposal.");
    } finally {
      setGenerateBusy(false);
    }
  }

  async function handleSaveDraft() {
    if (!activeDraft) return;
    setSaveBusy(true);
    setNotice("");
    try {
      const result = await saveMedicationProposalDraft(patientId, {
        clinical_situation_text: activeDraft.clinical_situation_text,
        candidate_medication_name: medicationName.trim(),
        candidate_dose_frequency: doseFrequency.trim(),
        rationale_text: rationale,
        citations: activeDraft.citations
      });
      setProposals((current) => [result, ...current]);
      setNotice("Draft saved -- safety check re-run against the current candidate.");
    } catch (err) {
      setNotice(err instanceof Error ? err.message : "Could not save the draft.");
    } finally {
      setSaveBusy(false);
    }
  }

  async function handleRelease() {
    if (!activeDraft) return;
    if (!medicationName.trim() || !doseFrequency.trim()) {
      setNotice("Enter a candidate medication and dose/frequency before releasing.");
      return;
    }
    if (hasFlags && !overrideReason.trim()) {
      setNotice("This candidate has an unresolved allergy or interaction flag -- enter an override reason to release it anyway.");
      return;
    }
    const confirmed = window.confirm(
      "Release this medication proposal to the patient's portal now? This cannot be undone -- the patient will see it immediately."
    );
    if (!confirmed) return;

    setReleaseBusy(true);
    setNotice("");
    try {
      const result = await releaseMedicationProposal(patientId, {
        clinical_situation_text: activeDraft.clinical_situation_text,
        candidate_medication_name: medicationName.trim(),
        candidate_dose_frequency: doseFrequency.trim(),
        rationale_text: rationale,
        citations: activeDraft.citations,
        override_reason: overrideReason.trim()
      });
      setProposals((current) => [result, ...current]);
      setOverrideReason("");
      setNotice("Medication proposal released to the patient's portal.");
    } catch (err) {
      setNotice(err instanceof Error ? err.message : "Could not release the medication proposal.");
    } finally {
      setReleaseBusy(false);
    }
  }

  return (
    <section className="surface-card previsit-workspace">
      <h3>Medication proposal</h3>
      {notice && <p className="notice warn">{notice}</p>}

      {!activeDraft ? (
        <div className="previsit-empty">
          <textarea
            className="previsit-draft"
            rows={4}
            value={situation}
            onChange={(event) => setSituation(event.target.value)}
            placeholder="Describe the clinical situation -- e.g. 'Moderate postpartum depression, 8 weeks postpartum, no response to talk therapy after 6 weeks, no contraindications noted.'"
          />
          <button className="primary" onClick={handleGenerate} disabled={generateBusy}>
            <Pill size={16} /> {generateBusy ? "Researching..." : "Generate proposal"}
          </button>
        </div>
      ) : (
        <>
          <p className="muted">{activeDraft.clinical_situation_text}</p>

          <label className="col-label">Candidate medication</label>
          <input value={medicationName} onChange={(event) => setMedicationName(event.target.value)} />

          <label className="col-label">Dose / frequency</label>
          <input value={doseFrequency} onChange={(event) => setDoseFrequency(event.target.value)} />

          <label className="col-label">Rationale (cited) -- preview</label>
          <div className="markdown previsit-draft-preview">
            <ReactMarkdown>{rationale || "*Nothing to preview yet.*"}</ReactMarkdown>
          </div>

          <label className="col-label">Edit rationale (markdown source)</label>
          <textarea
            className="previsit-draft"
            rows={8}
            value={rationale}
            onChange={(event) => setRationale(event.target.value)}
          />

          {(hasFlags || activeDraft.safety_check?.allergy_flags?.length || activeDraft.safety_check?.interaction_flags?.length) ? (
            <div className="notice danger medication-safety-flags">
              <AlertTriangle size={16} />
              <div>
                <strong>Safety check flagged possible conflicts -- review before releasing:</strong>
                <ul>
                  {(activeDraft.safety_check.allergy_flags ?? []).map((flag, index) => (
                    <li key={`allergy-${index}`}>{flag.summary}</li>
                  ))}
                  {(activeDraft.safety_check.interaction_flags ?? []).map((flag, index) => (
                    <li key={`interaction-${index}`}>{flag.summary}</li>
                  ))}
                </ul>
              </div>
            </div>
          ) : (
            <p className="notice success">No allergy or interaction conflicts detected against this patient's recorded records.</p>
          )}

          {hasFlags && (
            <>
              <label className="col-label">Override reason (required to release with flags present)</label>
              <textarea
                rows={2}
                value={overrideReason}
                onChange={(event) => setOverrideReason(event.target.value)}
                placeholder="Document why this proposal is appropriate despite the flag above."
              />
            </>
          )}

          <div className="previsit-actions">
            <button className="ghost" onClick={handleGenerate} disabled={generateBusy}>
              <RefreshCw size={16} /> {generateBusy ? "Working..." : "Regenerate proposal"}
            </button>
            <button className="ghost" onClick={handleSaveDraft} disabled={saveBusy || !medicationName.trim()}>
              Save draft
            </button>
            <button className="primary" onClick={handleRelease} disabled={releaseBusy || !medicationName.trim()}>
              <ShieldCheck size={16} /> {releaseBusy ? "Releasing..." : "Release to patient"}
            </button>
          </div>
        </>
      )}

      {!!releasedProposals.length && (
        <div className="previsit-history">
          <button className="link-btn" onClick={() => setHistoryOpen((open) => !open)}>
            {historyOpen ? "Hide" : "Show"} {releasedProposals.length} past released proposal
            {releasedProposals.length === 1 ? "" : "s"}
          </button>
          {historyOpen &&
            releasedProposals.map((item) => (
              <article key={item.id} className="previsit-history-item">
                <p className="muted">
                  Released {formatTimestamp(item.released_at)} by {item.authored_by_display_name}
                  {item.authored_by_clinical_role ? ` (${item.authored_by_clinical_role})` : ""}
                </p>
                <p>
                  <strong>{item.candidate_medication_name}</strong> -- {item.candidate_dose_frequency}
                </p>
                {item.override_reason && (
                  <p className="muted">Override reason: {item.override_reason}</p>
                )}
                <div className="markdown">
                  <ReactMarkdown>{item.rationale_text}</ReactMarkdown>
                </div>
              </article>
            ))}
        </div>
      )}
    </section>
  );
}

function ClinicalDataList({
  title,
  items,
  primary,
  secondary,
  suffix,
  secondaryAsMarkdown
}: {
  title: string;
  items: Dict<any>[];
  primary: string;
  secondary: string;
  suffix?: string;
  secondaryAsMarkdown?: boolean;
}) {
  return (
    <section className="surface-card chart-section">
      {title && <h3>{title}</h3>}
      {!items.length && <p className="muted">None recorded.</p>}
      <div className="clinical-list">
        {items.map((item, index) => (
          <div key={String(item.id ?? index)}>
            <strong>{clean(item[primary], "Recorded item")}</strong>
            {secondaryAsMarkdown ? (
              <div className="markdown">
                <ReactMarkdown>{clean(item[secondary])}</ReactMarkdown>
              </div>
            ) : (
              <span>
                {clean(item[secondary])}
                {suffix && item[suffix] ? ` ${item[suffix]}` : ""}
              </span>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}

function ActionButton({
  icon,
  title,
  body,
  onClick
}: {
  icon: React.ReactNode;
  title: string;
  body: string;
  onClick: () => void;
}) {
  return (
    <button className="action-card" onClick={onClick}>
      <span className="action-icon">{icon}</span>
      <strong>{title}</strong>
      <p>{body}</p>
    </button>
  );
}

function ChatView({
  snapshot,
  setSnapshot,
  setNotice
}: {
  snapshot: Snapshot;
  setSnapshot: (snapshot: Snapshot) => void;
  setNotice: (notice: string) => void;
}) {
  const [messages, setMessages] = useState<Message[]>(snapshot.chat_history ?? []);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [streamText, setStreamText] = useState("");
  const [pendingAttachment, setPendingAttachment] = useState<
    { kind: "image"; file: File } | { kind: "document"; file: File } | null
  >(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState("");
  const [feedbackBusy, setFeedbackBusy] = useState<Record<string, boolean>>({});
  const [panelOpen, setPanelOpen] = useState(true);
  const listRef = useRef<HTMLDivElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const documentInputRef = useRef<HTMLInputElement>(null);
  const role = snapshot.profile.clinical_role || snapshot.profile.role;
  const patientView = isPatientRole(role);

  useLayoutEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, []);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, streamText, status]);

  useEffect(() => {
    if (pendingAttachment?.kind !== "image") {
      setImagePreviewUrl("");
      return;
    }
    const url = URL.createObjectURL(pendingAttachment.file);
    setImagePreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [pendingAttachment]);

  async function sendMessage(text = draft) {
    const message = text.trim();
    if ((!message && !pendingAttachment) || busy) {
      return;
    }
    const attachment = pendingAttachment;
    setDraft("");
    setPendingAttachment(null);
    setBusy(true);
    setStatus(
      attachment?.kind === "image"
        ? "Checking image..."
        : attachment?.kind === "document"
        ? "Checking document..."
        : "Starting evidence review..."
    );
    setStreamText("");
    try {
      const stream =
        attachment?.kind === "image"
          ? streamImageAnalysis(message, attachment.file, handleStreamEvent)
          : attachment?.kind === "document"
          ? streamDocumentAnalysis(message, attachment.file, handleStreamEvent)
          : streamChat(message, handleStreamEvent);

      function handleStreamEvent(event: ChatStreamEvent) {
        if (event.type === "user_message") {
          setMessages((current) => [...current, event.message]);
        }
        if (event.type === "status") {
          setStatus(event.message);
        }
        if (event.type === "token") {
          setStreamText((current) => current + event.delta);
        }
        if (event.type === "assistant_message") {
          setMessages((current) => [...current, event.message]);
          setStreamText("");
          setStatus("");
        }
        if (event.type === "snapshot") {
          setSnapshot(event.snapshot);
        }
        if (event.type === "error") {
          setNotice(event.message);
          if (event.assistant_message) {
            setMessages((current) => [...current, event.assistant_message as Message]);
          }
          setStreamText("");
          setStatus("");
        }
      }

      await stream;
    } catch (caught) {
      setNotice(caught instanceof Error ? caught.message : "Message failed.");
    } finally {
      setBusy(false);
      setStatus("");
    }
  }

  function handleImagePick(file: File | null) {
    if (!file) {
      return;
    }
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
      setNotice("Upload a JPG, PNG, or WebP medical image.");
      return;
    }
    if (file.size > MAX_IMAGE_UPLOAD_BYTES) {
      setNotice("Image uploads must be 5 MB or smaller.");
      return;
    }
    setPendingAttachment({ kind: "image", file });
  }

  function handleDocumentPick(file: File | null) {
    if (!file) {
      return;
    }
    if (file.type !== "application/pdf") {
      setNotice("Upload a PDF document.");
      return;
    }
    if (file.size > MAX_DOCUMENT_UPLOAD_BYTES) {
      setNotice("Document uploads must be 15 MB or smaller.");
      return;
    }
    setPendingAttachment({ kind: "document", file });
  }

  async function rateMessage(message: Message, rating: FeedbackRating) {
    const traceId = clean(message.trace_id);
    if (!traceId) {
      setNotice("This response cannot be rated because it has no trace.");
      return;
    }
    const key = message.message_id ?? traceId;
    setFeedbackBusy((current) => ({ ...current, [key]: true }));
    try {
      const response = await rateResponse({
        trace_id: traceId,
        message_id: message.message_id,
        rating
      });
      setSnapshot(response.snapshot);
      setMessages(response.snapshot.chat_history);
      setNotice("Response rating saved.");
    } catch (caught) {
      setNotice(caught instanceof Error ? caught.message : "Could not save response rating.");
    } finally {
      setFeedbackBusy((current) => ({ ...current, [key]: false }));
    }
  }

  return (
    <div className={`chat-layout${panelOpen && patientView ? "" : " panel-closed"}`}>
      <section className="chat-panel">
        <div className="chat-head">
          <div className="chat-head-title">
            <img src="/assistant.png" alt="" className="chat-head-avatar" />
            <div>
              <strong>FlynnMed</strong>
              {!patientView && <span className="chat-head-role">{role}</span>}
            </div>
          </div>
          <div className="chat-head-actions">
            <button
              className="ghost icon-btn"
              onClick={async () => {
                setSnapshot(await apiRequest<Snapshot>("/api/chat", { method: "DELETE" }));
                setMessages([]);
              }}
              title="Clear conversation"
            >
              <Trash2 size={15} />
            </button>
            {patientView && !panelOpen && (
              <button className="ghost icon-btn" onClick={() => setPanelOpen(true)} title="Show panel">
                <PanelRight size={15} />
              </button>
            )}
          </div>
        </div>

        {!messages.length && (
          <div className="starter-row">
            {STARTER_PROMPTS.map((prompt) => (
              <button key={prompt} onClick={() => sendMessage(prompt)}>
                {prompt}
              </button>
            ))}
          </div>
        )}

        <div className="message-list" ref={listRef}>
          {messages.map((message, index) => (
            <MessageBubble
              key={message.message_id ?? `${message.role}-${index}`}
              message={message}
              feedbackBusy={!!feedbackBusy[message.message_id ?? message.trace_id ?? ""]}
              onFollowUp={sendMessage}
              onRate={rateMessage}
              role={role}
              snapshot={snapshot}
            />
          ))}
          {streamText && (
            <MessageBubble
              message={{
                role: "assistant",
                content: `${streamText}\n\n...`,
                metadata: {},
                sources: []
              }}
              transient
              role={role}
            />
          )}
          {status && (
            <div className="status-line">
              <span className="status-dots"><span /><span /><span /></span>
              {status}
            </div>
          )}
        </div>

        <div className="composer">
          {pendingAttachment && (
            <div className="composer-attachment">
              {pendingAttachment.kind === "image" ? (
                imagePreviewUrl && <img src={imagePreviewUrl} alt="" />
              ) : (
                <FileText size={28} />
              )}
              <div>
                <strong>{pendingAttachment.file.name}</strong>
                <span>{Math.max(1, Math.round(pendingAttachment.file.size / 1024))} KB</span>
              </div>
              <button
                className="ghost icon-btn"
                onClick={() => setPendingAttachment(null)}
                type="button"
                title="Remove attachment"
              >
                <X size={14} />
              </button>
            </div>
          )}
          <textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } }}
            placeholder="Type a question, or attach a medical image or PDF document…"
            rows={2}
          />
          <div className="composer-actions">
            <input
              ref={imageInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              className="sr-only"
              onChange={(event) => {
                handleImagePick(event.target.files?.[0] ?? null);
                event.target.value = "";
              }}
            />
            <button
              className={`ghost icon-btn${pendingAttachment?.kind === "image" ? " active" : ""}`}
              onClick={() => imageInputRef.current?.click()}
              type="button"
              title="Attach medical image"
            >
              <ImageIcon size={16} />
            </button>
            <input
              ref={documentInputRef}
              type="file"
              accept="application/pdf"
              className="sr-only"
              onChange={(event) => {
                handleDocumentPick(event.target.files?.[0] ?? null);
                event.target.value = "";
              }}
            />
            <button
              className={`ghost icon-btn${pendingAttachment?.kind === "document" ? " active" : ""}`}
              onClick={() => documentInputRef.current?.click()}
              type="button"
              title="Attach document (PDF)"
            >
              <FileText size={16} />
            </button>
            <VoiceRecorder onTranscript={(text) => setDraft((current) => `${current}${current ? " " : ""}${text}`)} />
            <div className="composer-spacer" />
            <button className="ghost icon-btn" onClick={() => { setDraft(""); setPendingAttachment(null); }} type="button" title="Clear input">
              <Trash2 size={14} />
            </button>
            <button className="primary" onClick={() => sendMessage()} disabled={busy || (!draft.trim() && !pendingAttachment)}>
              <Send size={15} />
              Send
            </button>
          </div>
        </div>
      </section>

      {patientView && panelOpen && (
        <aside className="side-panel">
          <div className="side-panel-header">
            <button className="ghost side-panel-close" onClick={() => setPanelOpen(false)} title="Hide panel">
              <PanelRight size={17} />
              Hide panel
            </button>
          </div>
          <NotesPanel
            snapshot={snapshot}
            setSnapshot={setSnapshot}
            setNotice={setNotice}
            role={role}
          />
          {patientView && <PrevisitSummariesPanel />}
          {patientView && <MedicationProposalsPanel />}
          <UploadPanel snapshot={snapshot} setSnapshot={setSnapshot} setNotice={setNotice} />
          <ExportPanel snapshot={snapshot} setNotice={setNotice} />
          <RecordPanel snapshot={snapshot} setSnapshot={setSnapshot} compact={false} />
        </aside>
      )}
    </div>
  );
}

function MessageBubble({
  message,
  transient = false,
  feedbackBusy = false,
  onFollowUp,
  onRate,
  role,
  snapshot
}: {
  message: Message;
  transient?: boolean;
  feedbackBusy?: boolean;
  onFollowUp?: (text: string) => void;
  onRate?: (message: Message, rating: FeedbackRating) => void;
  role?: string;
  snapshot?: Snapshot;
}) {
  const isUser = message.role === "user";
  const patientView = isPatientRole(role);
  const metadata = message.metadata ?? {};
  const triage = metadata.triage_summary;
  const alerts = metadata.medication_alerts ?? [];
  const imageSrc = metadata.image_b64 ? `data:image/png;base64,${metadata.image_b64}` : metadata.image_url;
  const uploadedImageSrc = metadata.uploaded_image_b64
    ? `data:${clean(metadata.uploaded_image_mime, "image/jpeg")};base64,${metadata.uploaded_image_b64}`
    : "";
  const feedback = metadata.feedback as Dict<any> | undefined;
  const selectedRating = clean(feedback?.rating);
  const canRate = !isUser && !transient && !!message.trace_id;

  return (
    <article className={`message ${isUser ? "user-message" : "assistant-message"} ${transient ? "transient" : ""}`}>
      {!isUser && <img src="/assistant.png" alt="" />}
      <div className="message-body">
        {!isUser && (
          <div className="message-meta">
            <strong>FlynnMed</strong>
            {message.timestamp && <span>{formatTimestamp(message.timestamp)}</span>}
            {!patientView && message.trace_id && <span className="trace-id">{message.trace_id}</span>}
          </div>
        )}
        {isUser && (uploadedImageSrc || metadata.uploaded_image_name) && (
          <figure className="uploaded-image-block">
            {uploadedImageSrc && <img src={uploadedImageSrc} alt="" />}
            <figcaption>{clean(metadata.uploaded_image_name, "Uploaded medical image")}</figcaption>
          </figure>
        )}
        {isUser && metadata.uploaded_document_name && (
          <div className="uploaded-document-chip">
            <FileText size={16} />
            <span>{clean(metadata.uploaded_document_name, "Uploaded document")}</span>
          </div>
        )}
        <div className="markdown">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>
        {imageSrc && (
          <figure className="media-block">
            <img src={imageSrc} alt={clean(metadata.image_caption, "Generated illustration")} />
            <figcaption>{clean(metadata.image_caption, "AI-generated educational illustration")}</figcaption>
          </figure>
        )}
        {metadata.video_url && (
          <figure className="media-block">
            <video src={metadata.video_url} controls />
            <figcaption>{clean(metadata.video_caption, "Generated educational video")}</figcaption>
          </figure>
        )}
        {metadata.video_rate_limit_msg && <div className="notice warn">{metadata.video_rate_limit_msg}</div>}

        {/* Patient: subtle urgency strip only when action is needed (no duplicate triage card) */}
        {patientView && triage && Object.keys(triage).length > 0 && (() => {
          const urgencyKey = String(triage.urgency_level ?? "").toLowerCase();
          if (urgencyKey === "high" || urgencyKey === "urgent" || urgencyKey === "crisis") {
            const u = PATIENT_URGENCY[urgencyKey] ?? PATIENT_URGENCY.high;
            return (
              <div className="urgency-strip" style={{ borderColor: u.color, background: u.bg, color: u.color }}>
                <ShieldCheck size={14} />
                <span>{u.label}</span>
              </div>
            );
          }
          return null;
        })()}

        {/* Clinician: collapsible clinical summary */}
        {!patientView && triage && Object.keys(triage).length > 0 && (
          <details className="details-block triage-details">
            <summary>Clinical triage · <strong>{clean(triage.urgency_level, "Routine")}</strong></summary>
            <TriageCard summary={triage} />
          </details>
        )}

        {alerts.length > 0 && <MedicationAlerts alerts={alerts} />}

        {!patientView && message.sources && message.sources.length > 0 && (
          <SourceList sources={message.sources} />
        )}

        {!isUser && metadata.follow_up_questions?.length > 0 && (
          <div className="follow-ups">
            {metadata.follow_up_questions.slice(0, 3).map((q: string | { display: string; prompt: string }, i: number) => {
              const display = typeof q === "string" ? q : q.display;
              const prompt = typeof q === "string" ? q : q.prompt;
              return (
                <button key={i} className="follow-up-btn" onClick={() => onFollowUp?.(prompt)}>
                  {display}
                </button>
              );
            })}
          </div>
        )}

        {!isUser && !patientView && metadata.trace && Object.keys(metadata.trace).length > 0 && (
          <EvidenceBasis trace={metadata.trace} sourceCount={message.sources?.length ?? 0} />
        )}

        {canRate && !isUser && (
          <ResponseFeedback
            selectedRating={selectedRating}
            busy={feedbackBusy}
            onRate={(rating) => onRate?.(message, rating)}
          />
        )}
      </div>
    </article>
  );
}

function ResponseFeedback({
  selectedRating,
  busy,
  onRate
}: {
  selectedRating: string;
  busy: boolean;
  onRate: (rating: FeedbackRating) => void;
}) {
  const locked = busy || !!selectedRating;
  return (
    <div className="feedback-row" aria-label="Response rating">
      <button
        className={`feedback-button ${selectedRating === "thumbs_up" ? "active" : ""}`}
        onClick={() => onRate("thumbs_up")}
        disabled={locked}
        title="Rate helpful"
        aria-label="Rate response helpful"
        type="button"
      >
        <ThumbsUp size={17} />
      </button>
      <button
        className={`feedback-button ${selectedRating === "thumbs_down" ? "active down" : ""}`}
        onClick={() => onRate("thumbs_down")}
        disabled={locked}
        title="Rate not helpful"
        aria-label="Rate response not helpful"
        type="button"
      >
        <ThumbsDown size={17} />
      </button>
      {(busy || selectedRating) && <span>{busy ? "Saving" : "Saved"}</span>}
    </div>
  );
}

function TriageCard({ summary }: { summary: Dict<any> }) {
  const monitor = Array.isArray(summary.what_to_monitor) ? summary.what_to_monitor.slice(0, 3) : [];
  const actions = Array.isArray(summary.immediate_actions) ? summary.immediate_actions.slice(0, 3) : [];
  const escalation = Array.isArray(summary.escalation_triggers) ? summary.escalation_triggers.slice(0, 3) : [];
  return (
    <div className="triage-card">
      <div>
        <span>Structured triage</span>
        <strong>{clean(summary.next_step, "Self-care")}</strong>
      </div>
      <p>
        <b>Urgency:</b> {clean(summary.urgency_level, "Routine")}
      </p>
      {summary.pathway_label && <p>{summary.pathway_label}</p>}
      {!!monitor.length && (
        <ul>
          {monitor.map((item: string) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      )}
      {!!actions.length && <p>{actions.join(" ")}</p>}
      {!!escalation.length && <p className="escalation">Escalate if: {escalation.join("; ")}</p>}
    </div>
  );
}

function PatientTriageCard({ summary }: { summary: Dict<any> }) {
  const urgencyKey = String(summary.urgency_level ?? "").toLowerCase();
  const urgency = PATIENT_URGENCY[urgencyKey] ?? PATIENT_URGENCY.routine;
  const nextStep = clean(summary.next_step, "Continue monitoring your symptoms.");
  const monitors = Array.isArray(summary.what_to_monitor) ? summary.what_to_monitor.slice(0, 2) : [];
  return (
    <div className="patient-triage">
      <span
        className="urgency-badge"
        style={{ color: urgency.color, background: urgency.bg, borderColor: urgency.color }}
      >
        {urgency.label}
      </span>
      <p><strong>What to do:</strong> {nextStep}</p>
      {monitors.length > 0 && (
        <p className="patient-triage-watch"><strong>Watch for:</strong> {monitors.join("; ")}</p>
      )}
    </div>
  );
}

function ClinicalTrendPanel({ snapshot }: { snapshot: Snapshot }) {
  const vitals = snapshot.vitals ?? [];
  const vitalTypes = unique(vitals.map((v) => v.type));
  const chartable = vitalTypes.filter((type) => buildSeries(vitals, type).length >= 2);
  if (!chartable.length) return null;
  return (
    <div className="clinical-trends">
      <span className="evidence-label">Patient vitals trend</span>
      <div className="clinical-trends-grid">
        {chartable.slice(0, 4).map((type) => {
          const series = buildSeries(vitals, type).map((p) => ({
            date: p.date,
            value: p.value,
            secondValue: p.secondValue
          }));
          return (
            <div key={type} className="clinical-trend-item">
              <span>{vitalLabel(type)}</span>
              <TinyChart series={series} />
            </div>
          );
        })}
      </div>
    </div>
  );
}

function MedicationAlerts({ alerts }: { alerts: Dict<any>[] }) {
  return (
    <div className="alert-list">
      <strong>Medication interaction check</strong>
      {alerts.slice(0, 3).map((alert) => (
        <div key={clean(alert.pair)} className={`interaction ${clean(alert.severity, "mentioned")}`}>
          <span>{clean(alert.pair, "Medication pair")}</span>
          <p>{clean(alert.summary)}</p>
        </div>
      ))}
    </div>
  );
}

const TIER_LABELS: Record<number, string> = {
  1: "NHS / NICE guidelines",
  2: "Systematic reviews",
  3: "Research papers",
};

const RISK_PLAIN: Record<string, { label: string; color: string }> = {
  routine:  { label: "Routine assessment",             color: "var(--green)" },
  elevated: { label: "Careful attention recommended",  color: "var(--gold)" },
  high:     { label: "Prompt review advised",          color: "var(--accent)" },
  urgent:   { label: "Urgent review recommended",      color: "var(--danger)" },
  crisis:   { label: "Seek immediate help",            color: "var(--danger)" },
};

function toNumber(value: unknown): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function EvidenceBasis({ trace, sourceCount }: { trace: Dict<any>; sourceCount: number }) {
  const tiers: number[] = Array.isArray(trace.evidence_tiers_present) ? trace.evidence_tiers_present : [];
  const risk = RISK_PLAIN[String(trace.risk_level ?? "").toLowerCase()];
  const escalated: boolean = !!trace.escalation_triggered;
  const crisis: boolean = !!trace.crisis_detected;
  const quality = trace.evidence_quality as Dict<any> | undefined;
  const qualityCounts = quality?.status_counts as Dict<any> | undefined;
  const patientAligned = toNumber(qualityCounts?.patient_aligned);
  const backgroundOnly = toNumber(qualityCounts?.background_only) + toNumber(qualityCounts?.question_aligned);
  const excluded = toNumber(quality?.excluded_source_count);
  const qualityStatus = clean(quality?.overall_status);
  const qualityLabel = patientAligned
    ? `${patientAligned} patient-aligned`
    : qualityStatus === "question_aligned_only"
      ? "Question-aligned only"
      : qualityStatus === "no_sources_passed_quality_gate"
        ? "Quality gate filtered"
        : "";

  const tierBadges = tiers.map((t) => TIER_LABELS[t]).filter(Boolean);

  return (
    <div className="evidence-basis">
      <span className="evidence-label">Evidence basis</span>
      <div className="evidence-chips">
        {tierBadges.map((label) => (
          <span key={label} className="ev-chip ev-chip--source">{label}</span>
        ))}
        {sourceCount > 0 && (
          <span className="ev-chip ev-chip--count">{sourceCount} source{sourceCount === 1 ? "" : "s"} reviewed</span>
        )}
        {qualityLabel && (
          <span className={`ev-chip ev-chip--quality ${patientAligned ? "aligned" : excluded ? "filtered" : ""}`}>
            {qualityLabel}
          </span>
        )}
        {backgroundOnly > 0 && patientAligned > 0 && (
          <span className="ev-chip ev-chip--count">{backgroundOnly} general-context source{backgroundOnly === 1 ? "" : "s"}</span>
        )}
        {excluded > 0 && (
          <span className="ev-chip ev-chip--filtered">{excluded} filtered</span>
        )}
        {risk && (
          <span className="ev-chip ev-chip--risk" style={{ color: risk.color, borderColor: risk.color }}>
            {risk.label}
          </span>
        )}
        {(escalated || crisis) && (
          <span className="ev-chip ev-chip--alert">Safety guidance applied</span>
        )}
      </div>
    </div>
  );
}

function SourceList({ sources }: { sources: NonNullable<Message["sources"]> }) {
  return (
    <details className="details-block">
      <summary>{sources.length} source{sources.length === 1 ? "" : "s"}</summary>
      <div className="source-grid">
        {sources.map((source, index) => {
          const qualityStatus = clean(source.evidence_quality_status);
          const qualityLabel = source.usable_for_patient_specific_guidance
            ? "Patient-aligned"
            : qualityStatus === "background_only"
              ? "Background only"
              : qualityStatus === "question_aligned"
                ? "Question-aligned"
                : "";
          return (
            <a key={source.source_id ?? index} href={source.url || undefined} target="_blank" rel="noreferrer">
              <span>{source.source_id ?? `S${index + 1}`}</span>
              <strong>{clean(source.title, "Untitled source")}</strong>
              <small>{unique([source.journal, source.year, source.tier_label]).join(" - ")}</small>
              {qualityLabel && (
                <em className={`source-quality ${source.usable_for_patient_specific_guidance ? "aligned" : ""}`}>
                  {qualityLabel}
                </em>
              )}
              {!!source.patient_alignment_facts?.length && (
                <small>Matches: {source.patient_alignment_facts.slice(0, 3).join(", ")}</small>
              )}
              {source.exact_passage && (
                <blockquote className="source-passage">
                  "{source.exact_passage}"
                  <footer>
                    {[
                      source.passage_locator,
                      source.source_version ? `version ${source.source_version}` : "",
                      source.retrieved_at ? `checked ${formatTimestamp(source.retrieved_at)}` : "",
                    ]
                      .filter(Boolean)
                      .join(" - ")}
                  </footer>
                </blockquote>
              )}
            </a>
          );
        })}
      </div>
    </details>
  );
}

function VoiceRecorder({ onTranscript }: { onTranscript: (text: string) => void }) {
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);

  async function toggle() {
    if (recording) {
      mediaRecorder.current?.stop();
      setRecording(false);
      return;
    }
    if (!navigator.mediaDevices || typeof MediaRecorder === "undefined") {
      return;
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    chunks.current = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size) {
        chunks.current.push(event.data);
      }
    };
    recorder.onstop = async () => {
      stream.getTracks().forEach((track) => track.stop());
      const blob = new Blob(chunks.current, { type: "audio/webm" });
      const file = new File([blob], "recording.webm", { type: "audio/webm" });
      setBusy(true);
      try {
        const result = await transcribeAudio(file);
        if (result.text) {
          onTranscript(result.text);
        }
      } finally {
        setBusy(false);
      }
    };
    mediaRecorder.current = recorder;
    recorder.start();
    setRecording(true);
  }

  return (
    <button
      className={`ghost icon-btn${recording ? " recording" : ""}`}
      onClick={toggle}
      type="button"
      disabled={busy}
      title={recording ? "Stop recording" : busy ? "Transcribing…" : "Voice input"}
    >
      {recording ? <StopCircle size={16} /> : busy ? <span style={{ fontSize: "0.7rem", fontWeight: 750 }}>…</span> : <Mic size={16} />}
    </button>
  );
}

const SOAP_ROLE_LABELS: Record<string, { s: string; o: string; a: string; p: string }> = {
  doctor: {
    s: "Subjective",
    o: "Objective",
    a: "Assessment",
    p: "Plan",
  },
  nurse: {
    s: "Presenting concern",
    o: "Observations",
    a: "Nursing assessment",
    p: "Care plan",
  },
  midwife: {
    s: "Maternal concern",
    o: "Maternal & fetal assessment",
    a: "Risk assessment",
    p: "Maternity plan",
  },
  physiotherapist: {
    s: "Presenting complaint",
    o: "Physical assessment",
    a: "Clinical impression",
    p: "Treatment plan",
  },
};

const SOAP_URGENCY_BADGE: Record<string, { label: string; cls: string }> = {
  routine:  { label: "Routine",            cls: "urgency-badge" },
  elevated: { label: "Worth monitoring",   cls: "urgency-badge urgency-elevated" },
  high:     { label: "See GP soon",        cls: "urgency-badge urgency-high" },
  urgent:   { label: "Urgent care",        cls: "urgency-badge urgency-urgent" },
  crisis:   { label: "Emergency",          cls: "urgency-badge urgency-urgent" },
};

function NotesPanel({
  snapshot,
  setSnapshot,
  setNotice,
  role
}: {
  snapshot: Snapshot;
  setSnapshot: (snapshot: Snapshot) => void;
  setNotice: (notice: string) => void;
  role?: string;
}) {
  const [busy, setBusy] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editDraft, setEditDraft] = useState<Partial<ClinicalNote>>({});
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const isClinician = !isPatientRole(role);
  const notes = snapshot.clinical_notes ?? [];

  async function handleGenerate() {
    setBusy(true);
    try {
      const result = await generateNote({});
      setSnapshot(result.snapshot);
      setExpandedId(result.note.note_id);
      setNotice("Clinical note generated.");
    } catch (err) {
      setNotice(err instanceof Error ? err.message : "Note generation failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleSaveEdit(noteId: string) {
    setBusy(true);
    try {
      const result = await updateNote(noteId, editDraft);
      setSnapshot(result.snapshot);
      setEditingId(null);
      setEditDraft({});
      setNotice("Note saved.");
    } catch (err) {
      setNotice(err instanceof Error ? err.message : "Save failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleEmail(noteId: string) {
    setBusy(true);
    try {
      const result = await emailNote(noteId);
      setSnapshot(result.snapshot);
      setNotice(`Note sent to ${result.sent_to}`);
    } catch (err) {
      setNotice(err instanceof Error ? err.message : "Email failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete(noteId: string) {
    setBusy(true);
    try {
      await deleteNote(noteId);
      const updated = await fetchSnapshot();
      setSnapshot(updated);
      if (expandedId === noteId) setExpandedId(null);
      setNotice("Note deleted.");
    } catch (err) {
      setNotice(err instanceof Error ? err.message : "Delete failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleUrgentAlert(note: ClinicalNote) {
    setBusy(true);
    try {
      const result = await sendUrgentAlert(
        note.gp_visit_reason || note.assessment || "Urgent clinical concern flagged",
        note.urgency_level
      );
      setNotice(`Urgent alert sent to ${result.sent_to}`);
    } catch (err) {
      setNotice(err instanceof Error ? err.message : "Alert failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="tool-panel notes-panel">
      <div className="panel-head">
        <ClipboardList size={19} />
        <strong>Clinical notes</strong>
      </div>

      {isClinician && (
        <button className="primary full" onClick={handleGenerate} disabled={busy}>
          <Plus size={17} />
          {busy ? "Generating…" : "Generate SOAP note"}
        </button>
      )}

      {notes.length === 0 && (
        <p className="muted" style={{ fontSize: "0.85rem" }}>
          {isClinician
            ? "Generate a note from the current conversation."
            : "No consultation notes yet."}
        </p>
      )}

      <div className="notes-list">
        {notes.slice(0, 5).map((note) => {
          const badge = SOAP_URGENCY_BADGE[note.urgency_level] ?? SOAP_URGENCY_BADGE.routine;
          const isExpanded = expandedId === note.note_id;
          const isEditing = editingId === note.note_id;
          const roleLabels = SOAP_ROLE_LABELS[String((note as any).role_key ?? "doctor")] ?? SOAP_ROLE_LABELS.doctor;

          return (
            <article key={note.note_id} className="note-card">
              <div className="note-card-head" onClick={() => setExpandedId(isExpanded ? null : note.note_id)}>
                <div>
                  <span className={badge.cls}>{badge.label}</span>
                  <time>{note.created_at.slice(0, 10)}</time>
                  {note.edited_by && <span className="note-edited">edited</span>}
                </div>
                <p className="note-question">{note.question?.slice(0, 70) || "Consultation note"}</p>
              </div>

              {isExpanded && !isEditing && (
                <div className="note-body">
                  {isClinician ? (
                    <>
                      <SoapSection label={roleLabels.s} text={note.subjective} />
                      <SoapSection label={roleLabels.o} text={note.objective} />
                      <SoapSection label={roleLabels.a} text={note.assessment} />
                      <SoapSection label={roleLabels.p} text={note.plan} />
                    </>
                  ) : (
                    <PatientNoteView note={note} />
                  )}

                  <div className="note-actions">
                    {isClinician && (
                      <button className="ghost" onClick={() => { setEditingId(note.note_id); setEditDraft(note); }}>
                        Edit note
                      </button>
                    )}
                    <button className="ghost" onClick={() => handleEmail(note.note_id)} disabled={busy}>
                      {note.email_sent ? "Resend" : "Email to me"}
                    </button>
                    {note.requires_gp_visit && isPatientRole(role) && (
                      <button className="ghost urgent-btn" onClick={() => handleUrgentAlert(note)} disabled={busy}>
                        Email GP advice
                      </button>
                    )}
                    {note.requires_gp_visit && isClinician && (
                      <button className="ghost urgent-btn" onClick={() => handleUrgentAlert(note)} disabled={busy}>
                        Send GP alert
                      </button>
                    )}
                    {isClinician && (
                      <button className="ghost" onClick={() => handleDelete(note.note_id)} disabled={busy} title="Delete note">
                        <Trash2 size={14} />
                      </button>
                    )}
                  </div>
                </div>
              )}

              {isExpanded && isEditing && isClinician && (
                <div className="note-editor">
                  <SoapEditor label={roleLabels.s} value={editDraft.subjective ?? ""} onChange={v => setEditDraft(d => ({ ...d, subjective: v }))} />
                  <SoapEditor label={roleLabels.o} value={editDraft.objective ?? ""} onChange={v => setEditDraft(d => ({ ...d, objective: v }))} />
                  <SoapEditor label={roleLabels.a} value={editDraft.assessment ?? ""} onChange={v => setEditDraft(d => ({ ...d, assessment: v }))} />
                  <SoapEditor label={roleLabels.p} value={editDraft.plan ?? ""} onChange={v => setEditDraft(d => ({ ...d, plan: v }))} />
                  <div className="note-actions">
                    <button className="primary" onClick={() => handleSaveEdit(note.note_id)} disabled={busy}>Save changes</button>
                    <button className="ghost" onClick={() => { setEditingId(null); setEditDraft({}); }}>Cancel</button>
                  </div>
                </div>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}

function SoapSection({ label, text }: { label: string; text: string }) {
  return (
    <div className="soap-section">
      <span className="soap-label">{label}</span>
      <div className="soap-content markdown">
        <ReactMarkdown>{text || "-"}</ReactMarkdown>
      </div>
    </div>
  );
}

function SoapEditor({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  return (
    <div className="soap-section">
      <span className="soap-label">{label}</span>
      <textarea value={value} onChange={e => onChange(e.target.value)} rows={5} />
    </div>
  );
}

function PatientNoteView({ note }: { note: ClinicalNote }) {
  const urgency = PATIENT_URGENCY[String(note.urgency_level ?? "").toLowerCase()] ?? PATIENT_URGENCY.routine;
  return (
    <div className="patient-note-view">
      <div className="patient-note-header">
        <span className="urgency-badge" style={{ color: urgency.color, background: urgency.bg, borderColor: urgency.color }}>
          {urgency.label}
        </span>
        {note.requires_gp_visit && (
          <span style={{ fontSize: "0.78rem", color: "var(--accent)", fontWeight: 750 }}>GP visit recommended</span>
        )}
      </div>
      <div className="patient-note-section">
        <span className="soap-label">What was discussed</span>
        <div className="soap-content markdown">
          <ReactMarkdown>{note.subjective || "-"}</ReactMarkdown>
        </div>
      </div>
      <div className="patient-note-section">
        <span className="soap-label">What happens next</span>
        <div className="soap-content markdown">
          <ReactMarkdown>{note.plan || "-"}</ReactMarkdown>
        </div>
      </div>
    </div>
  );
}

function PrevisitSummariesPanel() {
  const [summaries, setSummaries] = useState<PreVisitSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchMyPrevisitSummaries()
      .then((res) => {
        if (!cancelled) setSummaries(res.summaries);
      })
      .catch(() => {
        // Nothing released yet, or a transient load failure -- an empty
        // panel is the right fallback either way.
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading || !summaries.length) return null;

  return (
    <section className="tool-panel notes-panel">
      <div className="panel-head">
        <Stethoscope size={19} />
        <strong>Notes from your care team</strong>
      </div>
      <div className="notes-list">
        {summaries.map((item) => {
          const isExpanded = expandedId === item.id;
          return (
            <article key={item.id} className="note-card">
              <div className="note-card-head" onClick={() => setExpandedId(isExpanded ? null : item.id)}>
                <div>
                  <time>{formatDate(item.released_at)}</time>
                </div>
                <p className="note-question">
                  {item.authored_by_display_name}
                  {item.authored_by_clinical_role ? ` (${item.authored_by_clinical_role})` : ""}
                </p>
              </div>
              {isExpanded && (
                <div className="note-body">
                  <div className="soap-content markdown">
                    <ReactMarkdown>{item.summary_text}</ReactMarkdown>
                  </div>
                </div>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}

function MedicationProposalsPanel() {
  const [proposals, setProposals] = useState<ProposedMedication[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchMyMedicationProposals()
      .then((res) => {
        if (!cancelled) setProposals(res.proposals);
      })
      .catch(() => {
        // Nothing released yet, or a transient load failure -- an empty
        // panel is the right fallback either way.
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading || !proposals.length) return null;

  return (
    <section className="tool-panel notes-panel">
      <div className="panel-head">
        <Pill size={19} />
        <strong>Medications from your care team</strong>
      </div>
      <div className="notes-list">
        {proposals.map((item) => {
          const isExpanded = expandedId === item.id;
          return (
            <article key={item.id} className="note-card">
              <div className="note-card-head" onClick={() => setExpandedId(isExpanded ? null : item.id)}>
                <div>
                  <time>{formatDate(item.released_at)}</time>
                </div>
                <p className="note-question">
                  {item.candidate_medication_name} -- {item.candidate_dose_frequency}
                </p>
              </div>
              {isExpanded && (
                <div className="note-body">
                  <p className="muted">
                    Proposed by {item.authored_by_display_name}
                    {item.authored_by_clinical_role ? ` (${item.authored_by_clinical_role})` : ""}
                  </p>
                  <div className="soap-content markdown">
                    <ReactMarkdown>{item.rationale_text}</ReactMarkdown>
                  </div>
                </div>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}

function UploadPanel({
  snapshot,
  setSnapshot,
  setNotice
}: {
  snapshot: Snapshot;
  setSnapshot: (snapshot: Snapshot) => void;
  setNotice: (notice: string) => void;
}) {
  const [files, setFiles] = useState<File[]>([]);
  const [processUnverified, setProcessUnverified] = useState(false);
  const [busy, setBusy] = useState(false);
  const [lastResult, setLastResult] = useState<{ processed: number; pending: number; duplicates: string[]; rejected: string[]; noDataExtracted: string[] } | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const uploads = [...snapshot.uploads].sort((a, b) => (b.uploaded_at || "").localeCompare(a.uploaded_at || ""));

  async function submit() {
    if (!files.length) {
      return;
    }
    setBusy(true);
    setLastResult(null);
    try {
      const result = await uploadDocuments(files, processUnverified);
      setSnapshot(result.snapshot);
      setFiles([]);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
      const noDataExtracted = result.processed
        .filter((item) => {
          const extracted = item.extracted;
          if (!extracted) return false;
          const hasData = ["vitals", "medications", "allergies", "conditions"].some(
            (key) => (extracted[key as keyof UploadExtracted] as unknown[] | undefined ?? []).length > 0
          );
          return !hasData;
        })
        .map((item) => {
          const reasons = item.extracted?.extraction_errors;
          const reasonText = reasons && reasons.length ? reasons.join("; ") : "no structured health data could be extracted from this document.";
          return `"${item.file}" -- ${reasonText}`;
        });

      setLastResult({
        processed: result.processed.length,
        pending: result.pending.length,
        duplicates: result.duplicates.map((item) => `"${item.file}" -- ${item.message}`),
        rejected: result.rejected.map((item) => `"${item.file}" -- ${item.message}`),
        noDataExtracted
      });
      if (result.duplicates.length) {
        setNotice(`${result.duplicates.length} document already uploaded -- skipped.`);
      } else if (result.rejected.length) {
        setNotice(`${result.rejected.length} document rejected -- not a health record.`);
      } else if (result.pending.length) {
        setNotice(`${result.pending.length} document needs review before extraction.`);
      } else if (noDataExtracted.length) {
        setNotice(`${result.processed.length} document processed, but no health data could be read from ${noDataExtracted.length} of them.`);
      } else {
        setNotice(`${result.processed.length} document processed.`);
      }
    } catch (caught) {
      setNotice(caught instanceof Error ? caught.message : "Upload failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="tool-panel">
      <div className="panel-head">
        <Upload size={19} />
        <strong>Documents</strong>
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept="application/pdf"
        multiple
        onChange={(event) => setFiles(Array.from(event.target.files ?? []))}
      />
      <label className="check-row">
        <input type="checkbox" checked={processUnverified} onChange={(event) => setProcessUnverified(event.target.checked)} />
        Process even when the patient name needs review
      </label>
      <button className="primary full" onClick={submit} disabled={busy || !files.length}>
        <Upload size={18} />
        {busy ? "Uploading..." : "Upload PDFs"}
      </button>
      {lastResult && (lastResult.processed > 0 || lastResult.pending > 0) && (
        <p className="notice success upload-confirm">
          <CheckCircle2 size={16} />
          {lastResult.processed} processed{lastResult.pending ? `, ${lastResult.pending} need review` : ""}.
        </p>
      )}
      {lastResult && lastResult.duplicates.map((message, index) => (
        <p className="notice warn upload-confirm" key={`dup-${index}`}>
          <AlertTriangle size={16} />
          {message}
        </p>
      ))}
      {lastResult && lastResult.rejected.map((message, index) => (
        <p className="notice error upload-confirm" key={`rej-${index}`}>
          <AlertTriangle size={16} />
          {message}
        </p>
      ))}
      {lastResult && lastResult.noDataExtracted.map((message, index) => (
        <p className="notice warn upload-confirm" key={`no-data-${index}`}>
          <AlertTriangle size={16} />
          {message}
        </p>
      ))}
      {uploads.length > 0 ? (
        <ul className="upload-list">
          {uploads.map((upload) => (
            <li key={`${upload.file}-${upload.uploaded_at}`}>
              <FileText size={16} />
              <div>
                <strong>{clean(upload.file, "Document")}</strong>
                <span>{formatTimestamp(upload.uploaded_at) || "Recently uploaded"}</span>
              </div>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted upload-empty">No documents uploaded yet.</p>
      )}
    </section>
  );
}

function ExportPanel({ snapshot, setNotice }: { snapshot: Snapshot; setNotice: (notice: string) => void }) {
  async function download(path: string, filename: string) {
    try {
      await downloadProtectedFile(path, filename);
    } catch (caught) {
      setNotice(caught instanceof Error ? caught.message : "Download failed.");
    }
  }

  return (
    <section className="tool-panel">
      <div className="panel-head">
        <FileDown size={19} />
        <strong>Exports</strong>
      </div>
      <button className="ghost full" onClick={() => download("/api/export/summary.pdf", `${snapshot.user}-health-summary.pdf`)}>
        <Download size={17} />
        Health summary PDF
      </button>
      <button className="ghost full" onClick={() => download("/api/export/account", `${snapshot.user}-audit.json`)}>
        <Download size={17} />
        Account export
      </button>
    </section>
  );
}

function RecordPanel({
  snapshot,
  setSnapshot,
  compact
}: {
  snapshot: Snapshot;
  setSnapshot: (snapshot: Snapshot) => void;
  compact: boolean;
}) {
  const [profileOpen, setProfileOpen] = useState(false);
  const [profile, setProfile] = useState({
    display_name: clean(snapshot.profile.display_name),
    email: clean(snapshot.profile.email),
    care_context: clean(snapshot.profile.care_context),
    organization: clean(snapshot.profile.organization),
    date_of_birth: clean(snapshot.profile.date_of_birth),
    biological_sex: clean(snapshot.profile.biological_sex)
  });
  const [symptom, setSymptom] = useState({ symptom: "", logged_for: new Date().toISOString().slice(0, 10), severity: 5, triggers: "", notes: "" });
  const [condition, setCondition] = useState({ name: "", status: "active", recorded_on: "", notes: "" });
  const [medication, setMedication] = useState({ name: "", dose: "", schedule: "", reason: "", started_on: "", notes: "" });
  const [allergy, setAllergy] = useState({ name: "", reaction: "", severity: "unknown", allergy_type: "other", confirmed: true, notes: "" });
  const [vitals, setVitals] = useState({ type: "blood_pressure", value: "", unit: "mmHg", recorded_on: new Date().toISOString().slice(0, 10), notes: "" });

  async function save(path: string, payload: Record<string, unknown>, reset?: () => void) {
    const updated = await apiRequest<Snapshot>(path, {
      method: "POST",
      body: JSON.stringify(payload)
    });
    setSnapshot(updated);
    reset?.();
  }

  async function remove(path: string) {
    setSnapshot(await apiRequest<Snapshot>(path, { method: "DELETE" }));
  }

  const recentItems = [
    ...snapshot.conditions.slice(0, 3).map((item) => ({ kind: "Condition", title: clean(item.name), id: clean(item.condition_id), path: `/api/conditions/${item.condition_id}` })),
    ...snapshot.medications.slice(0, 3).map((item) => ({ kind: "Medication", title: clean(item.name), id: clean(item.medication_id), path: `/api/medications/${item.medication_id}` })),
    ...snapshot.allergies.slice(0, 2).map((item) => ({ kind: "Allergy", title: clean(item.name), id: clean(item.allergy_id), path: `/api/allergies/${item.allergy_id}` }))
  ];

  return (
    <section className={compact ? "record-panel compact" : "record-panel"}>
      <div className="panel-head">
        <ClipboardList size={19} />
        <strong>Health record</strong>
      </div>

      <details open={!compact || profileOpen} onToggle={(event) => setProfileOpen(event.currentTarget.open)}>
        <summary>Profile</summary>
        <form
          className="stack"
          onSubmit={async (event) => {
            event.preventDefault();
            setSnapshot(
              await apiRequest<Snapshot>("/api/profile", {
                method: "PUT",
                body: JSON.stringify(profile)
              })
            );
          }}
        >
          <label>
            Display name
            <input value={profile.display_name} onChange={(event) => setProfile({ ...profile, display_name: event.target.value })} />
          </label>
          <label>
            Email
            <input value={profile.email} onChange={(event) => setProfile({ ...profile, email: event.target.value })} />
          </label>
          <label>
            Care context
            <input value={profile.care_context} onChange={(event) => setProfile({ ...profile, care_context: event.target.value })} />
          </label>
          <div className="form-grid">
            <label>
              Date of birth
              <input type="date" value={profile.date_of_birth} onChange={(event) => setProfile({ ...profile, date_of_birth: event.target.value })} />
            </label>
            <label>
              Biological sex
              <select value={profile.biological_sex} onChange={(event) => setProfile({ ...profile, biological_sex: event.target.value })}>
                <option value="">Prefer not to say</option>
                <option>Male</option>
                <option>Female</option>
                <option>Other</option>
              </select>
            </label>
          </div>
          <button className="primary full" type="submit">
            <ShieldCheck size={18} />
            Save profile
          </button>
        </form>
      </details>

      <details open={!compact}>
        <summary>Symptoms</summary>
        <form
          className="stack"
          onSubmit={(event) => {
            event.preventDefault();
            save("/api/symptoms", symptom, () => setSymptom({ ...symptom, symptom: "", notes: "", triggers: "" }));
          }}
        >
          <input placeholder="Symptom" value={symptom.symptom} onChange={(event) => setSymptom({ ...symptom, symptom: event.target.value })} />
          <div className="form-grid">
            <input type="date" value={symptom.logged_for} onChange={(event) => setSymptom({ ...symptom, logged_for: event.target.value })} />
            <label className="range-label">
              Severity {symptom.severity}/10
              <input type="range" min="0" max="10" value={symptom.severity} onChange={(event) => setSymptom({ ...symptom, severity: Number(event.target.value) })} />
            </label>
          </div>
          <input placeholder="Triggers" value={symptom.triggers} onChange={(event) => setSymptom({ ...symptom, triggers: event.target.value })} />
          <textarea placeholder="Notes" value={symptom.notes} onChange={(event) => setSymptom({ ...symptom, notes: event.target.value })} rows={2} />
          <button className="primary full" type="submit">
            <Plus size={18} />
            Save symptom
          </button>
        </form>
        <MiniList items={snapshot.symptom_logs.slice(0, 5)} titleKey="symptom" detailKey="logged_for" deleteKey="log_id" deletePath="/api/symptoms" onDelete={remove} />
      </details>

      <details>
        <summary>Conditions</summary>
        <form
          className="stack"
          onSubmit={(event) => {
            event.preventDefault();
            save("/api/conditions", condition, () => setCondition({ name: "", status: "active", recorded_on: "", notes: "" }));
          }}
        >
          <input placeholder="Condition" value={condition.name} onChange={(event) => setCondition({ ...condition, name: event.target.value })} />
          <div className="form-grid">
            <select value={condition.status} onChange={(event) => setCondition({ ...condition, status: event.target.value })}>
              <option value="active">Active</option>
              <option value="past">Past</option>
              <option value="resolved">Resolved</option>
              <option value="unknown">Unknown</option>
            </select>
            <input type="date" value={condition.recorded_on} onChange={(event) => setCondition({ ...condition, recorded_on: event.target.value })} />
          </div>
          <textarea placeholder="Notes" value={condition.notes} onChange={(event) => setCondition({ ...condition, notes: event.target.value })} rows={2} />
          <button className="primary full" type="submit">
            <Plus size={18} />
            Save condition
          </button>
        </form>
        <MiniList items={snapshot.conditions.slice(0, 8)} titleKey="name" detailKey="status" deleteKey="condition_id" deletePath="/api/conditions" onDelete={remove} />
      </details>

      <details>
        <summary>Medications</summary>
        <form
          className="stack"
          onSubmit={(event) => {
            event.preventDefault();
            save("/api/medications", medication, () => setMedication({ name: "", dose: "", schedule: "", reason: "", started_on: "", notes: "" }));
          }}
        >
          <input placeholder="Medication" value={medication.name} onChange={(event) => setMedication({ ...medication, name: event.target.value })} />
          <div className="form-grid">
            <input placeholder="Dose" value={medication.dose} onChange={(event) => setMedication({ ...medication, dose: event.target.value })} />
            <input placeholder="Schedule" value={medication.schedule} onChange={(event) => setMedication({ ...medication, schedule: event.target.value })} />
          </div>
          <input placeholder="Reason" value={medication.reason} onChange={(event) => setMedication({ ...medication, reason: event.target.value })} />
          <button className="primary full" type="submit">
            <Plus size={18} />
            Save medication
          </button>
        </form>
        <MiniList items={snapshot.medications.slice(0, 8)} titleKey="name" detailKey="dose" deleteKey="medication_id" deletePath="/api/medications" onDelete={remove} />
      </details>

      <details>
        <summary>Allergies</summary>
        <form
          className="stack"
          onSubmit={(event) => {
            event.preventDefault();
            save("/api/allergies", allergy, () => setAllergy({ name: "", reaction: "", severity: "unknown", allergy_type: "other", confirmed: true, notes: "" }));
          }}
        >
          <input placeholder="Allergy" value={allergy.name} onChange={(event) => setAllergy({ ...allergy, name: event.target.value })} />
          <input placeholder="Reaction" value={allergy.reaction} onChange={(event) => setAllergy({ ...allergy, reaction: event.target.value })} />
          <div className="form-grid">
            <select value={allergy.severity} onChange={(event) => setAllergy({ ...allergy, severity: event.target.value })}>
              <option value="unknown">Unknown</option>
              <option value="mild">Mild</option>
              <option value="moderate">Moderate</option>
              <option value="severe">Severe</option>
            </select>
            <select value={allergy.allergy_type} onChange={(event) => setAllergy({ ...allergy, allergy_type: event.target.value })}>
              <option value="drug">Drug</option>
              <option value="food">Food</option>
              <option value="environmental">Environmental</option>
              <option value="other">Other</option>
            </select>
          </div>
          <button className="primary full" type="submit">
            <Plus size={18} />
            Save allergy
          </button>
        </form>
        <MiniList items={snapshot.allergies.slice(0, 8)} titleKey="name" detailKey="reaction" deleteKey="allergy_id" deletePath="/api/allergies" onDelete={remove} />
      </details>

      <details>
        <summary>Clinical relationships</summary>
        {snapshot.clinical_relationships?.length ? (
          <div className="mini-list">
            {snapshot.clinical_relationships.slice(0, 10).map((relationship) => (
              <div key={clean(relationship.relationship_id)}>
                <span>
                  {clean(relationship.relation_class, "association").replace(/_/g, " ")} · {clean(relationship.certainty, "recorded association").replace(/_/g, " ")}
                </span>
                <strong>
                  {clean(relationship.source_name)} {clean(relationship.relation).replace(/_/g, " ")} {clean(relationship.target_name)}
                </strong>
                {relationship.source !== "structured_record" && (
                  <button
                    className="icon-button"
                    onClick={() => remove(`/api/relationships/${relationship.relationship_id}`)}
                    title="Remove"
                    aria-label="Remove"
                  >
                    <Trash2 size={15} />
                  </button>
                )}
              </div>
            ))}
          </div>
        ) : (
          <p className="muted">No patient-reported relationships saved yet.</p>
        )}
      </details>

      <details>
        <summary>Vitals and labs</summary>
        <form
          className="stack"
          onSubmit={(event) => {
            event.preventDefault();
            save("/api/vitals", vitals, () => setVitals({ ...vitals, value: "", notes: "" }));
          }}
        >
          <select
            value={vitals.type}
            onChange={(event) => {
              const option = VITAL_OPTIONS.find((item) => item[0] === event.target.value);
              setVitals({ ...vitals, type: event.target.value, unit: option?.[2] ?? "" });
            }}
          >
            {VITAL_OPTIONS.map(([key, label]) => (
              <option key={key} value={key}>
                {label}
              </option>
            ))}
          </select>
          <div className="form-grid">
            <input placeholder={VITAL_OPTIONS.find((item) => item[0] === vitals.type)?.[3]} value={vitals.value} onChange={(event) => setVitals({ ...vitals, value: event.target.value })} />
            <input placeholder="Unit" value={vitals.unit} onChange={(event) => setVitals({ ...vitals, unit: event.target.value })} />
          </div>
          <input type="date" value={vitals.recorded_on} onChange={(event) => setVitals({ ...vitals, recorded_on: event.target.value })} />
          <button className="primary full" type="submit">
            <Plus size={18} />
            Save reading
          </button>
        </form>
        <MiniList items={snapshot.vitals.slice(0, 5)} titleKey="type" detailKey="value" deleteKey="vitals_id" deletePath="/api/vitals" onDelete={remove} transformTitle={vitalLabel} />
      </details>

      {compact && !!recentItems.length && (
        <div className="mini-list">
          {recentItems.map((item) => (
            <div key={`${item.kind}-${item.id}`}>
              <span>{item.kind}</span>
              <strong>{item.title}</strong>
              <button className="icon-button" onClick={() => remove(item.path)} title="Remove" aria-label="Remove">
                <Trash2 size={15} />
              </button>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function MiniList({
  items,
  titleKey,
  detailKey,
  deleteKey,
  deletePath,
  onDelete,
  transformTitle
}: {
  items: Dict<any>[];
  titleKey: string;
  detailKey: string;
  deleteKey: string;
  deletePath: string;
  onDelete: (path: string) => Promise<void>;
  transformTitle?: (value: unknown) => string;
}) {
  if (!items.length) {
    return <p className="muted">No saved records yet.</p>;
  }
  return (
    <div className="mini-list">
      {items.map((item) => (
        <div key={clean(item[deleteKey])}>
          <span>{clean(item[detailKey])}</span>
          <strong>{transformTitle ? transformTitle(item[titleKey]) : clean(item[titleKey])}</strong>
          <button className="icon-button" onClick={() => onDelete(`${deletePath}/${item[deleteKey]}`)} title="Remove" aria-label="Remove">
            <Trash2 size={15} />
          </button>
        </div>
      ))}
    </div>
  );
}

function TimelineView({ snapshot }: { snapshot: Snapshot }) {
  const [tab, setTab] = useState<"summary" | "timeline" | "trends">("summary");
  const events = useMemo(() => buildTimelineEvents(snapshot), [snapshot]);
  const insights = useMemo(() => buildTrendInsights(snapshot), [snapshot]);
  const activeConditions = snapshot.conditions.filter((item) => clean(item.status, "unknown").toLowerCase() === "active");
  const latestVitalsByType = unique(snapshot.vitals.map((item) => item.type)).map((type) =>
    snapshot.vitals.find((entry) => clean(entry.type).toLowerCase() === type.toLowerCase())
  );

  return (
    <div className="view-stack">
      <section className="workspace-band">
        <div>
          <span className="eyebrow">Health timeline</span>
          <h2>Saved context and trends.</h2>
          <p>Review the account record separately from the chat thread.</p>
        </div>
      </section>

      <div className="segmented tabs">
        <button className={tab === "summary" ? "active" : ""} onClick={() => setTab("summary")}>
          Summary
        </button>
        <button className={tab === "timeline" ? "active" : ""} onClick={() => setTab("timeline")}>
          Timeline
        </button>
        <button className={tab === "trends" ? "active" : ""} onClick={() => setTab("trends")}>
          Trends
        </button>
      </div>

      {tab === "summary" && (
        <section className="summary-grid">
          <SummaryCard title="Current health snapshot" count={`${activeConditions.length} active`}>
            {activeConditions.length ? activeConditions.slice(0, 6).map((item) => clean(item.name)).join(", ") : "No active conditions saved yet."}
          </SummaryCard>
          <SummaryCard title="Recent readings" count={`${latestVitalsByType.length} types`}>
            {latestVitalsByType.length
              ? latestVitalsByType
                  .slice(0, 8)
                  .map((item) => `${vitalLabel(item?.type)} ${clean(item?.value)} ${clean(item?.unit)}`)
                  .join("; ")
              : "No readings saved yet."}
          </SummaryCard>
          <SummaryCard title="Medicines and allergies" count={`${snapshot.medications.length + snapshot.allergies.length} saved`}>
            {unique([...snapshot.medications.map((item) => item.name), ...snapshot.allergies.map((item) => `Allergy: ${item.name}`)]).slice(0, 8).join("; ") ||
              "No medications or allergies recorded."}
          </SummaryCard>
          <SummaryCard title="Latest triage" count={clean(snapshot.latest_triage.urgency_level, "Routine")}>
            {clean(snapshot.latest_triage.next_step, "No triage summary saved yet.")}
          </SummaryCard>
          <SummaryCard title="Key patterns" count={`${insights.length} found`}>
            {insights.length ? insights.map((item) => item.title).join("; ") : "Add readings over time to unlock trend cards."}
          </SummaryCard>
        </section>
      )}

      {tab === "timeline" && (
        <section className="timeline-list">
          {events.length ? (
            events.slice(0, 30).map((event, index) => (
              <article key={`${event.type}-${index}`} className="timeline-event">
                <time>{formatDate(event.when)}</time>
                <div>
                  <span>{clean(event.type)}</span>
                  <strong>{clean(event.title)}</strong>
                  <p>{clean(event.detail)}</p>
                </div>
              </article>
            ))
          ) : (
            <div className="empty-state">No timeline events yet.</div>
          )}
        </section>
      )}

      {tab === "trends" && (
        <section className="trend-grid">
          {insights.length ? (
            insights.map((insight) => (
              <article className="trend-card" key={insight.title}>
                <span>Pattern found</span>
                <strong>{insight.title}</strong>
                <p>{insight.body}</p>
                <TrendChart snapshot={snapshot} detail={insight.detail} />
              </article>
            ))
          ) : (
            <div className="empty-state">No trend warnings yet. Add multiple readings over time.</div>
          )}
          <TrendLibrary snapshot={snapshot} />
        </section>
      )}
    </div>
  );
}

function SummaryCard({ title, count, children }: { title: string; count: string; children: React.ReactNode }) {
  return (
    <article className="summary-card">
      <span>{count}</span>
      <strong>{title}</strong>
      <p>{children}</p>
    </article>
  );
}

function MemorySummaryPanel({ memory }: { memory: Dict<any> }) {
  const summary = clean(memory?.summary);
  const sections = useMemo(() => parseMemorySections(summary), [summary]);
  const updatedAt = clean(memory?.updated_at);
  return (
    <article className="summary-card memory-summary-card">
      <span>{updatedAt ? formatDate(updatedAt) : "Not yet updated"}</span>
      <strong>Clinical context</strong>
      {!summary ? (
        <p>No clinical context yet. It builds automatically as you use the chat.</p>
      ) : sections.length === 0 ? (
        <p>{summary}</p>
      ) : (
        <div className="memory-sections">
          {sections.map(({ label, value, empty }) => (
            <div key={label} className="memory-row">
              <span className="memory-label">{label}</span>
              <span className={empty ? "memory-value muted" : "memory-value"}>{value}</span>
            </div>
          ))}
        </div>
      )}
    </article>
  );
}

function TrendLibrary({ snapshot }: { snapshot: Snapshot }) {
  const vitalTypes = unique(snapshot.vitals.map((entry) => entry.type));
  const symptoms = unique(snapshot.symptom_logs.map((entry) => entry.symptom));
  return (
    <>
      {vitalTypes.map((type) => (
        <article className="trend-card" key={type}>
          <span>{buildSeries(snapshot.vitals, type).length} readings</span>
          <strong>{vitalLabel(type)}</strong>
          <TrendChart snapshot={snapshot} detail={type} />
        </article>
      ))}
      {symptoms.map((symptom) => {
        const series = buildSymptomSeries(snapshot.symptom_logs, symptom);
        if (series.length < 2) {
          return null;
        }
        return (
          <article className="trend-card" key={symptom}>
            <span>{series.length} readings</span>
            <strong>{symptom}</strong>
            <TrendChart snapshot={snapshot} detail={`symptom:${symptom}`} />
          </article>
        );
      })}
    </>
  );
}

function TrendChart({ snapshot, detail }: { snapshot: Snapshot; detail: string }) {
  const series = detail.startsWith("symptom:")
    ? buildSymptomSeries(snapshot.symptom_logs, detail.split(":")[1])
    : buildSeries(snapshot.vitals, detail).map((point) => ({ date: point.date, value: point.value, secondValue: point.secondValue }));
  if (series.length < 2) {
    return <p className="muted">No chartable pattern yet.</p>;
  }
  return <TinyChart series={series} />;
}

function TinyChart({ series }: { series: { date: string; value: number; secondValue?: number }[] }) {
  const width = 360;
  const height = 140;
  const values = series.flatMap((point) => [point.value, point.secondValue].filter((value): value is number => typeof value === "number"));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const point = (value: number, index: number) => {
    const x = series.length === 1 ? width / 2 : (index / (series.length - 1)) * (width - 28) + 14;
    const y = height - 22 - ((value - min) / span) * (height - 42);
    return `${x},${y}`;
  };
  const primary = series.map((entry, index) => point(entry.value, index)).join(" ");
  const secondary = series.filter((entry) => typeof entry.secondValue === "number").map((entry, index) => point(entry.secondValue as number, index)).join(" ");

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="tiny-chart" role="img" aria-label="Trend chart">
      <line x1="12" x2={width - 12} y1={height - 20} y2={height - 20} />
      <polyline points={primary} />
      {secondary && <polyline points={secondary} className="secondary" />}
      {series.map((entry, index) => (
        <circle key={`${entry.date}-${index}`} cx={point(entry.value, index).split(",")[0]} cy={point(entry.value, index).split(",")[1]} r="3" />
      ))}
    </svg>
  );
}

function TrialsView({
  snapshot,
  setSnapshot,
  setNotice
}: {
  snapshot: Snapshot;
  setSnapshot: (snapshot: Snapshot) => void;
  setNotice: (notice: string) => void;
}) {
  const [location, setLocation] = useState(snapshot.trial_search_result?.location || "United Kingdom");
  const [busy, setBusy] = useState(false);
  const result = snapshot.trial_search_result as TrialSearchResult | null | undefined;

  async function searchTrials() {
    setBusy(true);
    try {
      const response = await apiRequest<{ result: TrialSearchResult; snapshot: Snapshot }>("/api/trials/search", {
        method: "POST",
        body: JSON.stringify({ location, max_results: 10 })
      });
      setSnapshot(response.snapshot);
      setNotice(`Found ${response.result.trials.length} ranked trial records.`);
    } catch (caught) {
      setNotice(caught instanceof Error ? caught.message : "Trial search failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="view-stack">
      <section className="workspace-band">
        <div>
          <span className="eyebrow">Clinical trials</span>
          <h2>Find recruiting studies.</h2>
          <p>Results use saved health context and need confirmation by a study team or clinician.</p>
        </div>
        <div className="trial-search">
          <select value={location} onChange={(event) => setLocation(event.target.value)}>
            {COUNTRIES.map((country) => (
              <option key={country}>{country}</option>
            ))}
          </select>
          <button className="primary" onClick={searchTrials} disabled={busy}>
            <Search size={18} />
            {busy ? "Searching" : "Find trials"}
          </button>
        </div>
      </section>

      <section className="metric-grid">
        <div className="metric-card">
          <Activity size={20} />
          <strong>{snapshot.metrics.symptoms}</strong>
          <span>Symptoms</span>
        </div>
        <div className="metric-card">
          <Pill size={20} />
          <strong>{snapshot.metrics.medications}</strong>
          <span>Medications</span>
        </div>
        <div className="metric-card">
          <HeartPulse size={20} />
          <strong>{snapshot.metrics.vitals}</strong>
          <span>Vitals/labs</span>
        </div>
        <div className="metric-card">
          <ClipboardList size={20} />
          <strong>{snapshot.metrics.triage_records}</strong>
          <span>Triage records</span>
        </div>
      </section>

      {result?.error && <div className="notice warn">{result.error}</div>}
      {result && !result.error && (
        <section className="trial-results">
          {result.clinical_context?.topic && (
            <div className="notice info">
              Search basis: {String(result.clinical_context.topic)} ({String(result.context_status ?? "reviewed")}).
            </div>
          )}
          <div className="toolbar-card">
            <span>Recruiting only</span>
            <span>Top {result.trials.length} ranked matches</span>
            <span>Country: {result.location}</span>
            <span>{formatTimestamp(result.searched_at) || "Saved search"}</span>
          </div>
          {(result.condition_terms.length > 0 || result.medication_terms.length > 0) && (
            <p className="muted">
              Searches run for {unique([...result.condition_terms, ...result.medication_terms.map((item) => `${item} (drug)`)]).join(", ")}
            </p>
          )}
          {result.trials.length ? (
            result.trials.map((trial, index) => <TrialCard trial={trial} index={index + 1} key={clean(trial.nct_id, String(index))} />)
          ) : (
            <div className="empty-state">No recruiting trials were returned for this context and location.</div>
          )}
        </section>
      )}
    </div>
  );
}

function TrialCard({ trial, index }: { trial: Dict<any>; index: number }) {
  const score = Number(trial.match_score ?? 0);
  const bestLocation = (trial.best_location ?? {}) as Dict<any>;
  const contacts = Array.isArray(trial.contacts) ? trial.contacts : [];
  const officials = Array.isArray(trial.officials) ? trial.officials : [];
  return (
    <article className="trial-card">
      <div className="trial-head">
        <span>#{index} {score >= 80 ? "Strong record match" : "Needs manual review"}</span>
        <strong>{score}%</strong>
      </div>
      <h3>{clean(trial.title, "Untitled trial")}</h3>
      <p>{Array.isArray(trial.conditions) ? trial.conditions.slice(0, 6).join(", ") : "Condition not listed"}</p>
      <div className="trial-chip-row">
        <span>{clean(trial.status, "Recruiting")}</span>
        <span>{clean(trial.phase, "Phase not listed")}</span>
        <span>{clean(trial.study_type, "Study type not listed")}</span>
      </div>
      <div className="trial-detail-grid">
        <div>
          <strong>Location</strong>
          <p>{unique([bestLocation.city, bestLocation.country]).join(", ") || "No trial site listed"}</p>
        </div>
        <div>
          <strong>Hospital / site</strong>
          <p>{clean(bestLocation.facility, "Not listed")}</p>
        </div>
        <div>
          <strong>Official</strong>
          <p>{officials.length ? unique([officials[0].name, officials[0].role, officials[0].affiliation]).join(" - ") : "Not listed"}</p>
        </div>
        <div>
          <strong>Contact</strong>
          <p>{contacts.length ? unique([contacts[0].name, contacts[0].phone, contacts[0].email]).join(" - ") : "No public contact listed"}</p>
        </div>
      </div>
      {trial.url && (
        <a className="open-link" href={String(trial.url)} target="_blank" rel="noreferrer">
          Open ClinicalTrials.gov record
        </a>
      )}
      <details className="details-block">
        <summary>Why this score?</summary>
        <p>Alignment {trial.alignment_score ?? 0}/50, coverage {trial.coverage_score ?? 0}/30, location {trial.location_score ?? 0}/20.</p>
        {trial.llm_reasoning && <p>{String(trial.llm_reasoning)}</p>}
        {Array.isArray(trial.exclusion_risks) && trial.exclusion_risks.length > 0 && <p>Potential exclusion factors: {trial.exclusion_risks.join("; ")}</p>}
      </details>
    </article>
  );
}

// ── Care Plans ───────────────────────────────────────────────────────────────

const URGENCY_META: Record<string, { label: string; color: string; bg: string }> = {
  call_999:   { label: "Call 999 now",    color: "#fff", bg: "#c0392b" },
  a_and_e:    { label: "Go to A&E",       color: "#c0392b", bg: "#fff0ed" },
  gp_same_day:{ label: "See GP today",    color: "#b07d00", bg: "#fff8e5" },
  gp_routine: { label: "Book GP visit",   color: "#1a6b5a", bg: "#edf8f4" },
  self_monitor:{ label: "Self-monitor",   color: "#4a7a8a", bg: "#edf4f3" },
};

const TIME_OF_DAY_ICON: Record<string, string> = {
  morning: "🌅",
  afternoon: "☀️",
  evening: "🌆",
  bedtime: "🌙",
  any: "⏱️",
};

function todayStr() {
  return new Date().toISOString().slice(0, 10);
}

function isTaskDoneToday(task: CarePlanTask): boolean {
  return (task.completed_dates ?? []).includes(todayStr());
}

function planProgress(plan: CarePlan): { done: number; total: number } {
  const today = todayStr();
  const daily = plan.daily_tasks ?? [];
  const done = daily.filter((t) => (t.completed_dates ?? []).includes(today)).length;
  return { done, total: daily.length };
}

function ProgressRing({ done, total, size = 52 }: { done: number; total: number; size?: number }) {
  const r = (size - 8) / 2;
  const circ = 2 * Math.PI * r;
  const pct = total === 0 ? 0 : done / total;
  const dash = circ * pct;
  return (
    <svg width={size} height={size} className="progress-ring">
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#e0eceb" strokeWidth={6} />
      <circle
        cx={size / 2} cy={size / 2} r={r}
        fill="none"
        stroke="var(--primary)"
        strokeWidth={6}
        strokeDasharray={`${dash} ${circ}`}
        strokeLinecap="round"
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
      />
      <text x={size / 2} y={size / 2 + 5} textAnchor="middle" fontSize={11} fill="var(--primary)" fontWeight={700}>
        {total === 0 ? "--" : `${done}/${total}`}
      </text>
    </svg>
  );
}

function CarePlanCard({
  plan,
  onOpen,
  onDelete,
}: {
  plan: CarePlan;
  onOpen: () => void;
  onDelete: () => void;
}) {
  const { done, total } = planProgress(plan);
  const hasAlerts = (plan.missed_care_checklist ?? []).some((i) => i.overdue);

  return (
    <article className="cp-card" onClick={onOpen}>
      <div className="cp-card-top">
        <div className="cp-card-info">
          <h3>{plan.title}</h3>
          <span className="cp-condition-tag">{plan.condition}</span>
        </div>
        <ProgressRing done={done} total={total} />
      </div>
      <p className="cp-card-evidence">{plan.evidence_summary?.slice(0, 120)}…</p>
      <div className="cp-card-footer">
        <span className="cp-card-date">Created {plan.created_at?.slice(0, 10)}</span>
        {hasAlerts && (
          <span className="cp-alert-chip">
            <AlertTriangle size={12} /> Overdue care
          </span>
        )}
        <button
          className="icon-button danger-hover"
          onClick={(e) => { e.stopPropagation(); onDelete(); }}
          title="Remove plan"
        >
          <Trash2 size={15} />
        </button>
      </div>
    </article>
  );
}

function NewPlanPanel({
  snapshot,
  onCreated,
}: {
  snapshot: Snapshot;
  onCreated: (plan: CarePlan) => void;
}) {
  const knownConditions = (snapshot.conditions ?? []).map((c: Dict) => c.name as string).filter(Boolean);
  const suggestFirst = knownConditions.length > 0 ? knownConditions[0] : "";

  const [condition, setCondition] = useState(suggestFirst);
  const [progress, setProgress] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [clarify, setClarify] = useState<{ question: string; options: ClarifyOption[] } | null>(null);
  const [pendingChatSummary, setPendingChatSummary] = useState("");
  const [clarifyOther, setClarifyOther] = useState("");

  const effectiveCondition = condition.trim();

  async function runGenerate(chatSummary: string, clarification?: { question: string; answer: string }) {
    setBusy(true);
    setError("");
    setClarify(null);
    setClarifyOther("");
    setProgress(["Starting evidence search..."]);
    try {
      const plan = await generateCarePlan(
        effectiveCondition,
        chatSummary,
        (msg) => setProgress((prev) => [...prev, msg]),
        (question, options) => {
          setPendingChatSummary(chatSummary);
          setClarify({ question, options });
        },
        clarification
      );
      if (plan) onCreated(plan);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleGenerate() {
    if (!effectiveCondition) return;
    const recentMessages = (snapshot.chat_history ?? []).slice(-6);
    const chatSummary = recentMessages.map((m: Message) => `${m.role}: ${m.content}`).join("\n");
    await runGenerate(chatSummary);
  }

  async function handleClarifyChoice(option: ClarifyOption) {
    if (!clarify) return;
    await runGenerate(
      `${pendingChatSummary}\n\nuser: ${option.prompt}`,
      { question: clarify.question, answer: option.display }
    );
  }

  async function handleClarifyOther() {
    const answer = clarifyOther.trim();
    if (!answer || !clarify) return;
    await runGenerate(
      `${pendingChatSummary}\n\nuser: ${answer}`,
      { question: clarify.question, answer }
    );
  }

  return (
    <div className="cp-new-panel">
      <div className="cp-new-header">
        <Sparkles size={22} color="var(--primary)" />
        <div>
          <h3>Create a care plan</h3>
          <p>The AI agent searches NHS/NICE guidelines and clinical evidence to build a personalised plan.</p>
        </div>
      </div>

      {!clarify && knownConditions.length > 0 && (
        <div className="cp-condition-chips">
          {knownConditions.map((c) => (
            <button
              key={c}
              className={`cp-chip${effectiveCondition === c ? " active" : ""}`}
              onClick={() => setCondition(c)}
              type="button"
            >
              {c}
            </button>
          ))}
        </div>
      )}

      {!clarify && (
        <div className="cp-condition-row">
          <input
            value={condition}
            onChange={(e) => setCondition(e.target.value)}
            placeholder="e.g. Migraine, Psoriasis…"
            className="cp-custom-input"
          />
        </div>
      )}

      {error && <div className="notice error">{error}</div>}

      {clarify && (
        <div className="notice cp-clarify">
          <p>{clarify.question}</p>
          <div className="follow-ups">
            {clarify.options.map((option, i) => (
              <button
                key={i}
                className="follow-up-btn"
                type="button"
                disabled={busy}
                onClick={() => handleClarifyChoice(option)}
              >
                {option.display}
              </button>
            ))}
          </div>
          <div className="cp-condition-row cp-clarify-other">
            <span className="cp-or">Or describe it yourself:</span>
            <input
              value={clarifyOther}
              onChange={(e) => setClarifyOther(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") handleClarifyOther(); }}
              placeholder="Type your own answer…"
              className="cp-custom-input"
              disabled={busy}
            />
            <button
              className="secondary"
              type="button"
              disabled={busy || !clarifyOther.trim()}
              onClick={handleClarifyOther}
            >
              Continue
            </button>
          </div>
        </div>
      )}

      {busy && (
        <div className="cp-progress-log">
          {progress.map((msg, i) => (
            <div key={i} className={`cp-progress-line${i === progress.length - 1 ? " active" : " done"}`}>
              {i === progress.length - 1 ? <RefreshCw size={13} className="spin" /> : <CheckCircle2 size={13} />}
              {msg}
            </div>
          ))}
        </div>
      )}

      {!clarify && (
        <button className="primary" disabled={busy || !effectiveCondition} onClick={handleGenerate} type="button">
          <Sparkles size={16} />
          {busy ? "Building plan…" : "Generate evidence-based plan"}
        </button>
      )}
    </div>
  );
}

type PlanTab = "tasks" | "lifestyle" | "meds" | "labs" | "escalation" | "gp-prep" | "after-visit";

function CarePlanDetail({
  plan: initialPlan,
  onBack,
  onUpdated,
}: {
  plan: CarePlan;
  onBack: () => void;
  onUpdated: (plan: CarePlan) => void;
}) {
  const [plan, setPlan] = useState(initialPlan);
  const [tab, setTab] = useState<PlanTab>("tasks");
  const [gpBusy, setGpBusy] = useState(false);
  const [afterVisitText, setAfterVisitText] = useState("");
  const [savingNote, setSavingNote] = useState(false);

  async function handleToggle(taskId: string, done: boolean) {
    try {
      const updated = await toggleCarePlanTask(plan.id, taskId, done);
      setPlan(updated);
      onUpdated(updated);
    } catch { /* ignore */ }
  }

  async function handleGpPrep() {
    setGpBusy(true);
    try {
      const { plan: updated } = await generateGpPrep(plan.id);
      setPlan(updated);
      onUpdated(updated);
      setTab("gp-prep");
    } catch { /* ignore */ } finally {
      setGpBusy(false);
    }
  }

  async function handleAfterVisit() {
    if (!afterVisitText.trim()) return;
    setSavingNote(true);
    try {
      const updated = await addAfterVisitNote(plan.id, afterVisitText.trim());
      setPlan(updated);
      onUpdated(updated);
      setAfterVisitText("");
    } catch { /* ignore */ } finally {
      setSavingNote(false);
    }
  }

  const { done, total } = planProgress(plan);
  const tabs: { id: PlanTab; label: string }[] = [
    { id: "tasks", label: "Daily Plan" },
    { id: "lifestyle", label: "Lifestyle" },
    { id: "meds", label: "Medications" },
    { id: "labs", label: "Monitoring" },
    { id: "escalation", label: "Warning Signs" },
    { id: "gp-prep", label: "GP Prep" },
    { id: "after-visit", label: "After Visit" },
  ];

  return (
    <div className="cp-detail">
      <div className="cp-detail-header">
        <button className="link-btn" onClick={onBack} type="button">← All plans</button>
        <div className="cp-detail-title">
          <h2>{plan.title}</h2>
          <span className="cp-condition-tag">{plan.condition}</span>
        </div>
        <div className="cp-detail-actions">
          <ProgressRing done={done} total={total} size={48} />
          <button className="secondary" onClick={handleGpPrep} disabled={gpBusy} type="button">
            <Stethoscope size={15} />
            {gpBusy ? "Preparing…" : "Prepare for GP"}
          </button>
        </div>
      </div>

      {plan.safety_notes && (
        <div className="notice warn cp-safety">
          <AlertTriangle size={15} /> {plan.safety_notes}
        </div>
      )}
      {plan.validation?.status === "review_required" && (
        <div className="notice info cp-safety">
          <ShieldCheck size={15} /> This plan was paused by the clinical-context safety check. No condition-specific tasks were issued.
        </div>
      )}

      <div className="cp-tabs">
        {tabs.map((t) => (
          <button
            key={t.id}
            className={`cp-tab${tab === t.id ? " active" : ""}`}
            onClick={() => setTab(t.id)}
            type="button"
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="cp-tab-content">
        {tab === "tasks" && (
          <div className="cp-tasks-view">
            <div className="cp-section-label">
              <Activity size={15} /> Today's tasks ({done}/{total} done)
            </div>
            {plan.daily_tasks.map((task) => {
              const done = isTaskDoneToday(task);
              return (
                <label key={task.id} className={`cp-task-row${done ? " done" : ""}`}>
                  <input
                    type="checkbox"
                    checked={done}
                    onChange={(e) => handleToggle(task.id, e.target.checked)}
                  />
                  <span className="cp-task-time">{TIME_OF_DAY_ICON[task.time_of_day ?? "any"]}</span>
                  <div>
                    <span className="cp-task-text">{task.text}</span>
                    {task.rationale && <p className="cp-task-rationale">{task.rationale}</p>}
                  </div>
                </label>
              );
            })}
            {plan.weekly_tasks.length > 0 && (
              <>
                <div className="cp-section-label" style={{ marginTop: "1.2rem" }}>
                  <CalendarClock size={15} /> This week
                </div>
                {plan.weekly_tasks.map((task) => (
                  <div key={task.id} className="cp-weekly-task">
                    <span>{task.text}</span>
                    {task.rationale && <p className="cp-task-rationale">{task.rationale}</p>}
                  </div>
                ))}
              </>
            )}
            {plan.goals.length > 0 && (
              <>
                <div className="cp-section-label" style={{ marginTop: "1.4rem" }}>
                  <CheckCircle2 size={15} /> Your goals
                </div>
                {plan.goals.map((g) => (
                  <div key={g.id} className="cp-goal-row">
                    <span>{g.text}</span>
                    {g.metric && <span className="cp-goal-metric">{g.metric}</span>}
                    {g.target_months && <span className="cp-goal-target">Target: {g.target_months} months</span>}
                  </div>
                ))}
              </>
            )}
          </div>
        )}

        {tab === "lifestyle" && (
          <div className="cp-lifestyle-view">
            {(Object.entries(plan.lifestyle ?? {}) as [string, string][])
              .filter(([, v]) => v)
              .map(([key, value]) => (
                <div key={key} className="cp-lifestyle-block">
                  <h4 className="cp-lifestyle-heading">{key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}</h4>
                  <p>{value}</p>
                </div>
              ))}
            {plan.missed_care_checklist.length > 0 && (
              <div className="cp-missed-section">
                <h4 className="cp-lifestyle-heading">
                  <AlertTriangle size={14} /> Preventive care checklist
                </h4>
                {plan.missed_care_checklist.map((item: MissedCareItem) => (
                  <div key={item.id} className={`cp-missed-item${item.overdue ? " overdue" : ""}`}>
                    <span>{item.item}</span>
                    <span className="cp-freq">Every {item.frequency_months}m</span>
                    {item.overdue && <span className="cp-overdue-tag">Overdue</span>}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {tab === "meds" && (
          <div className="cp-meds-view">
            {plan.medication_reminders.length === 0 ? (
              <p className="cp-empty">No medication reminders in this plan.</p>
            ) : (
              plan.medication_reminders.map((med: MedReminder) => (
                <div key={med.id} className="cp-med-card">
                  <div className="cp-med-name">
                    <Pill size={16} color="var(--primary)" />
                    <strong>{med.medication}</strong>
                    {med.dose && <span className="cp-med-dose">{med.dose}</span>}
                  </div>
                  <div className="cp-med-timing">{med.timing}</div>
                  {med.notes && <p className="cp-med-notes">{med.notes}</p>}
                </div>
              ))
            )}
          </div>
        )}

        {tab === "labs" && (
          <div className="cp-labs-view">
            {plan.lab_reminders.map((lab: LabReminder) => (
              <div key={lab.id} className="cp-lab-card">
                <div className="cp-lab-name">
                  <Activity size={15} color="var(--primary)" />
                  <strong>{lab.test}</strong>
                </div>
                <div className="cp-lab-meta">
                  <span>Every {lab.frequency_months} months</span>
                  {lab.target_value && <span className="cp-lab-target">Target: {lab.target_value}</span>}
                </div>
                {lab.notes && <p className="cp-lab-notes">{lab.notes}</p>}
              </div>
            ))}
          </div>
        )}

        {tab === "escalation" && (
          <div className="cp-escalation-view">
            <p className="cp-escalation-intro">
              If you experience any of the following, act as described. Always call 999 for life-threatening emergencies.
            </p>
            {plan.escalation_thresholds.map((esc: EscalationThreshold) => {
              const meta = URGENCY_META[esc.urgency ?? "gp_routine"];
              return (
                <div
                  key={esc.id}
                  className="cp-esc-card"
                  style={{ borderLeftColor: meta.bg === "#fff0ed" ? "var(--danger)" : meta.bg === "#fff8e5" ? "var(--gold)" : "var(--primary)" }}
                >
                  <div className="cp-esc-top">
                    <strong>{esc.symptom}</strong>
                    <span className="cp-urgency-badge" style={{ background: meta.bg, color: meta.color }}>
                      {meta.label}
                    </span>
                  </div>
                  {esc.threshold && <p className="cp-esc-threshold">Threshold: {esc.threshold}</p>}
                  <p className="cp-esc-action">{esc.action}</p>
                </div>
              );
            })}
          </div>
        )}

        {tab === "gp-prep" && (
          <div className="cp-gp-view">
            {plan.gp_prep_summary ? (
              <div className="cp-gp-content">
                <ReactMarkdown>{plan.gp_prep_summary}</ReactMarkdown>
                <button className="secondary" onClick={handleGpPrep} disabled={gpBusy} type="button">
                  <RefreshCw size={14} />
                  {gpBusy ? "Refreshing…" : "Regenerate"}
                </button>
              </div>
            ) : (
              <div className="cp-gp-empty">
                <Stethoscope size={36} color="var(--primary)" strokeWidth={1.5} />
                <p>Generate a personalised GP appointment guide based on your current plan, goals, and medications.</p>
                <button className="primary" onClick={handleGpPrep} disabled={gpBusy} type="button">
                  <Stethoscope size={16} />
                  {gpBusy ? "Preparing…" : "Prepare for my GP appointment"}
                </button>
              </div>
            )}
          </div>
        )}

        {tab === "after-visit" && (
          <div className="cp-after-view">
            <div className="cp-after-input-row">
              <textarea
                value={afterVisitText}
                onChange={(e) => setAfterVisitText(e.target.value)}
                placeholder="Log what your GP said, new instructions, or next steps…"
                rows={3}
              />
              <button className="primary" disabled={savingNote || !afterVisitText.trim()} onClick={handleAfterVisit} type="button">
                Save note
              </button>
            </div>
            {(plan.after_visit_notes ?? []).length === 0 ? (
              <p className="cp-empty">No after-visit notes yet.</p>
            ) : (
              [...(plan.after_visit_notes ?? [])].reverse().map((note, i) => (
                <div key={i} className="cp-after-note">
                  <span className="cp-after-date">{note.date}</span>
                  <p>{note.text}</p>
                </div>
              ))
            )}
          </div>
        )}
      </div>

      <div className="cp-evidence-footer">
        <ShieldCheck size={13} />
        {plan.evidence_summary}
      </div>
    </div>
  );
}

function CarePlanScreen({ snapshot }: { snapshot: Snapshot }) {
  const [plans, setPlans] = useState<CarePlan[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<CarePlan | null>(null);
  const [showNew, setShowNew] = useState(false);

  useEffect(() => {
    listCarePlans()
      .then(setPlans)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  function handleCreated(plan: CarePlan) {
    setPlans((prev) => [plan, ...prev]);
    setSelected(plan);
    setShowNew(false);
  }

  function handleUpdated(plan: CarePlan) {
    setPlans((prev) => prev.map((p) => (p.id === plan.id ? plan : p)));
    setSelected(plan);
  }

  async function handleDelete(planId: string) {
    await deleteCarePlan(planId);
    setPlans((prev) => prev.filter((p) => p.id !== planId));
    if (selected?.id === planId) setSelected(null);
  }

  if (selected) {
    return (
      <CarePlanDetail
        plan={selected}
        onBack={() => setSelected(null)}
        onUpdated={handleUpdated}
      />
    );
  }

  return (
    <div className="cp-screen">
      <div className="cp-screen-header">
        <div>
          <h2>Care Plans</h2>
          <p>Personalised, evidence-based plans built from NHS/NICE guidelines and your health context.</p>
        </div>
        <button className="primary" onClick={() => setShowNew(true)} type="button">
          <Plus size={16} /> New plan
        </button>
      </div>

      {showNew && (
        <NewPlanPanel snapshot={snapshot} onCreated={handleCreated} />
      )}

      {loading ? (
        <div className="cp-loading">Building your care plans view…</div>
      ) : plans.length === 0 && !showNew ? (
        <div className="cp-empty-state">
          <ListChecks size={48} strokeWidth={1} color="var(--primary)" />
          <h3>No care plans yet</h3>
          <p>Create your first plan, the AI agent will search NHS guidelines and build a comprehensive, personalised care roadmap.</p>
          <button className="primary" onClick={() => setShowNew(true)} type="button">
            <Sparkles size={16} /> Create your first plan
          </button>
        </div>
      ) : (
        <div className="cp-grid">
          {plans.map((plan) => (
            <CarePlanCard
              key={plan.id}
              plan={plan}
              onOpen={() => setSelected(plan)}
              onDelete={() => handleDelete(plan.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default App;
