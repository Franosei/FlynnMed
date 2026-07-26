import type { Dict, Snapshot } from "./types";

export function clean(value: unknown, fallback = ""): string {
  const text = String(value ?? "").trim();
  return text || fallback;
}

export function isClinicianRole(role: string | undefined): boolean {
  const normalized = (role ?? "").toLowerCase().trim();
  return [
    "doctor / physician",
    "nurse",
    "midwife",
    "physiotherapist",
    "other clinician"
  ].includes(normalized);
}

export function isPatientRole(role: string | undefined): boolean {
  return !isClinicianRole(role);
}

export function formatDate(value?: unknown): string {
  const text = clean(value);
  if (!text) {
    return "Date not recorded";
  }
  const parsed = new Date(text);
  if (Number.isNaN(parsed.getTime())) {
    return text;
  }
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric"
  }).format(parsed);
}

export function formatTimestamp(value?: unknown): string {
  const text = clean(value);
  if (!text) {
    return "";
  }
  const parsed = new Date(text);
  if (Number.isNaN(parsed.getTime())) {
    return text;
  }
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  }).format(parsed);
}

export function unique(values: unknown[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  values.forEach((value) => {
    const text = clean(value);
    const key = text.toLowerCase();
    if (text && !seen.has(key)) {
      seen.add(key);
      result.push(text);
    }
  });
  return result;
}

export function vitalLabel(type: unknown): string {
  const key = clean(type).toLowerCase();
  const overrides: Record<string, string> = {
    bmi: "BMI",
    hba1c: "HbA1c",
    egfr: "eGFR",
    mcv: "MCV",
    mch: "MCH",
    mchc: "MCHC",
    alt: "ALT",
    ast: "AST",
    alp: "ALP",
    ggt: "GGT",
    ldh: "LDH",
    crp: "CRP",
    esr: "ESR",
    tsh: "TSH",
    inr: "INR",
    psa: "PSA",
    bnp: "BNP",
    aptt: "APTT",
    nt_probnp: "NT-proBNP",
    b12: "Vitamin B12",
    free_t4: "Free T4",
    free_t3: "Free T3",
    d_dimer: "D-Dimer",
    peak_expiratory_flow: "Peak Expiratory Flow",
    peak_urinary_flow_rate: "Peak Urinary Flow Rate (Qmax)",
    peak_flow: "Peak Flow (unspecified -- verify respiratory vs. urology)"
  };
  return overrides[key] ?? key.replace(/_/g, " ").replace(/\b\w/g, (match) => match.toUpperCase());
}

export function firstNumber(value: unknown): number | null {
  const match = clean(value).match(/-?\d+(?:\.\d+)?/);
  return match ? Number(match[0]) : null;
}

export function parseBloodPressure(value: unknown): [number | null, number | null] {
  const match = clean(value).match(/(\d{2,3})\s*\/\s*(\d{2,3})/);
  if (!match) {
    return [null, null];
  }
  return [Number(match[1]), Number(match[2])];
}

export function buildTimelineEvents(snapshot: Snapshot): Dict<any>[] {
  const events: Dict<any>[] = [];

  snapshot.uploads.forEach((upload) => {
    events.push({
      when: upload.uploaded_at,
      type: "Uploaded document",
      title: clean(upload.file, "Document uploaded"),
      detail: upload.summary_available ? "Summary available" : "Saved to account"
    });
  });

  snapshot.document_summaries.forEach((summary) => {
    events.push({
      when: summary.updated_at,
      type: "Document indexed",
      title: clean(summary.file, "Document"),
      detail: "Document summary refreshed"
    });
  });

  snapshot.symptom_logs.forEach((entry) => {
    events.push({
      when: entry.logged_for || entry.created_at,
      type: "Symptom logged",
      title: clean(entry.symptom, "Symptom"),
      detail: `Severity ${entry.severity ?? 0}/10${entry.triggers ? ` - ${entry.triggers}` : ""}`
    });
  });

  snapshot.medications.forEach((medication) => {
    events.push({
      when: medication.updated_at || medication.created_at,
      type: "Medication changed",
      title: clean(medication.name, "Medication"),
      detail: unique([medication.dose, medication.schedule, medication.reason]).join(" - ") || "Medication list updated"
    });
  });

  snapshot.allergies.forEach((allergy) => {
    events.push({
      when: allergy.created_at,
      type: "Allergy recorded",
      title: clean(allergy.name, "Allergy"),
      detail: unique([allergy.severity, allergy.reaction, allergy.allergy_type]).join(" - ")
    });
  });

  snapshot.conditions.forEach((condition) => {
    events.push({
      when: condition.recorded_on || condition.created_at,
      type: "Condition recorded",
      title: clean(condition.name, "Condition"),
      detail: unique([condition.status, condition.notes]).join(" - ")
    });
  });

  snapshot.vitals.forEach((entry) => {
    events.push({
      when: entry.recorded_on || entry.created_at,
      type: "Vitals/labs changed",
      title: vitalLabel(entry.type),
      detail: unique([entry.value, entry.unit]).join(" ")
    });
  });

  snapshot.triage_summaries.forEach((summary) => {
    events.push({
      when: summary.created_at,
      type: "Triage summary saved",
      title: clean(summary.urgency_level, "Routine"),
      detail: clean(summary.next_step, "Self-care")
    });
  });

  return events.sort((a, b) => {
    const da = new Date(clean(a.when)).getTime() || 0;
    const db = new Date(clean(b.when)).getTime() || 0;
    return db - da;
  });
}

export function buildSeries(rows: Dict<any>[], type: string): { date: string; value: number; secondValue?: number }[] {
  return rows
    .filter((entry) => clean(entry.type).toLowerCase() === type)
    .map((entry) => {
      const date = clean(entry.recorded_on || entry.created_at).slice(0, 10);
      if (type === "blood_pressure") {
        const [systolic, diastolic] = parseBloodPressure(entry.value);
        return systolic === null
          ? null
          : {
              date,
              value: systolic,
              secondValue: diastolic ?? undefined
            };
      }
      const value = firstNumber(entry.value);
      return value === null ? null : { date, value };
    })
    .filter(Boolean)
    .sort((a, b) => clean(a!.date).localeCompare(clean(b!.date))) as { date: string; value: number; secondValue?: number }[];
}

export function buildSymptomSeries(rows: Dict<any>[], symptom: string): { date: string; value: number }[] {
  return rows
    .filter((entry) => clean(entry.symptom).toLowerCase() === symptom.toLowerCase())
    .map((entry) => ({
      date: clean(entry.logged_for || entry.created_at).slice(0, 10),
      value: Number(entry.severity ?? 0)
    }))
    .filter((entry) => entry.date)
    .sort((a, b) => a.date.localeCompare(b.date));
}

const CLINICAL_SECTION_LABELS = [
  "Conditions and history",
  "Current treatments and medicines",
  "Recent symptoms or active concerns",
  "Investigations or notable results",
  "Risks, allergies, or safety flags",
  "Care plan and follow-up",
  "Open questions or uncertainties"
] as const;

export function parseMemorySections(summary: string): { label: string; value: string; empty: boolean }[] {
  if (!summary?.trim()) return [];
  const text = summary.trim();
  const found: { label: string; labelStart: number; valueStart: number }[] = [];
  for (const label of CLINICAL_SECTION_LABELS) {
    const idx = text.indexOf(`${label}:`);
    if (idx !== -1) found.push({ label, labelStart: idx, valueStart: idx + label.length + 1 });
  }
  found.sort((a, b) => a.labelStart - b.labelStart);
  return found.map((item, i) => {
    const end = i + 1 < found.length ? found[i + 1].labelStart : text.length;
    const value = text.slice(item.valueStart, end).trim();
    const empty = !value || ["none noted", "none", "n/a", "not noted"].includes(value.toLowerCase());
    return { label: item.label, value: empty ? "None noted" : value, empty };
  });
}

export type TrendInsight = {
  title: string;
  body: string;
  detail: string;
};

function average(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function recentSplit(series: { date: string; value: number }[]): [number[], number[]] {
  if (!series.length) {
    return [[], []];
  }
  const latest = new Date(series[series.length - 1].date).getTime();
  const recentStart = latest - 7 * 24 * 60 * 60 * 1000;
  const recent: number[] = [];
  const prior: number[] = [];
  series.forEach((point) => {
    const time = new Date(point.date).getTime();
    if (time >= recentStart) {
      recent.push(point.value);
    } else {
      prior.push(point.value);
    }
  });
  return [recent, prior];
}

export function buildTrendInsights(snapshot: Snapshot): TrendInsight[] {
  const insights: TrendInsight[] = [];
  const checkAverage = (type: string, title: string, threshold: number, unit = "") => {
    const series = buildSeries(snapshot.vitals, type).map((point) => ({ date: point.date, value: point.value }));
    const [recent, prior] = recentSplit(series);
    if (!recent.length) {
      return;
    }
    const recentAverage = average(recent);
    const priorAverage = prior.length ? average(prior) : null;
    if (recentAverage >= threshold || (priorAverage !== null && recentAverage >= priorAverage + threshold * 0.08)) {
      insights.push({
        title,
        body: `Recent average ${recentAverage.toFixed(recentAverage >= 10 ? 0 : 1)}${unit ? ` ${unit}` : ""}. Review this pattern in context.`,
        detail: type
      });
    }
  };

  checkAverage("blood_pressure", "Blood pressure pattern", 140, "mmHg systolic");
  checkAverage("blood_glucose", "Glucose pattern", 7, "mmol/L");
  checkAverage("egfr", "eGFR pattern", 60);
  checkAverage("hba1c", "HbA1c pattern", 48, "mmol/mol");
  checkAverage("cholesterol_total", "Cholesterol pattern", 5, "mmol/L");

  const weight = buildSeries(snapshot.vitals, "weight");
  if (weight.length >= 2) {
    const change = weight[weight.length - 1].value - weight[0].value;
    if (Math.abs(change) >= 2) {
      insights.push({
        title: "Weight change",
        body: `Weight has ${change > 0 ? "increased" : "decreased"} by ${Math.abs(change).toFixed(1)} kg across the recorded period.`,
        detail: "weight"
      });
    }
  }

  unique(snapshot.symptom_logs.map((entry) => entry.symptom)).forEach((symptom) => {
    const series = buildSymptomSeries(snapshot.symptom_logs, symptom);
    if (series.length >= 2 && series[series.length - 1].value >= 7) {
      insights.push({
        title: `${symptom} severity`,
        body: `Latest recorded severity is ${series[series.length - 1].value}/10. Consider reviewing the symptom pattern.`,
        detail: `symptom:${symptom}`
      });
    }
  });

  return insights.slice(0, 8);
}
