import { describe, expect, it } from "vitest";
import type { Snapshot } from "./types";
import {
  buildSeries,
  buildSymptomSeries,
  buildTimelineEvents,
  buildTrendInsights,
  clean,
  firstNumber,
  formatDate,
  formatTimestamp,
  isClinicianRole,
  isPatientRole,
  parseBloodPressure,
  parseMemorySections,
  unique,
  vitalLabel
} from "./utils";

function emptySnapshot(overrides: Partial<Snapshot> = {}): Snapshot {
  return {
    product: { name: "FlynnMed", tagline: "", subtitle: "", support_email: "" },
    user: "test-user",
    profile: { username: "test-user" },
    metrics: {},
    latest_triage: {},
    chat_history: [],
    uploads: [],
    document_summaries: [],
    symptom_logs: [],
    medications: [],
    allergies: [],
    conditions: [],
    vitals: [],
    triage_summaries: [],
    traces: [],
    audit: [],
    memory: {},
    clinical_notes: [],
    ...overrides
  };
}

describe("clean", () => {
  it("trims strings and falls back on empty input", () => {
    expect(clean("  hello  ")).toBe("hello");
    expect(clean(null)).toBe("");
    expect(clean(undefined, "fallback")).toBe("fallback");
    expect(clean("", "fallback")).toBe("fallback");
    expect(clean(42)).toBe("42");
  });
});

describe("role routing", () => {
  it("recognizes every configured clinician role", () => {
    expect(isClinicianRole("Doctor / Physician")).toBe(true);
    expect(isClinicianRole("Nurse")).toBe(true);
    expect(isClinicianRole("Midwife")).toBe(true);
    expect(isClinicianRole("Physiotherapist")).toBe(true);
    expect(isClinicianRole("Other Clinician")).toBe(true);
  });

  it("keeps patient and caregiver roles in the patient workspace", () => {
    expect(isPatientRole("Patient / Individual")).toBe(true);
    expect(isPatientRole("Caregiver")).toBe(true);
    expect(isPatientRole(undefined)).toBe(true);
  });
});

describe("formatDate", () => {
  it("returns a placeholder for missing dates", () => {
    expect(formatDate(undefined)).toBe("Date not recorded");
    expect(formatDate("")).toBe("Date not recorded");
  });

  it("formats a valid ISO date in en-GB style", () => {
    expect(formatDate("2026-04-13")).toBe("13 Apr 2026");
  });

  it("returns the raw text when the date cannot be parsed", () => {
    expect(formatDate("not-a-date")).toBe("not-a-date");
  });
});

describe("formatTimestamp", () => {
  it("returns an empty string for missing input", () => {
    expect(formatTimestamp(undefined)).toBe("");
  });

  it("includes a time component for valid timestamps", () => {
    const result = formatTimestamp("2026-04-13T09:30:00Z");
    expect(result).toContain("2026");
    expect(result).toMatch(/\d{2}:\d{2}/);
  });
});

describe("unique", () => {
  it("removes case-insensitive duplicates while preserving first-seen casing", () => {
    expect(unique(["Metformin", "metformin", "Aspirin", "", null, "ASPIRIN"])).toEqual([
      "Metformin",
      "Aspirin"
    ]);
  });
});

describe("vitalLabel", () => {
  it("uses known clinical overrides", () => {
    expect(vitalLabel("hba1c")).toBe("HbA1c");
    expect(vitalLabel("egfr")).toBe("eGFR");
    expect(vitalLabel("peak_urinary_flow_rate")).toBe("Peak Urinary Flow Rate (Qmax)");
  });

  it("title-cases unknown vital types", () => {
    expect(vitalLabel("blood_glucose")).toBe("Blood Glucose");
  });
});

describe("firstNumber", () => {
  it("extracts the first numeric value from a string", () => {
    expect(firstNumber("120 mmHg")).toBe(120);
    expect(firstNumber("-3.5")).toBe(-3.5);
  });

  it("returns null when there is no number", () => {
    expect(firstNumber("no numbers here")).toBeNull();
  });
});

describe("parseBloodPressure", () => {
  it("parses systolic/diastolic pairs", () => {
    expect(parseBloodPressure("128/82")).toEqual([128, 82]);
  });

  it("returns [null, null] for unparseable values", () => {
    expect(parseBloodPressure("normal")).toEqual([null, null]);
  });
});

describe("buildSeries", () => {
  it("builds a sorted numeric series for a vital type", () => {
    const rows = [
      { type: "weight", value: "80", recorded_on: "2026-02-01" },
      { type: "weight", value: "78", recorded_on: "2026-01-01" },
      { type: "blood_pressure", value: "120/80", recorded_on: "2026-01-15" }
    ];
    expect(buildSeries(rows, "weight")).toEqual([
      { date: "2026-01-01", value: 78 },
      { date: "2026-02-01", value: 80 }
    ]);
  });

  it("splits blood pressure into systolic/diastolic", () => {
    const rows = [{ type: "blood_pressure", value: "130/85", recorded_on: "2026-01-01" }];
    expect(buildSeries(rows, "blood_pressure")).toEqual([
      { date: "2026-01-01", value: 130, secondValue: 85 }
    ]);
  });
});

describe("buildSymptomSeries", () => {
  it("filters by symptom name case-insensitively and sorts by date", () => {
    const rows = [
      { symptom: "Headache", severity: 6, logged_for: "2026-01-02" },
      { symptom: "headache", severity: 4, logged_for: "2026-01-01" },
      { symptom: "Nausea", severity: 8, logged_for: "2026-01-01" }
    ];
    expect(buildSymptomSeries(rows, "headache")).toEqual([
      { date: "2026-01-01", value: 4 },
      { date: "2026-01-02", value: 6 }
    ]);
  });
});

describe("buildTimelineEvents", () => {
  it("merges all snapshot collections into one sorted timeline", () => {
    const snapshot = emptySnapshot({
      uploads: [{ file: "letter.pdf", uploaded_at: "2026-01-01T00:00:00Z" }],
      medications: [{ name: "Metformin", updated_at: "2026-03-01T00:00:00Z" }],
      symptom_logs: [{ symptom: "Cough", severity: 3, logged_for: "2026-02-01" }]
    });

    const events = buildTimelineEvents(snapshot);
    expect(events).toHaveLength(3);
    // Most recent first
    expect(events[0].type).toBe("Medication changed");
    expect(events[events.length - 1].type).toBe("Uploaded document");
  });

  it("returns an empty array for an empty snapshot", () => {
    expect(buildTimelineEvents(emptySnapshot())).toEqual([]);
  });
});

describe("parseMemorySections", () => {
  it("splits a structured memory summary into labelled sections", () => {
    const summary =
      "Conditions and history: Type 2 diabetes.\n" +
      "Current treatments and medicines: Metformin 500mg.\n" +
      "Recent symptoms or active concerns: None noted";
    const sections = parseMemorySections(summary);
    expect(sections).toHaveLength(3);
    expect(sections[0]).toEqual({
      label: "Conditions and history",
      value: "Type 2 diabetes.",
      empty: false
    });
    expect(sections[2].empty).toBe(true);
    expect(sections[2].value).toBe("None noted");
  });

  it("returns an empty array for blank input", () => {
    expect(parseMemorySections("")).toEqual([]);
  });
});

describe("buildTrendInsights", () => {
  it("flags a raised blood pressure pattern", () => {
    const snapshot = emptySnapshot({
      vitals: [
        { type: "blood_pressure", value: "150/95", recorded_on: "2026-04-01" },
        { type: "blood_pressure", value: "148/92", recorded_on: "2026-04-03" }
      ]
    });
    const insights = buildTrendInsights(snapshot);
    expect(insights.some((insight) => insight.title === "Blood pressure pattern")).toBe(true);
  });

  it("returns no insights for an empty snapshot", () => {
    expect(buildTrendInsights(emptySnapshot())).toEqual([]);
  });
});
