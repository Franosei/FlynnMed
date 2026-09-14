import { beforeEach, describe, expect, it } from "vitest";
import { consumeNdjsonStream, getStoredToken, setStoredToken } from "./api";

describe("token storage", () => {
  beforeEach(() => {
    setStoredToken("");
    localStorage.clear();
  });

  it("returns an empty string when no token is stored", () => {
    expect(getStoredToken()).toBe("");
  });

  it("stores and retrieves a token", () => {
    setStoredToken("abc123");
    expect(getStoredToken()).toBe("abc123");
    expect(localStorage.getItem("flynnmed_token")).toBeNull();
  });

  it("clears the stored token when set with an empty string", () => {
    setStoredToken("abc123");
    setStoredToken("");
    expect(getStoredToken()).toBe("");
  });
});

function ndjsonResponse(body: string): Response {
  return new Response(body, {
    status: 200,
    headers: { "Content-Type": "application/x-ndjson" }
  });
}

describe("clinical stream protocol", () => {
  it("requires a terminal completion event", async () => {
    const events: Array<{ type: string; message?: string }> = [];

    await expect(
      consumeNdjsonStream<{ type: string; message?: string }>(
        ndjsonResponse('{"type":"status","message":"Working"}\n'),
        (event) => events.push(event)
      )
    ).rejects.toThrow("completion event");
    expect(events[events.length - 1]?.type).toBe("error");
  });

  it("rejects malformed non-empty records", async () => {
    const events: Array<{ type: string; message?: string }> = [];

    await expect(
      consumeNdjsonStream<{ type: string; message?: string }>(
        ndjsonResponse('not-json\n{"type":"done"}\n'),
        (event) => events.push(event)
      )
    ).rejects.toThrow("Malformed clinical stream record");
    expect(events[events.length - 1]?.message).toContain("Malformed clinical stream record");
  });

  it("accepts a well-formed stream ending in done", async () => {
    const events: Array<{ type: string }> = [];

    await consumeNdjsonStream<{ type: string }>(
      ndjsonResponse('{"type":"status","message":"Working"}\n{"type":"done"}\n'),
      (event) => events.push(event)
    );

    expect(events.map((event) => event.type)).toEqual(["status", "done"]);
  });
});
