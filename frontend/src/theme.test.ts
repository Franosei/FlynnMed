import { beforeEach, describe, expect, it } from "vitest";
import {
  applyTheme,
  loadThemePreference,
  resolveTheme,
  saveThemePreference,
  THEME_STORAGE_KEY,
} from "./theme";

describe("colour theme", () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.removeAttribute("data-theme");
    document.documentElement.style.removeProperty("color-scheme");
  });

  it("uses the saved valid preference and rejects unknown values", () => {
    localStorage.setItem(THEME_STORAGE_KEY, "dark");
    expect(loadThemePreference()).toBe("dark");

    localStorage.setItem(THEME_STORAGE_KEY, "sepia");
    expect(loadThemePreference()).toBe("system");
  });

  it("resolves system mode from the operating-system preference", () => {
    expect(resolveTheme("system", true)).toBe("dark");
    expect(resolveTheme("system", false)).toBe("light");
    expect(resolveTheme("light", true)).toBe("light");
  });

  it("applies and persists a selected theme", () => {
    saveThemePreference("dark");
    const resolved = applyTheme("dark", false);

    expect(resolved).toBe("dark");
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe("dark");
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(document.documentElement.style.colorScheme).toBe("dark");
  });
});
