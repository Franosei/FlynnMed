export type ThemePreference = "system" | "light" | "dark";
export type ResolvedTheme = "light" | "dark";

export const THEME_STORAGE_KEY = "flynnmed-colour-theme";

export function isThemePreference(value: unknown): value is ThemePreference {
  return value === "system" || value === "light" || value === "dark";
}

export function loadThemePreference(): ThemePreference {
  if (typeof window === "undefined") return "system";
  try {
    const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
    return isThemePreference(stored) ? stored : "system";
  } catch {
    return "system";
  }
}

export function saveThemePreference(preference: ThemePreference): void {
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, preference);
  } catch {
    // The theme still applies for this session when storage is unavailable.
  }
}

export function resolveTheme(
  preference: ThemePreference,
  systemPrefersDark: boolean
): ResolvedTheme {
  if (preference === "system") return systemPrefersDark ? "dark" : "light";
  return preference;
}

export function applyTheme(
  preference: ThemePreference,
  systemPrefersDark: boolean,
  root: HTMLElement = document.documentElement
): ResolvedTheme {
  const resolved = resolveTheme(preference, systemPrefersDark);
  root.dataset.theme = resolved;
  root.style.colorScheme = resolved;
  return resolved;
}

export function initialiseTheme(): ThemePreference {
  const preference = loadThemePreference();
  const systemPrefersDark =
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-color-scheme: dark)").matches;
  applyTheme(preference, systemPrefersDark);
  return preference;
}
