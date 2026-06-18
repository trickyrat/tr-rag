import { useEffect, useState } from "react";

/**
 * React hook that returns whether dark mode is active.
 * Watches the `dark` class on `document.documentElement`.
 */
export function useIsDark(): boolean {
  const [isDark, setIsDark] = useState(() =>
    typeof document !== "undefined"
      ? document.documentElement.classList.contains("dark")
      : false
  );

  useEffect(() => {
    const el = document.documentElement;
    const observer = new MutationObserver(() => {
      setIsDark(el.classList.contains("dark"));
    });
    observer.observe(el, { attributes: true, attributeFilter: ["class"] });
    setIsDark(el.classList.contains("dark"));
    return () => observer.disconnect();
  }, []);

  return isDark;
}
