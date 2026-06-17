import * as runtime from '@/paraglide/runtime';
import type { Locale } from '@/paraglide/runtime';
import * as messages from "@/paraglide/messages";

import { useEffect, useState } from "react"


const listeners = new Set<(locale: Locale) => void>();

export function getLocale() {
  return runtime.getLocale();
}

export function setLocale(locale: Locale) {
  runtime.setLocale(locale);
  document.documentElement.lang = locale;
  listeners.forEach(listener => listener(locale));
}

export function subscribe(listener: (locale: Locale) => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useLocale(): Locale {
  const [locale, setLocaleState] = useState<Locale>(getLocale);
  useEffect(() => {
    const unsubscribe = subscribe(setLocaleState);
    return unsubscribe;
  }, []);
  return locale;
}

export const m = messages.m;