/**
 * Typed API client for the RAG Evaluation API.
 * Default base URL: http://localhost:8000
 *
 * Override via VITE_API_BASE_URL env var or by passing a custom baseUrl.
 */

import type {
  CompareResponse,
  ErrorResponse,
  MetricsSummary,
  RunDetail,
  RunSummary,
  VersionInfo,
} from "@/types/evaluation";

const DEFAULT_BASE = "http://localhost:8000";

let _baseUrl = import.meta.env.VITE_API_BASE_URL as string | undefined ?? DEFAULT_BASE;

export function setBaseUrl(url: string) {
  _baseUrl = url;
}

export function getBaseUrl() {
  return _baseUrl;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${_baseUrl}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as ErrorResponse;
    throw new Error(body.detail ?? `HTTP ${res.status}: ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

/** List all distinct config versions with latest metrics. */
export async function getVersions(): Promise<VersionInfo[]> {
  return request<VersionInfo[]>("/api/versions");
}

/** Get latest run per version. */
export async function getLatestByVersion(): Promise<RunSummary[]> {
  return request<RunSummary[]>("/api/versions/latest");
}

/** List evaluation runs with optional filters. */
export async function getRuns(params?: Record<string, string | number | boolean>): Promise<RunSummary[]> {
  const qs = params
    ? "?" + new URLSearchParams(
        Object.entries(params)
          .filter(([, v]) => v !== undefined && v !== null)
          .map(([k, v]) => [k, String(v)])
      ).toString()
    : "";
  return request<RunSummary[]>(`/api/runs${qs}`);
}

/** Get a single run with per-query details. */
export async function getRunDetail(runId: number): Promise<RunDetail> {
  return request<RunDetail>(`/api/runs/${runId}`);
}

/** Get aggregate metrics for a single run. */
export async function getRunMetrics(runId: number): Promise<MetricsSummary> {
  return request<MetricsSummary>(`/api/runs/${runId}/metrics`);
}

/** Compare multiple runs by ID. */
export async function compareRuns(runIds: number[]): Promise<CompareResponse> {
  return request<CompareResponse>("/api/runs/compare", {
    method: "POST",
    body: JSON.stringify({ run_ids: runIds }),
  });
}

/** Delete a run. */
export async function deleteRun(runId: number): Promise<{ deleted: number; message: string }> {
  return request(`/api/runs/${runId}`, { method: "DELETE" });
}
