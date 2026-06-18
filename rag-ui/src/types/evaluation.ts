/** Pydantic-aligned types for the RAG Evaluation API. */

export interface MetricsSummary {
  /** hit@1 — Hit rate at rank 1 */
  "hit@1": number;
  /** hit@3 — Hit rate at rank 3 */
  "hit@3": number;
  /** hit@5 — Hit rate at rank 5 */
  "hit@5": number;
  /** MRR — Mean Reciprocal Rank */
  mrr: number;
  /** precision@1 */
  "precision@1"?: number;
  /** precision@3 */
  "precision@3"?: number;
  /** precision@5 */
  "precision@5"?: number;
  /** recall@1 */
  "recall@1"?: number;
  /** recall@3 */
  "recall@3"?: number;
  /** recall@5 */
  "recall@5"?: number;
  /** ndcg@1 */
  "ndcg@1"?: number;
  /** ndcg@3 */
  "ndcg@3"?: number;
  /** ndcg@5 */
  "ndcg@5"?: number;
}

export interface ConfigSnapshot {
  chunker: string;
  embedding_model: string;
  retrieval_strategy: string;
  use_cross_encoder: boolean;
  cross_encoder_model?: string;
  use_sparse: boolean;
  use_parent_expansion: boolean;
  top_k: number;
  vector_k: number;
  sparse_k: number;
  rrf_constant: number;
  chunk_size: number;
  chunk_overlap: number;
  num_chunks: number;
}

export interface RunSummary {
  id: number;
  config: ConfigSnapshot;
  timestamp: string;
  elapsed_seconds: number;
  total_queries: number;
  metrics: MetricsSummary;
  created_at?: string;
}

export interface QueryResultItem {
  id: number;
  run_id: number;
  query: string;
  retrieved_ids: string[];
  relevant_ids: string[];
  category?: string;
  difficulty?: string;
  query_type?: string;
  metrics?: MetricsSummary;
}

export interface RunDetail extends RunSummary {
  query_results: QueryResultItem[];
}

export interface VersionInfo {
  version: Record<string, unknown>;
  latest_run_id: number;
  run_count: number;
  metrics: MetricsSummary;
  timestamp: string;
  elapsed_seconds: number;
  total_queries: number;
}

export interface CompareResponse {
  runs: RunSummary[];
}

export interface DeleteResponse {
  deleted: number;
  message: string;
}

export interface ErrorResponse {
  detail: string;
}

/** Human-readable label for a version. */
export function versionLabel(v: VersionInfo): string {
  const vv = v.version as Record<string, string | number | boolean>;
  const parts: string[] = [];
  if (vv.embedding_model) parts.push(String(vv.embedding_model));
  if (vv.chunker) parts.push(String(vv.chunker));
  if (vv.retrieval_strategy) parts.push(String(vv.retrieval_strategy));
  if (vv.use_cross_encoder) parts.push("CE");
  if (vv.use_sparse) parts.push("sparse");
  if (vv.top_k) parts.push(`k=${vv.top_k}`);
  return parts.join(" | ") || `Run #${v.latest_run_id}`;
}

/** All K-values available in metrics. */
export const METRIC_K_VALUES = [1, 3, 5] as const;

/** Metric categories for the chart tabs. */
export type MetricCategory = "hit" | "mrr" | "precision" | "recall" | "ndcg";

export const METRIC_CATEGORIES: { key: MetricCategory; label: string }[] = [
  { key: "hit", label: "Hit@K" },
  { key: "mrr", label: "MRR" },
  { key: "precision", label: "Precision@K" },
  { key: "recall", label: "Recall@K" },
  { key: "ndcg", label: "NDCG@K" },
];

/** Get the metric value from MetricsSummary by category and K. */
export function getMetricValue(m: MetricsSummary, cat: MetricCategory, k?: number): number {
  switch (cat) {
    case "hit":
      return m[`hit@${k}` as keyof MetricsSummary] as number;
    case "mrr":
      return m.mrr;
    case "precision":
      return (m[`precision@${k}` as keyof MetricsSummary] as number) ?? 0;
    case "recall":
      return (m[`recall@${k}` as keyof MetricsSummary] as number) ?? 0;
    case "ndcg":
      return (m[`ndcg@${k}` as keyof MetricsSummary] as number) ?? 0;
  }
}
