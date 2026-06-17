"""Pydantic models for the RAG evaluation REST API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Request models ──────────────────────────────────────────────

class RunFilter(BaseModel):
    """Optional filters for listing runs."""
    chunker: Optional[str] = None
    embedding_model: Optional[str] = None
    retrieval_strategy: Optional[str] = None
    use_cross_encoder: Optional[bool] = None
    use_sparse: Optional[bool] = None
    use_parent_expansion: Optional[bool] = None
    top_k: Optional[int] = None
    vector_k: Optional[int] = None
    sparse_k: Optional[int] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class CompareRequest(BaseModel):
    run_ids: List[int] = Field(..., min_length=1, max_length=50)


# ── Response models ─────────────────────────────────────────────

class ConfigSnapshot(BaseModel):
    """Snapshot of run configuration."""
    chunker: str
    embedding_model: str
    retrieval_strategy: str
    use_cross_encoder: bool
    cross_encoder_model: Optional[str] = None
    use_sparse: bool
    use_parent_expansion: bool
    top_k: int
    vector_k: int
    sparse_k: int
    rrf_constant: int
    chunk_size: int
    chunk_overlap: int
    num_chunks: int


class MetricsSummary(BaseModel):
    """Aggregate evaluation metrics."""
    hit_at_1: float = Field(alias="hit@1")
    hit_at_3: float = Field(alias="hit@3")
    hit_at_5: float = Field(alias="hit@5")
    mrr: float
    precision_at_1: Optional[float] = Field(default=None, alias="precision@1")
    precision_at_3: Optional[float] = Field(default=None, alias="precision@3")
    precision_at_5: Optional[float] = Field(default=None, alias="precision@5")
    recall_at_1: Optional[float] = Field(default=None, alias="recall@1")
    recall_at_3: Optional[float] = Field(default=None, alias="recall@3")
    recall_at_5: Optional[float] = Field(default=None, alias="recall@5")
    ndcg_at_1: Optional[float] = Field(default=None, alias="ndcg@1")
    ndcg_at_3: Optional[float] = Field(default=None, alias="ndcg@3")
    ndcg_at_5: Optional[float] = Field(default=None, alias="ndcg@5")

    model_config = {"populate_by_name": True}


class RunSummary(BaseModel):
    """A single run in list views (no per-query details)."""
    id: int
    config: ConfigSnapshot
    timestamp: str
    elapsed_seconds: float
    total_queries: int
    metrics: MetricsSummary
    created_at: Optional[str] = None


class QueryResultItem(BaseModel):
    """Per-query evaluation detail."""
    id: int
    run_id: int
    query: str
    retrieved_ids: List[str]
    relevant_ids: List[str]
    category: Optional[str] = None
    difficulty: Optional[str] = None
    query_type: Optional[str] = None
    metrics: Optional[MetricsSummary] = None


class RunDetail(RunSummary):
    """A single run with per-query details."""
    query_results: List[QueryResultItem] = []


class VersionInfo(BaseModel):
    """A distinct configuration combination (version)."""
    version: Dict[str, Any]
    latest_run_id: int
    run_count: int
    metrics: MetricsSummary
    timestamp: str
    elapsed_seconds: float
    total_queries: int


class CompareResponse(BaseModel):
    runs: List[RunSummary]


class DeleteResponse(BaseModel):
    deleted: int
    message: str = "Run deleted"


class ErrorResponse(BaseModel):
    detail: str


# ── Helper: build response from ORM objects ─────────────────────

def _orm_to_config(run) -> ConfigSnapshot:
    return ConfigSnapshot(
        chunker=run.chunker,
        embedding_model=run.embedding_model,
        retrieval_strategy=run.retrieval_strategy,
        use_cross_encoder=run.use_cross_encoder,
        cross_encoder_model=run.cross_encoder_model,
        use_sparse=run.use_sparse,
        use_parent_expansion=run.use_parent_expansion,
        top_k=run.top_k,
        vector_k=run.vector_k,
        sparse_k=run.sparse_k,
        rrf_constant=run.rrf_constant,
        chunk_size=run.chunk_size,
        chunk_overlap=run.chunk_overlap,
        num_chunks=run.num_chunks,
    )


def _orm_to_metrics(run) -> MetricsSummary:
    return MetricsSummary.model_validate(
        {
            "hit_at_1": run.hit_at_1,
            "hit_at_3": run.hit_at_3,
            "hit_at_5": run.hit_at_5,
            "mrr": run.mrr,
            "precision_at_1": run.precision_at_1,
            "precision_at_3": run.precision_at_3,
            "precision_at_5": run.precision_at_5,
            "recall_at_1": run.recall_at_1,
            "recall_at_3": run.recall_at_3,
            "recall_at_5": run.recall_at_5,
            "ndcg_at_1": run.ndcg_at_1,
            "ndcg_at_3": run.ndcg_at_3,
            "ndcg_at_5": run.ndcg_at_5,
        }
    )


def run_to_summary(run) -> RunSummary:
    return RunSummary(
        id=run.id,
        config=_orm_to_config(run),
        timestamp=run.timestamp,
        elapsed_seconds=run.elapsed_seconds,
        total_queries=run.total_queries,
        metrics=_orm_to_metrics(run),
        created_at=run.created_at,
    )


def run_to_detail(run) -> RunDetail:
    import json

    qr_items: List[QueryResultItem] = []
    for qr in getattr(run, "query_results", []) or []:
        ret_ids = json.loads(qr.retrieved_ids) if qr.retrieved_ids else []
        rel_ids = json.loads(qr.relevant_ids) if qr.relevant_ids else []
        qr_items.append(
            QueryResultItem(
                id=qr.id,
                run_id=qr.run_id,
                query=qr.query,
                retrieved_ids=ret_ids,
                relevant_ids=rel_ids,
                category=qr.category,
                difficulty=qr.difficulty,
                query_type=qr.query_type,
                metrics=MetricsSummary.model_validate(
                    {
                        "hit_at_1": qr.hit_at_1 or 0,
                        "hit_at_3": qr.hit_at_3 or 0,
                        "hit_at_5": qr.hit_at_5 or 0,
                        "mrr": qr.mrr or 0,
                        "precision_at_1": qr.precision_at_1,
                        "precision_at_3": qr.precision_at_3,
                        "precision_at_5": qr.precision_at_5,
                        "recall_at_1": qr.recall_at_1,
                        "recall_at_3": qr.recall_at_3,
                        "recall_at_5": qr.recall_at_5,
                        "ndcg_at_1": qr.ndcg_at_1,
                        "ndcg_at_3": qr.ndcg_at_3,
                        "ndcg_at_5": qr.ndcg_at_5,
                    }
                ),
            )
        )

    return RunDetail(
        id=run.id,
        config=_orm_to_config(run),
        timestamp=run.timestamp,
        elapsed_seconds=run.elapsed_seconds,
        total_queries=run.total_queries,
        metrics=_orm_to_metrics(run),
        created_at=run.created_at,
        query_results=qr_items,
    )
