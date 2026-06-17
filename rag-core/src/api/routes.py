"""FastAPI routes for RAG evaluation results."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from evaluation.db import get_session
from evaluation.db.store import EvaluationStore
from api.models import (
    CompareRequest,
    CompareResponse,
    DeleteResponse,
    ErrorResponse,
    MetricsSummary,
    RunDetail,
    RunSummary,
    VersionInfo,
    _orm_to_metrics,
    run_to_detail,
    run_to_summary,
)

router = APIRouter(prefix="/api", tags=["evaluation"])


# ── Runs ────────────────────────────────────────────────────────

@router.get(
    "/runs",
    response_model=list[RunSummary],
    summary="List evaluation runs",
)
async def list_runs(
    chunker: Optional[str] = Query(None),
    embedding_model: Optional[str] = Query(None),
    retrieval_strategy: Optional[str] = Query(None),
    use_cross_encoder: Optional[bool] = Query(None),
    use_sparse: Optional[bool] = Query(None),
    use_parent_expansion: Optional[bool] = Query(None),
    top_k: Optional[int] = Query(None),
    vector_k: Optional[int] = Query(None),
    sparse_k: Optional[int] = Query(None),
    chunk_size: Optional[int] = Query(None),
    chunk_overlap: Optional[int] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    """List evaluation runs, with optional filtering by configuration fields."""
    filters = {
        k: v for k, v in {
            "chunker": chunker,
            "embedding_model": embedding_model,
            "retrieval_strategy": retrieval_strategy,
            "use_cross_encoder": int(use_cross_encoder) if use_cross_encoder is not None else None,
            "use_sparse": int(use_sparse) if use_sparse is not None else None,
            "use_parent_expansion": int(use_parent_expansion) if use_parent_expansion is not None else None,
            "top_k": top_k,
            "vector_k": vector_k,
            "sparse_k": sparse_k,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }.items() if v is not None
    }
    runs = await EvaluationStore.get_runs(session, filters=filters, limit=limit, offset=offset)
    return [run_to_summary(r) for r in runs]


@router.get(
    "/runs/{run_id}",
    response_model=RunDetail,
    responses={404: {"model": ErrorResponse}},
    summary="Get a single run with per-query details",
)
async def get_run(
    run_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Get a single evaluation run, including per-query results."""
    run = await EvaluationStore.get_run(session, run_id, include_queries=True)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return run_to_detail(run)


@router.get(
    "/runs/{run_id}/metrics",
    response_model=MetricsSummary,
    responses={404: {"model": ErrorResponse}},
    summary="Get aggregate metrics for a run",
)
async def get_run_metrics(
    run_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Get only the aggregate metrics for a single run."""
    run = await EvaluationStore.get_run(session, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return _orm_to_metrics(run)


@router.post(
    "/runs/compare",
    response_model=CompareResponse,
    summary="Compare multiple runs",
)
async def compare_runs(
    body: CompareRequest,
    session: AsyncSession = Depends(get_session),
):
    """Compare aggregate metrics across multiple runs by ID."""
    runs = await EvaluationStore.compare_runs(session, body.run_ids)
    return CompareResponse(runs=[run_to_summary(r) for r in runs])


@router.delete(
    "/runs/{run_id}",
    response_model=DeleteResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Delete a run",
)
async def delete_run(
    run_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Delete an evaluation run and its per-query results."""
    deleted = await EvaluationStore.delete_run(session, run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return DeleteResponse(deleted=run_id)


# ── Versions ────────────────────────────────────────────────────

@router.get(
    "/versions",
    response_model=list[VersionInfo],
    summary="List all distinct config versions",
)
async def list_versions(
    session: AsyncSession = Depends(get_session),
):
    """Return all distinct configuration combinations, each with its latest metrics."""
    versions = await EvaluationStore.get_versions(session)
    return [
        VersionInfo(
            version=v["version"],
            latest_run_id=v["latest_run_id"],
            run_count=v["run_count"],
            metrics=MetricsSummary.model_validate(v["metrics"]),
            timestamp=v["timestamp"],
            elapsed_seconds=v["elapsed_seconds"],
            total_queries=v["total_queries"],
        )
        for v in versions
    ]


@router.get(
    "/versions/latest",
    response_model=list[RunSummary],
    summary="Get latest run per version",
)
async def get_latest_by_version(
    session: AsyncSession = Depends(get_session),
):
    """Return the most recent evaluation run for each configuration combination."""
    runs = await EvaluationStore.get_latest_by_version(session)
    return [run_to_summary(r) for r in runs]
