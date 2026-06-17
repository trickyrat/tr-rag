"""SQLAlchemy ORM models for RAG evaluation results."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import ForeignKey, Index, Integer, String, Float, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Run(Base):
    """One row per evaluation run — config snapshot + aggregate metrics."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Configuration snapshot ──────────────────────────────────
    chunker: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(256), nullable=False)
    retrieval_strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    use_cross_encoder: Mapped[bool] = mapped_column(Integer, nullable=False, default=0)
    cross_encoder_model: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    use_sparse: Mapped[bool] = mapped_column(Integer, nullable=False, default=0)
    use_parent_expansion: Mapped[bool] = mapped_column(Integer, nullable=False, default=0)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    vector_k: Mapped[int] = mapped_column(Integer, nullable=False)
    sparse_k: Mapped[int] = mapped_column(Integer, nullable=False)
    rrf_constant: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_overlap: Mapped[int] = mapped_column(Integer, nullable=False)
    num_chunks: Mapped[int] = mapped_column(Integer, nullable=False)

    # ── Run metadata ────────────────────────────────────────────
    timestamp: Mapped[str] = mapped_column(String(64), nullable=False)
    elapsed_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    total_queries: Mapped[int] = mapped_column(Integer, nullable=False)

    # ── Aggregate metrics ───────────────────────────────────────
    mrr: Mapped[float] = mapped_column(Float, nullable=False)
    hit_at_1: Mapped[float] = mapped_column(Float, nullable=False)
    hit_at_3: Mapped[float] = mapped_column(Float, nullable=False)
    hit_at_5: Mapped[float] = mapped_column(Float, nullable=False)
    precision_at_1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precision_at_3: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precision_at_5: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recall_at_1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recall_at_3: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recall_at_5: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ndcg_at_1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ndcg_at_3: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ndcg_at_5: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[str] = mapped_column(
        String(64), nullable=False, server_default=func.now()
    )

    # ── Relationships ───────────────────────────────────────────
    query_results: Mapped[list["QueryResult"]] = relationship(
        "QueryResult", back_populates="run", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_runs_embedding", "embedding_model"),
        Index("ix_runs_strategy", "retrieval_strategy"),
        Index("ix_runs_timestamp", "timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"<Run(id={self.id}, {self.embedding_model}/{self.retrieval_strategy}, "
            f"top_k={self.top_k}, hit@1={self.hit_at_1:.4f})>"
        )


class QueryResult(Base):
    """Per-query evaluation details for a run."""

    __tablename__ = "query_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )

    query: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_ids: Mapped[str] = mapped_column(Text, nullable=False)  # JSON array
    relevant_ids: Mapped[str] = mapped_column(Text, nullable=False)   # JSON array

    category: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    difficulty: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    query_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # ── Per-query metrics ───────────────────────────────────────
    mrr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hit_at_1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hit_at_3: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hit_at_5: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precision_at_1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precision_at_3: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precision_at_5: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recall_at_1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recall_at_3: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recall_at_5: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ndcg_at_1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ndcg_at_3: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ndcg_at_5: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ── Relationships ───────────────────────────────────────────
    run: Mapped["Run"] = relationship("Run", back_populates="query_results")

    __table_args__ = (
        Index("ix_query_results_run_id", "run_id"),
    )

    def __repr__(self) -> str:
        return f"<QueryResult(id={self.id}, run_id={self.run_id}, query={self.query[:40]}...)>"
