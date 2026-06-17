"""Async SQLAlchemy engine and session management for evaluation DB."""

from __future__ import annotations

from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

# Default DB location
DEFAULT_DB_PATH = Path("src/evaluation/results/evaluations.db")

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_async_engine(db_path: str | Path | None = None) -> AsyncEngine:
    """Return the module-level async engine, creating it lazily."""
    global _engine
    if _engine is not None:
        return _engine
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite+aiosqlite:///{path}"
    _engine = create_async_engine(url, echo=False)
    return _engine


def get_session_factory(
    engine: AsyncEngine | None = None,
) -> async_sessionmaker[AsyncSession]:
    """Return the module-level session factory."""
    global _session_factory
    if _session_factory is not None:
        return _session_factory
    eng = engine or get_async_engine()
    _session_factory = async_sessionmaker(eng, expire_on_commit=False)
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yields an AsyncSession and closes it after use."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        finally:
            await session.close()
