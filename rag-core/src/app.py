"""FastAPI application entry point for RAG evaluation API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from evaluation.db import get_async_engine
from api.routes import router
import uvicorn

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ensure DB engine is initialized."""
    logger.info("Initializing async DB engine...")
    get_async_engine()  # lazy-init the engine
    yield
    logger.info("Shutting down API server")


app = FastAPI(
    title="RAG Evaluation API",
    description="Query and compare RAG evaluation runs stored in SQLite via SQLAlchemy async.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — permissive for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok"}


def main():
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)


# ── CLI entry ───────────────────────────────────────────────────
if __name__ == "__main__":
    main()
