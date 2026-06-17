"""RAG system configuration — loaded from config.yaml via pydantic-settings.

Priority (lowest to highest):
    1. Field defaults (defined below)
    2. config.yaml file
    3. Environment variables (prefixed with ``RAG_``)

Usage::

    from core.config import RAGConfig
    config = RAGConfig()          # loads config.yaml automatically
    config = RAGConfig(_yaml_file="prod.yaml")  # custom YAML path
"""

from pathlib import Path
from typing import Any, Dict, Tuple, Type

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class RAGConfig(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file=str(Path(__file__).parent / "config.yaml"),
        yaml_file_encoding="utf-8",
        env_prefix="RAG_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # ── paths ────────────────────────────────────────────────────
    knowledge_base_path: str = Field(
        default="./knowledge_base",
        description="Directory containing markdown knowledge base files.",
    )
    index_save_path: str = Field(
        default="./knowledge_base_store",
        description="ChromaDB persistent storage directory.",
    )
    evaluation_testset_path: str = Field(
        default="./test/evaluation/testset.jsonl",
        description="JSONL test set for retrieval evaluation.",
    )

    # ── models ───────────────────────────────────────────────────
    embedding_model: str = Field(
        default="D:/models/BAAI/bge-m3",
        description="Path or HuggingFace ID for the bi-encoder embedding model.",
    )
    llm_model: str = Field(
        default="deepseek-v4-pro",
        description="LLM model name for answer generation.",
    )

    # ── retrieval ────────────────────────────────────────────────
    top_k: int = Field(default=3, ge=1, description="Final documents returned per query.")
    vector_k: int = Field(default=30, ge=1, description="Vector candidates to retrieve.")
    bm25_k: int = Field(default=10, ge=1, description="BM25 / sparse candidates (legacy name).")
    sparse_k: int = Field(default=30, ge=1, description="Sparse (lexical) candidates.")
    rrf_constant: int = Field(default=60, ge=1, description="RRF constant k in 1/(k+rank).")

    # ── chunking ─────────────────────────────────────────────────
    chunk_size: int = Field(default=512, ge=64, description="Target chunk size in characters.")
    chunk_overlap: int = Field(default=64, ge=0, description="Chunk overlap in characters.")
    use_markdown_parser: bool = Field(
        default=False,
        description="True = new MarkdownParser (186 chunks); False = legacy splitter (105 chunks).",
    )
    use_parent_expansion: bool = Field(
        default=False,
        description="Enable Small-to-Big parent-document expansion.",
    )

    # ── hybrid retrieval ─────────────────────────────────────────
    use_sparse: bool = Field(
        default=False,
        description="Enable bge-m3 sparse lexical search in hybrid retrieval.",
    )

    # ── cross-encoder reranking ──────────────────────────────────
    use_cross_encoder: bool = Field(
        default=True,
        description="Enable cross-encoder reranking after hybrid retrieval.",
    )
    cross_encoder_model: str = Field(
        default="D:/models/BAAI/bge-reranker-v2-m3",
        description="Path or HuggingFace ID for the cross-encoder model.",
    )
    cross_encoder_top_k: int = Field(
        default=30, ge=1,
        description="Candidates to feed into the cross-encoder.",
    )

    # ── generation ───────────────────────────────────────────────
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

    # ── backward-compatible helpers ──────────────────────────────

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        """Create config from a dictionary (backward compat)."""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (backward compat)."""
        return self.model_dump()

    @classmethod
    def from_yaml(cls, path: str) -> "RAGConfig":
        """Explicitly load from a YAML file."""
        return cls(_yaml_file=path)


# Singleton default — loads config.yaml automatically
DEFAULT_CONFIG = RAGConfig()
