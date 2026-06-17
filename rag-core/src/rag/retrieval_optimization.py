from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field
import logging
from typing import List, Dict, Any, Optional, Callable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    vectorstore: Chroma
    chunks: List[Document] = Field(default_factory=list)
    vector_k: int = 10
    sparse_k: int = 10
    top_k: int = 3
    rrf_constant: int = 60
    use_sparse: bool = True
    # bge-m3 sparse backend (set externally after index_documents)
    sparse_search_fn: Optional[Callable] = None

    class Config:
        arbitrary_types_allowed = True

    def _sparse_search(self, query: str) -> List[Document]:
        """Delegate to bge-m3 sparse (lexical) embedding search."""
        if self.sparse_search_fn is None:
            return []
        results = self.sparse_search_fn(query, top_k=self.sparse_k)
        return [doc for doc, _ in results]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        vector_results = self.vectorstore.similarity_search_with_score(
            query, k=self.vector_k
        )
        vector_docs = [doc for doc, _ in vector_results]
        vector_scores = {
            doc.metadata.get("chunk_id", ""): score
            for doc, score in vector_results
        }

        sparse_docs = self._sparse_search(query) if self.use_sparse else []
        return self._rrf_rerank(vector_docs, sparse_docs, vector_scores)[: self.top_k]

    def _rrf_rerank(
        self,
        vector_docs: List[Document],
        sparse_docs: List[Document],
        vector_scores: Optional[Dict[str, float]] = None,
    ) -> List[Document]:
        """Reciprocal Rank Fusion (RRF).

        Uses rank-based scoring: score(d) = Σ 1/(k + rank_i(d)).
        ChromaDB returns L2/cosine distances (lower=better), so vector rank 0
        is the most similar document.  We always use rank-based RRF rather
        than raw distances to avoid score-direction inversion.
        """
        k = self.rrf_constant
        doc_scores: Dict[str, float] = {}
        doc_objects: Dict[str, Document] = {}

        # Vector search: always use rank-based RRF score
        for rank, doc in enumerate(vector_docs):
            doc_id = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
            if doc_id not in doc_objects:
                doc_objects[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        # Sparse (lexical) search: rank-based RRF score
        for rank, doc in enumerate(sparse_docs):
            doc_id = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
            if doc_id not in doc_objects:
                doc_objects[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        reranked = []
        for doc_id, score in sorted_docs:
            doc = doc_objects[doc_id]
            doc.metadata["rrf_score"] = score
            reranked.append(doc)

        logger.debug(
            f"RRF: {len(vector_docs)} vector + {len(sparse_docs)} sparse "
            f"→ {len(reranked)} merged"
        )
        return reranked

class RetrievalOptimization:
    """Unified retrieval with optional cross-encoder reranking.

    Usage::

        retriever = RetrievalOptimization(
            vectorstore=vs, chunks=chunks,
            cross_encoder_model="BAAI/bge-reranker-v2-m3",
        )
        docs = retriever.search("your query", top_k=3)
    """

    def __init__(
        self,
        vectorstore: Chroma,
        chunks: List[Document],
        parent_docs: Optional[List[Document]] = None,
        parent_child_map: Optional[Dict[str, str]] = None,
        sparse_search_fn: Optional[Callable] = None,
        use_sparse: bool = True,
        vector_k: int = 10,
        sparse_k: int = 10,
        top_k: int = 3,
        rrf_constant: int = 60,
        cross_encoder_model: Optional[str] = None,
        cross_encoder_candidate_k: int = 30,
        cross_encoder_device: Optional[str] = None,
    ):
        """
        Args:
            vectorstore: Chroma vector store.
            chunks: All document chunks (needed for sparse search).
            parent_docs: Full parent documents for Small-to-Big expansion.
            parent_child_map: child_id → parent_id mapping.
            sparse_search_fn: bge-m3 sparse (lexical) search callable.
            use_sparse: Enable sparse/lexical search in hybrid retrieval.
            vector_k: Vector candidates to retrieve.
            sparse_k: Sparse candidates to retrieve.
            top_k: Final documents returned per query.
            rrf_constant: RRF constant (k in 1/(k+rank)).
            cross_encoder_model: Path/name to cross-encoder model.
                ``None`` disables cross-encoder reranking.
            cross_encoder_candidate_k: How many hybrid candidates to
                feed into the cross-encoder.
            cross_encoder_device: "cpu" or "cuda". Auto-detected if None.
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.parent_docs = parent_docs or []
        self.parent_child_map = parent_child_map or {}
        self.top_k = top_k
        self.cross_encoder_candidate_k = cross_encoder_candidate_k
        self._cross_encoder: Optional[CrossEncoderReranker] = None

        self.hybrid_retriever = HybridRetriever(
            vectorstore=self.vectorstore,
            chunks=self.chunks,
            vector_k=vector_k,
            sparse_k=sparse_k,
            top_k=top_k,
            rrf_constant=rrf_constant,
            sparse_search_fn=sparse_search_fn,
            use_sparse=use_sparse,
        )

        if cross_encoder_model:
            import torch

            device = cross_encoder_device or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._cross_encoder = CrossEncoderReranker(
                model_name=cross_encoder_model,
                device=device,
            )
            if not self._cross_encoder.is_available:
                logger.warning(
                    "Cross-encoder failed to load; reranking disabled. "
                    "Hybrid retrieval will be used as-is."
                )
                self._cross_encoder = None
            else:
                logger.info(
                    f"Cross-encoder ready: {cross_encoder_model} "
                    f"(candidate_k={cross_encoder_candidate_k})"
                )

    # ── unified API ──────────────────────────────────────────────

    def search(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """Full pipeline: hybrid retrieval → cross-encoder reranking.

        If ``use_parent_expansion`` was configured, calls
        ``retrieve_with_parents`` instead of raw ``hybrid_search``.

        Args:
            query: Natural-language query.
            top_k: Documents to return (default: ``self.top_k``).

        Returns:
            Ranked document list.
        """
        k = top_k or self.top_k

        # Step 1: hybrid retrieval (get a larger candidate pool)
        ce_k = self.cross_encoder_candidate_k if self._cross_encoder else k
        docs = self.hybrid_search(query, top_k=ce_k)

        # Step 2: cross-encoder rerank (if available)
        if self._cross_encoder:
            docs = self._cross_encoder.rerank(query, docs, top_k=k)

        return docs[:k]

    def search_with_parents(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Document]:
        """Small-to-Big pipeline with optional cross-encoder reranking.

        Args:
            query: Natural-language query.
            top_k: Documents to return (default: ``self.top_k``).

        Returns:
            Ranked parent-document list.
        """
        k = top_k or self.top_k
        ce_k = self.cross_encoder_candidate_k if self._cross_encoder else k

        docs = self.retrieve_with_parents(query, top_k=ce_k)

        if self._cross_encoder:
            docs = self._cross_encoder.rerank(query, docs, top_k=k)

        return docs[:k]

    def rerank(
        self, query: str, documents: List[Document], top_k: Optional[int] = None
    ) -> List[Document]:
        """Explicit cross-encoder rerank (no-op if cross-encoder disabled)."""
        if self._cross_encoder:
            return self._cross_encoder.rerank(
                query, documents, top_k=top_k or self.top_k
            )
        return documents[: (top_k or self.top_k)]

    @property
    def has_cross_encoder(self) -> bool:
        return self._cross_encoder is not None

    # ── legacy methods (kept for backward compatibility) ──────────

    def retrieve_with_parents(self, query: str, top_k: int = 3) -> List[Document]:
        """Small-to-Big: retrieve child chunks, then expand to parent documents.

        Retrieves child chunks via hybrid search, maps each to its parent
        document via parent_child_map, deduplicates, and returns parent docs.
        Falls back to raw hybrid_search if no parent data is available.
        """
        if not self.parent_docs or not self.parent_child_map:
            logger.warning("No parent data, falling back to hybrid_search")
            return self.hybrid_search(query, top_k=top_k)

        # Retrieve more children to ensure sufficient parent coverage
        child_docs = self.hybrid_search(query, top_k=top_k * 3)

        # Map children to unique parents, accumulating all child content_hashes
        parent_hashes: Dict[str, list[str]] = {}
        parent_order: list[str] = []
        for child in child_docs:
            parent_id = child.metadata.get("parent_id")
            child_hash = child.metadata.get("content_hash", "")
            if parent_id:
                if parent_id not in parent_hashes:
                    parent_hashes[parent_id] = []
                    parent_order.append(parent_id)
                if child_hash:
                    parent_hashes[parent_id].append(child_hash)

        expanded: List[Document] = []
        for parent_id in parent_order:
            parent = self._find_parent(parent_id)
            if parent:
                parent.metadata = {
                    **parent.metadata,
                    "child_content_hashes": parent_hashes[parent_id],
                }
                expanded.append(parent)

        logger.info(
            f"Small-to-Big: {len(child_docs)} children → "
            f"{len(expanded)} unique parents (top_k={top_k})"
        )
        return expanded[:top_k]

    def _find_parent(self, parent_id: str) -> Optional[Document]:
        """Find parent document by parent_id."""
        for doc in self.parent_docs:
            if doc.metadata.get("parent_id") == parent_id:
                return doc
        return None

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        Hybrid search - combine vector search and BM25 search, use RRF for reranking

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of retrieved documents
        """

        self.hybrid_retriever.top_k = top_k
        return self.hybrid_retriever.invoke(query)

    def metadata_filtered_search(
        self, query: str, filters: Dict[str, Any], top_k: int = 5
    ) -> List[Document]:
        """
        Metadata-filtered search

        Args:
            query: Query text
            filters: Metadata filter conditions
            top_k: Number of results to return

        Returns:
            Filtered list of documents
        """

        try:
            return self.vectorstore.similarity_search(query, k=top_k, filters=filters)
        except Exception as e:
            logger.error(f"Native filter failed ({e}), falling back to post-filtering")
            docs = self.hybrid_search(query, top_k * 3)
            filtered = [
                doc
                for doc in docs
                if all(doc.metadata.get(k) == v for k, v in filters.items())
            ]
            return filtered[:top_k]


class CrossEncoderReranker:
    """Re-rank documents with a cross-encoder (joint query-document scoring).

    Uses ``sentence_transformers.CrossEncoder`` which is more stable
    than ``FlagEmbedding.FlagReranker`` across ``transformers`` versions.

    Default model: ``BAAI/bge-reranker-v2-m3`` (multilingual).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                model_name,
                device=device,
                trust_remote_code=True,
            )
            logger.info(
                "Cross-encoder loaded: %s on %s", model_name, device
            )
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; "
                "cross-encoder disabled.  Install: pip install sentence-transformers"
            )
        except Exception as e:
            logger.warning("Failed to load cross-encoder %s: %s", model_name, e)

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3,
    ) -> List[Document]:
        """Re-rank *documents* by cross-encoder score (descending).

        Each returned ``Document`` receives ``metadata["ce_score"]``.
        """
        if not self._model or not documents:
            return documents[:top_k]

        pairs = [[query, doc.page_content] for doc in documents]
        try:
            scores = self._model.predict(
                pairs,
                batch_size=min(32, len(pairs)),
                show_progress_bar=False,
                convert_to_tensor=False,
            )
        except Exception as e:
            logger.warning("Cross-encoder scoring failed: %s", e)
            return documents[:top_k]

        # scores is a numpy array or list; ensure it's iterable
        if not hasattr(scores, "__len__"):
            scores = [float(scores)]
        else:
            scores = [float(s) for s in scores]

        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked: List[Document] = []
        for doc, score in scored[:top_k]:
            doc.metadata["ce_score"] = round(score, 6)
            reranked.append(doc)

        logger.debug(
            "Cross-encoder: %d candidates → top %d", len(documents), len(reranked)
        )
        return reranked
