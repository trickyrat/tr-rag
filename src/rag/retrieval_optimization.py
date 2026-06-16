from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field, PrivateAttr
import logging
import re
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Tokenize text for BM25: lowercase + split on non-alphanumeric for English,
    and character-level for CJK characters."""
    # Normalize
    text = text.lower().strip()
    # Split on whitespace and punctuation, keeping CJK chars as individual tokens
    tokens = re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text)
    return tokens if tokens else text.split()


class HybridRetriever(BaseRetriever):
    vectorstore: Chroma
    chunks: List[Document] = Field(default_factory=list)
    vector_k: int = 5
    bm25_k: int = 5
    top_k: int = 3
    rrf_constant: int = 60

    _bm25_index: Optional[BM25Okapi] = PrivateAttr(default=None)
    _tokenized_corpus: Optional[List[List[str]]] = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed = True

    def _get_bm25_index(self) -> BM25Okapi:
        if self._bm25_index is None:
            logger.info("Building BM25 index (first use)...")
            self._tokenized_corpus = [_tokenize(doc.page_content) for doc in self.chunks]
            self._bm25_index = BM25Okapi(self._tokenized_corpus)
            logger.info("BM25 index built successfully")
        return self._bm25_index

    def _bm25_search(self, query: str) -> List[Document]:
        bm25 = self._get_bm25_index()
        tokenized_query = _tokenize(query)
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            : self.bm25_k
        ]
        return [self.chunks[i] for i in top_indices if scores[i] > 0]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        vector_docs = self.vectorstore.similarity_search(query, k=self.vector_k)
        bm25_docs = self._bm25_search(query)
        return self._rrf_rerank(vector_docs, bm25_docs)[: self.top_k]

    def _rrf_rerank(
        self, vector_docs: List[Document], bm25_docs: List[Document]
    ) -> List[Document]:
        k = self.rrf_constant
        doc_scores: Dict[str, float] = {}
        doc_objects: Dict[str, Document] = {}

        for rank, doc in enumerate(vector_docs):
            doc_id = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
            doc_objects[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        for rank, doc in enumerate(bm25_docs):
            doc_id = doc.metadata.get("chunk_id", str(hash(doc.page_content)))
            doc_objects[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        # Sort documents by their combined scores
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        reranked = []
        for doc_id, score in sorted_docs:
            doc = doc_objects[doc_id]
            doc.metadata["rrf_score"] = score
            reranked.append(doc)

        logger.info(
            f"RRF reranking: {len(vector_docs)} vector + {len(bm25_docs)} BM25 -> {len(reranked)} merged"
        )
        return reranked


class RetrievalOptimization:
    """Retrieval Optimization"""

    def __init__(self, vectorstore: Chroma, chunks: List[Document]):
        """
        Initialize the RetrievalOptimization class.

        Args:
            vectorstore: Chroma vector store
            chunks: List of documents
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.hybrid_retriever = HybridRetriever(
            vectorstore=self.vectorstore, chunks=self.chunks
        )

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
