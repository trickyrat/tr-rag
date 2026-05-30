import logging
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


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
        self.setup_retrievers()

    def setup_retrievers(self):
        """Setup the retrievers for both vector search and BM25 search"""
        logger.info("Setting up retrievers...")

        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

        self.bm25_retriever = BM25Retriever.from_documents(self.chunks, k=5)

        logger.info("Retrievers setup completed")

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        Hybrid search - combine vector search and BM25 search, use RRF for reranking

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of retrieved documents
        """

        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)
        return reranked_docs[:top_k]

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

        docs = self.hybrid_search(query, top_k * 3)

        filtered_docs = []
        for doc in docs:
            match = True
            for key, value in filters.items():
                if key in doc.metadata:
                    if isinstance(value, list):
                        if doc.metadata[key] not in value:
                            match = False
                            break
                    else:
                        if doc.metadata[key] != value:
                            match = False
                            break
                else:
                    match = False
                    break

            if match:
                filtered_docs.append(doc)
                if len(filtered_docs) >= top_k:
                    break

        return filtered_docs

    def _rrf_rerank(
        self, vector_docs: List[Document], bm25_docs: List[Document], k: int = 60
    ) -> List[Document]:
        """
        Re-rank documents using RRF (Reciprocal Rank Fusion) algorithm

        Args:
            vector_docs: Vector search results
            bm25_docs: BM25 search results
            k: RRF parameter, used for smoothing rankings

        Returns:
            Re-ranked list of documents
        """
        doc_scores = {}
        doc_objects = {}

        for rank, doc in enumerate(vector_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(
                f"Vector search - Document {rank + 1}: RRF score = {rrf_score:.4f}"
            )

        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(
                f"BM25 search - Document {rank + 1}: RRF score = {rrf_score:.4f}"
            )

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                doc.metadata["rrf_score"] = final_score
                reranked_docs.append(doc)
                logger.debug(
                    f"Final ranking - Document: {doc.page_content[:50]}... Final RRF score: {final_score:.4f}"
                )

        logger.info(
            f"RRF re-ranking completed: Vector search {len(vector_docs)} documents, BM25 search {len(bm25_docs)} documents, merged {len(reranked_docs)} documents"
        )

        return reranked_docs
