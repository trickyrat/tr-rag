from langchain_core.indexing import index, InMemoryRecordManager, IndexingResult
from langchain_core.embeddings import Embeddings
from FlagEmbedding import BGEM3FlagModel
import chromadb
import torch
import logging
from typing import List, Optional, Literal, Dict, Tuple
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

COLLECTION_NAME = "knowledge_base"


class BGEM3Embeddings(Embeddings):
    """LangChain-compatible embedding wrapper for BGE-M3.

    Provides dense embeddings for Chroma and sparse (lexical) embeddings
    for hybrid keyword-matching search.  Adds query instruction prefix
    for queries (critical for BGE-M3 retrieval quality).
    """

    QUERY_INSTRUCTION = ""  # BGEM3FlagModel handles query/passage internally

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model = BGEM3FlagModel(model_name, use_fp16=(device == "cuda"))
        self.device = device

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        output = self.model.encode(
            texts, return_dense=True, return_sparse=False, batch_size=12
        )
        return output["dense_vecs"].tolist()

    def embed_query(self, text: str) -> List[float]:
        output = self.model.encode(
            [self.QUERY_INSTRUCTION + text],
            return_dense=True,
            return_sparse=False,
        )
        return output["dense_vecs"][0].tolist()

    def encode_sparse(
        self, texts: List[str], is_query: bool = False
    ) -> List[Dict[int, float]]:
        """Return sparse lexical weights for each text."""
        if is_query:
            texts = [self.QUERY_INSTRUCTION + t for t in texts]
        output = self.model.encode(
            texts, return_dense=False, return_sparse=True, batch_size=12
        )
        return output["lexical_weights"]


class VectorStoreBuilder:
    def __init__(
        self,
        model_name: str,
        index_save_path: str,
    ):
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.embeddings: Optional[BGEM3Embeddings] = None
        self.vectorstore: Optional[Chroma] = None
        self._client: Optional[chromadb.ClientAPI] = None
        self._record_manager: Optional[InMemoryRecordManager] = None
        # Sparse embeddings for keyword search (bge-m3 lexical weights)
        self._sparse_embeddings: Optional[List[Dict[int, float]]] = None
        self._chunks: Optional[List[Document]] = None
        self.setup_embeddings()

    def setup_embeddings(self):
        """Setup embeddings with local file cache"""
        logger.info(f"Initializing embedding model: {self.model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings = BGEM3Embeddings(model_name=self.model_name, device=device)
        logger.info(f"Embedding model initialized on {device}")

    def _get_record_manager(self) -> InMemoryRecordManager:
        """Get an InMemoryRecordManager for dedup tracking.

        Note: InMemoryRecordManager state is lost on restart. This means
        re-indexing the same documents after restart will re-embed them.
        For production, consider upgrading langchain-core (>=0.3) which
        includes SQLRecordManager for persistent dedup state.
        """
        if self._record_manager is None:
            self._record_manager = InMemoryRecordManager(
                namespace=COLLECTION_NAME,
            )
            self._record_manager.create_schema()
        return self._record_manager

    def _get_client(self) -> chromadb.ClientAPI:
        """
        Get the ChromaDB client.

        Returns:
            ChromaDB client
        """
        if not self._client:
            Path(self.index_save_path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.index_save_path)
        return self._client

    def _get_vectorstore(self) -> Chroma:
        """
        Get the Chroma vector store.

        Returns:
            Chroma vector store
        """
        if not self.vectorstore:
            client = self._get_client()
            self.vectorstore = Chroma(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,
            )
        return self.vectorstore

    def collection_exists(self) -> bool:
        """
        Check if the collection exists in the vector store.

        Returns:
            True if the collection exists, False otherwise
        """
        try:
            client = self._get_client()
            collection = client.get_collection(name=COLLECTION_NAME)
            return collection.count() > 0
        except Exception:
            return False

    def index_documents(
        self,
        chunks: List[Document],
        cleanup: Literal["incremental", "full", "scoped_full"] = "incremental",
    ) -> IndexingResult:
        """
        Index documents using langchain's index() API with automatic deduplication and change detection.

        Args:
            chunks: List of Document objects to be indexed
            cleanup: Cleanup mode - "incremental", "full", or "scoped_full"

        Returns:
            Indexing result
        """
        if not chunks:
            raise ValueError("Document chunk list cannot be empty")

        logger.info(f"Indexing {len(chunks)} documents with cleanup mode: {cleanup}...")

        # Compute dense + sparse in one pass
        contents = [c.page_content for c in chunks]
        self._sparse_embeddings = self.embeddings.encode_sparse(contents)
        self._chunks = chunks

        vectorstore = self._get_vectorstore()
        record_manager = self._get_record_manager()

        result = index(
            docs_source=chunks,
            record_manager=record_manager,
            vector_store=vectorstore,
            cleanup=cleanup,
            source_id_key="source",
            key_encoder="sha256",
        )

        logger.info(
            f"Indexing complete: +{result['num_added']} ~{result['num_updated']} "
            f"-{result['num_deleted']} ={result['num_skipped']} | "
            f"Sparse dims: {len(self._sparse_embeddings)}"
        )
        return result

    def get_all_documents(self) -> List[Document]:
        """
        Retrieve all documents from the index

        Returns:
            List of all documents in the index
        """
        client = self._get_client()

        try:
            collection = client.get_collection(name=COLLECTION_NAME)
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

        results = collection.get(include=["documents", "metadatas"])
        documents = []
        for doc_text, metadata in zip(results["documents"], results["metadatas"]):
            documents.append(Document(page_content=doc_text, metadata=metadata))

        logger.info(f"Retrieved {len(documents)} documents from the index")
        return documents

    def get_vectorstore(self) -> Chroma:
        """
        Get the vector store object

        Returns:
            Vector store object
        """
        return self._get_vectorstore()

    def reset_collection(self):
        """Wipe the entire vector store (all collections + persisted data)."""
        import shutil

        # Close existing client
        if self._client is not None:
            try:
                del self._client
            except Exception:
                pass
            self._client = None
        self.vectorstore = None
        self._record_manager = None

        # Delete persisted ChromaDB directory
        persist_dir = Path(self.index_save_path)
        if persist_dir.exists():
            shutil.rmtree(persist_dir, ignore_errors=True)
            logger.info(f"Wiped vector store: {persist_dir}")

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Similarity search

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of similar documents
        """
        vectorstore = self._get_vectorstore()

        return vectorstore.similarity_search(query, k=k)

    def sparse_search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Keyword search using bge-m3 sparse (lexical) embeddings.

        Computes dot-product between query sparse vector and each document's
        sparse vector, returns top-K documents with scores.
        """
        if not self._sparse_embeddings or not self._chunks:
            logger.warning("Sparse embeddings not available")
            return []

        query_sparse = self.embeddings.encode_sparse([query], is_query=True)[0]

        # Dot-product scoring
        scored: List[Tuple[int, float]] = []
        for i, doc_sparse in enumerate(self._sparse_embeddings):
            score = sum(
                query_sparse.get(tok, 0.0) * weight
                for tok, weight in doc_sparse.items()
            )
            if score > 0:
                scored.append((i, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = [(self._chunks[i], s) for i, s in scored[:top_k]]
        logger.debug(f"Sparse search: {len(scored)} hits, top {len(results)} returned")
        return results
