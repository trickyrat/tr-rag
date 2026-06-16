from langchain_core.indexing import index, InMemoryRecordManager, IndexingResult
import chromadb
import torch
import logging
from typing import List, Optional, Literal
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

COLLECTION_NAME = "knowledge_base"


class VectorStoreBuilder:
    def __init__(
        self,
        model_name: str,
        index_save_path: str,
    ):
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.embeddings = None
        self.vectorstore: Optional[Chroma] = None
        self._client: Optional[chromadb.ClientAPI] = None
        self._record_manager: Optional[InMemoryRecordManager] = None
        self.setup_embeddings()

    def setup_embeddings(self):
        """Setup embeddings with local file cache"""
        logger.info(f"Initializing embedding model: {self.model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
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

        vectorstore = self._get_vectorstore()
        record_manager = self._get_record_manager()

        result = index(
            docs_source=chunks,
            record_manager=record_manager,
            vector_store=vectorstore,
            cleanup=cleanup,
            source_id_key="source",
            key_encoder="sha256"
        )

        logger.info(
            f"Indexing complete: +{result['num_added']} ~{result['num_updated']} "
            f"-{result['num_deleted']} = {result['num_skipped']}"
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
        """
        Reset the vector store collection
        """
        client = self._get_client()
        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            pass
        self.vectorstore = None

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
