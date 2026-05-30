import chromadb
import torch
import logging
from typing import List, Optional
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

COLLECTION_NAME = "knowledge_base"

class VectorStoreBuilder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        index_save_path: str = "./vector_index",
    ):
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.embeddings = None
        self.vectorstore: Optional[Chroma] = None
        self._client: Optional[chromadb.ClientAPI] = None
        self.setup_embeddings()

    def setup_embeddings(self):
        """Setup embeddings"""
        logger.info(f"Initializing embedding model: {self.model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info("Embedding model initialized successfully")

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

    def build_vector_index(self, chunks: List[Document]) -> Chroma:
        """
        Build a Chroma vector index from the provided document chunks.

        Args:
            chunks: List of Document objects to be indexed

        Returns:
            Chroma vector store
        """
        logger.info("Building Chroma vector index...")

        if not chunks:
            raise ValueError("Document chunk list cannot be empty")

        self.reset_collection()

        client = self._get_client()
        # Build Chroma vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            client=client,
            collection_name=COLLECTION_NAME,
        )

        logger.info(f"Vector index built successfully with {len(chunks)} documents")
        return self.vectorstore

    def add_documents(self, new_chunks: List[Document]):
        """
        Add new documents to the existing index

        Args:
            new_chunks: List of new document chunks
        """
        if not new_chunks:
            raise ValueError("New document chunk list cannot be empty")

        vectorstore = self._get_vectorstore()
        logger.info(f"Adding {len(new_chunks)} new documents to the index...")
        vectorstore.add_documents(new_chunks)
        logger.info("New documents added successfully")

    def delete_documents_by_source(self, sources: List[str]):
        """
        Delete documents from the index based on their source

        Args:
            sources: List of document sources to be deleted
        """
        if not sources:
            raise ValueError("Source list cannot be empty")

        client = self._get_client()
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return

        for source in sources:
            results = collection.get(where={"source": source})
            if results and results["ids"]:
                collection.delete(ids=results["ids"])
                logger.info(
                    f"Deleted {len(results['ids'])} documents with source: {source}"
                )

        self.vectorstore = None

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
