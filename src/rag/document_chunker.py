import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
import uuid

import json

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Data preparation class - responsible for knowledge base data loading, cleaning, and preprocessing for RAG system.
    """

    def __init__(self, data_path: str) -> None:
        """
        Initialize data preparation class.

        Args:
            data_path: Root directory path of the knowledge base.
        """
        self.data_path = data_path
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.parent_child_map: Dict[str, str] = {}

    def load_documents(self, file_paths: Optional[List[str]] = None) -> List[Document]:
        """
        Load documents from the knwoledge base directory

        Args:
            file_paths: List of file paths to load. If None, all files in the data directory will be loaded.

        Returns:
            List of loaded documents
        """
        logger.info(f"Loading documents from {self.data_path}")

        documents = []
        data_path_obj = Path(self.data_path)

        if file_paths is not None:
            md_files = [data_path_obj / fp for fp in file_paths]
        else:
            md_files = list(data_path_obj.rglob("*.md"))

        for md_file in md_files:
            try:
                with open(file=md_file, mode="r", encoding="utf-8") as f:
                    content = f.read()

                try:
                    data_root = Path(self.data_path).resolve()
                    relative_path = (
                        Path(md_file).resolve().relative_to(data_root).as_posix()
                    )
                except Exception:
                    relative_path = Path(md_file).as_posix()

                parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()

                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(md_file),
                        "parent_id": parent_id,
                        "doc_type": "parent",
                    },
                )

                documents.append(doc)

            except Exception as e:
                logger.error(f"Error reading file {md_file}: {e}")

        for doc in documents:
            self._enhance_metadata(doc)

        self.documents = documents
        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents

    def _enhance_metadata(self, doc: Document) -> None:
        """
        Enhance document metadata - extract category information from file path hierarchy

        Args:
            doc: Document to enhance metadata for
        """
        file_path = Path(doc.metadata.get("source", ""))
        data_root = Path(self.data_path).resolve()

        try:
            relative = file_path.resolve().relative_to(data_root)
            path_hierarchy = list(relative.parent.parts)
        except (ValueError, RuntimeError):
            path_hierarchy = []

        doc.metadata["doc_name"] = file_path.stem
        doc.metadata["path_hierarchy"] = path_hierarchy
        doc.metadata["primary_category"] = (
            path_hierarchy[0] if len(path_hierarchy) >= 1 else ""
        )
        doc.metadata["sub_category"] = (
            path_hierarchy[1] if len(path_hierarchy) >= 2 else ""
        )

    def chunk_documents(self) -> List[Document]:
        """
        Markdown structure-aware chunking

        Returns:
            List of chunked documents
        """
        logger.info("Performing Markdown structure-aware chunking...")

        if not self.documents:
            raise ValueError("Please load documents first.")

        chunks = self._markdown_header_split()

        for i, chunk in enumerate(chunks):
            if "chunk_id" not in chunk.metadata:
                chunk.metadata["chunk_id"] = str(uuid.uuid4())
            chunk.metadata["batch_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        self.chunks = chunks
        logger.info(f"Markdown chunking complete, generated {len(chunks)} chunks")
        return chunks

    def _markdown_header_split(self) -> List[Document]:
        """
        Perform structured splitting using Markdown header splitter

        Returns:
            List of documents split by header structure
        """
        headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )

        all_chunks = []

        for doc in self.documents:
            try:
                content_preview = doc.page_content[:200]
                has_headers = any(
                    line.strip().startswith("#") for line in content_preview.split("\n")
                )

                if not has_headers:
                    logger.warning(
                        f"Document {doc.metadata.get('doc_name', 'unknown')} has no Markdown headers"
                    )
                    logger.debug(f"Content preview: {content_preview}")

                md_chunks = markdown_splitter.split_text(doc.page_content)

                logger.debug(
                    f"Document {doc.metadata.get('doc_name', 'unknown')} split into {len(md_chunks)} chunks"
                )

                if len(md_chunks) <= 1:
                    logger.warning(
                        f"Document {doc.metadata.get('doc_name', 'unknown')} could not split by headers, may lack header structure"
                    )

                parent_id = doc.metadata["parent_id"]

                for i, chunk in enumerate(md_chunks):
                    child_id = str(uuid.uuid4())
                    content_hash = hashlib.sha256(
                        chunk.page_content.encode("utf-8")
                    ).hexdigest()
                    chunk.metadata.update(doc.metadata)
                    chunk.metadata.update(
                        {
                            "chunk_id": child_id,
                            "parent_id": parent_id,
                            "doc_type": "child",
                            "chunk_index": i,
                            "content_hash": content_hash,
                        }
                    )

                    self.parent_child_map[child_id] = parent_id

                all_chunks.extend(md_chunks)

            except Exception as e:
                logger.warning(
                    f"Markdown splitting failed for {doc.metadata.get('source', 'unknown')}: {e}"
                )
                all_chunks.append(doc)

        logger.info(
            f"Markdown structure splitting complete, generated {len(all_chunks)} structured chunks"
        )
        return all_chunks

    def filter_documents(self, field: str, value: str) -> List[Document]:
        """
        Filter documents by metadata field.

        Args:
            field: Metadata field name (e.g. primary_category, sub_category, doc_name)
            value: Expected field Value

        Returns:
            Filtered list of documents
        """
        return [doc for doc in self.documents if doc.metadata.get(field) == value]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get data statistics

        Returns:
            Dictionary of statistics
        """
        if not self.documents:
            return {}

        primary_categories: Dict[str, int] = {}
        subcategories: Dict[str, int] = {}

        for doc in self.documents:
            pc = doc.metadata.get("primary_category", "")
            sc = doc.metadata.get("sub_category", "")
            primary_categories[pc] = primary_categories.get(pc, 0) + 1
            if sc:
                subcategories[sc] = subcategories.get(sc, 0) + 1

        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "primary_categories": primary_categories,
            "subcategories": subcategories,
            "avg_chunk_size": sum(
                chunk.metadata.get("chunk_size", 0) for chunk in self.chunks
            )
            / len(self.chunks)
            if self.chunks
            else 0,
        }

    @staticmethod
    def get_statistics_from_chunks(chunks: List[Document]) -> Dict[str, Any]:
        """
        Get statistics from a list of document chunks.

        Args:
            chunks: List of document chunks

        Returns:
            Dictionary of statistics
        """
        if not chunks:
            return {}

        parent_ids = set()
        primary_categories: Dict[str, int] = {}
        subcategories: Dict[str, int] = {}

        for chunk in chunks:
            parent_id = chunk.metadata.get("parent_id", "")
            if parent_id:
                parent_ids.add(parent_id)

            pc = chunk.metadata.get("primary_category", "")
            sc = chunk.metadata.get("sub_category", "")
            if pc:
                primary_categories[pc] = primary_categories.get(pc, 0) + 1
            if sc:
                subcategories[sc] = subcategories.get(sc, 0) + 1

        return {
            "total_documents": len(parent_ids),
            "total_chunks": len(chunks),
            "primary_categories": primary_categories,
            "subcategories": subcategories,
            "avg_chunk_size": sum(
                chunk.metadata.get("chunk_size", 0) for chunk in chunks
            )
            / len(chunks)
            if chunks
            else 0,
        }

    def export_metadata(self, output_path: str):
        """
        Export metadata to a JSON file.

        Args:
            output_path: Output file path
        """

        metadata_list = []
        for doc in self.documents:
            metadata_list.append(
                {
                    "source": doc.metadata.get("source"),
                    "doc_name": doc.metadata.get("doc_name"),
                    "primary_category": doc.metadata.get("primary_category"),
                    "sub_category": doc.metadata.get("sub_category"),
                    "path_hierarchy": doc.metadata.get("path_hierarchy"),
                    "content_length": len(doc.page_content),
                }
            )

        with open(file=output_path, mode="w", encoding="utf-8") as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)

        logger.info(f"Metadata exported to: {output_path}")

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """
        Retrieves parent documents from child chunks (with smart deduplication).

        Args:
            child_chunks: List of retrieved child chunks

        Returns:
            Deduplicated list of parent documents, sorted by relevance
        """
        parent_relevance = {}
        parent_docs_map = {}

        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1

                if parent_id not in parent_docs_map:
                    for doc in self.documents:
                        if doc.metadata.get("parent_id") == parent_id:
                            parent_docs_map[parent_id] = doc
                            break

        sorted_parent_ids = sorted(
            parent_relevance.keys(), key=lambda x: parent_relevance[x], reverse=True
        )

        parent_docs = []
        for parent_id in sorted_parent_ids:
            if parent_id in parent_docs_map:
                parent_docs.append(parent_docs_map[parent_id])

        parent_info = []
        for doc in parent_docs:
            doc_name = doc.metadata.get("doc_name", "unknown")
            parent_id = doc.metadata.get("parent_id")
            relevance_count = parent_relevance.get(parent_id, 0)
            parent_info.append(f"{doc_name}({relevance_count}块)")

        logger.info(
            f"Found {len(child_chunks)} deduplicated parent documents from {len(parent_docs)} child chunks: {', '.join(parent_info)}"
        )
        return parent_docs
