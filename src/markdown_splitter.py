import re
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document
from typing import List, Dict, Any


class MarkdownParser:
    """Parser for markdown files."""

    def split_markdown_documents(
        self, docs: List[Document], chunk_size: int = 400, chunk_overlap: int = 60
    ) -> List[Document]:
        """Split markdown documents while preserving semantic integrity and hierarchical relationships.

        Args:
            docs: List of documents to split
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks

        Returns:
            List of document chunks with preserved hierarchical metadata
        """
        all_chunks: List[Document] = []

        # Define headers to split on with their corresponding metadata keys
        headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
            ("#####", "h5"),
            ("######", "h6"),
        ]

        for doc in docs:
            content = doc.page_content
            source = doc.metadata.get("source", "unknown")

            # First, parse the document to identify structural elements
            structured_chunks = self._parse_markdown_structures(content, source)

            # Process each structured chunk
            processed_chunks = self._process_structured_chunks(
                structured_chunks, doc, chunk_size, chunk_overlap, headers_to_split_on
            )
            all_chunks.extend(processed_chunks)

        return all_chunks

    def _process_structured_chunks(
        self,
        structured_chunks: List[Dict[str, Any]],
        original_doc: Document,
        chunk_size: int,
        chunk_overlap: int,
        headers_to_split_on: List[tuple],
    ) -> List[Document]:
        """Process structured chunks, splitting them further if necessary."""
        all_chunks = []

        for struct_chunk in structured_chunks:
            chunk_text = struct_chunk["content"]
            chunk_metadata = {**original_doc.metadata, **struct_chunk["metadata"]}

            # If the chunk is smaller than our target size, add it directly
            if len(chunk_text) <= chunk_size:
                all_chunks.append(
                    Document(page_content=chunk_text, metadata=chunk_metadata)
                )
            else:
                # For larger chunks, split by headers first, then by text splitter if still too large
                header_splitter = MarkdownHeaderTextSplitter(
                    headers_to_split_on=headers_to_split_on,
                    strip_headers=False,
                )

                header_docs = header_splitter.split_text(chunk_text)

                for header_doc in header_docs:
                    # Update metadata with header information
                    combined_metadata = {**chunk_metadata}
                    for key in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                        if key in header_doc.metadata:
                            combined_metadata[key] = header_doc.metadata[key]

                    # If still too large, use character-based splitter with sentence awareness
                    if len(header_doc.page_content) > chunk_size:
                        char_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            separators=[
                                "\n\n",
                                "\n",
                                "。",
                                ".",
                                "！",
                                "!",
                                "？",
                                "?",
                                "；",
                                ";",
                                "，",
                                ",",
                                " ",
                                "",
                            ],
                            keep_separator=True,
                        )

                        sub_docs = char_splitter.split_documents([header_doc])

                        for sub_doc in sub_docs:
                            # Combine all header levels in the metadata
                            final_metadata = {**combined_metadata}

                            # Preserve the hierarchy by combining all heading levels
                            for key in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                                if (
                                    key in sub_doc.metadata
                                    and key not in final_metadata
                                ):
                                    final_metadata[key] = sub_doc.metadata[key]

                            all_chunks.append(
                                Document(
                                    page_content=sub_doc.page_content,
                                    metadata=final_metadata,
                                )
                            )
                    else:
                        all_chunks.append(
                            Document(
                                page_content=header_doc.page_content,
                                metadata=combined_metadata,
                            )
                        )

        return all_chunks

    def _parse_markdown_structures(
        self, content: str, source: str
    ) -> List[Dict[str, Any]]:
        """Parse markdown content into structural elements that should stay together.

        Args:
            content: Markdown content to parse
            source: Source identifier for the document

        Returns:
            List of structured content blocks with metadata
        """
        lines = content.split("\n")
        structures = []

        i = 0
        while i < len(lines):
            line = lines[i]

            if self._is_code_block_start(line):
                structures.append(self._extract_code_block(lines, i, source))
                i = self._skip_until_code_block_end(lines, i)
            elif self._is_table_start(line):
                structures.append(self._extract_table(lines, i, source))
                i = self._skip_until_table_end(lines, i)
            else:
                # Handle regular content blocks
                content_block_data, next_index = self._extract_content_block(
                    lines, i, source
                )
                if content_block_data:
                    structures.append(content_block_data)
                i = next_index

        return structures

    def _is_code_block_start(self, line: str) -> bool:
        """Check if the line starts a code block."""
        return line.strip().startswith("```")

    def _is_table_start(self, line: str) -> bool:
        """Check if the line starts a table."""
        return line.strip().startswith("|")

    def _extract_code_block(
        self, lines: List[str], start_idx: int, source: str
    ) -> Dict[str, Any]:
        """Extract a complete code block."""
        block_type = (
            lines[start_idx].strip().split("```")[1]
            if len(lines[start_idx].strip()) > 3
            else "code"
        )

        end_idx = start_idx + 1
        while end_idx < len(lines) and not lines[end_idx].strip().startswith("```"):
            end_idx += 1

        if end_idx < len(lines):  # Include the closing ```
            end_idx += 1

        code_block = "\n".join(lines[start_idx:end_idx])
        return {
            "content": code_block,
            "metadata": {
                "structure_type": "code_block",
                "code_language": block_type,
                "source": source,
            },
        }

    def _skip_until_code_block_end(self, lines: List[str], start_idx: int) -> int:
        """Skip through the lines until the end of the current code block."""
        i = start_idx + 1
        while i < len(lines) and not lines[i].strip().startswith("```"):
            i += 1

        if i < len(lines):  # Skip the closing ```
            i += 1

        return i

    def _extract_table(
        self, lines: List[str], start_idx: int, source: str
    ) -> Dict[str, Any]:
        """Extract a complete table."""
        table_lines = [lines[start_idx]]
        i = start_idx + 1

        # Collect all table rows and potential header separator
        while i < len(lines) and (
            self._is_table_row(lines[i]) or self._is_table_separator(lines[i])
        ):
            table_lines.append(lines[i])
            i += 1

        table_content = "\n".join(table_lines)
        return {
            "content": table_content,
            "metadata": {"structure_type": "table", "source": source},
        }

    def _is_table_row(self, line: str) -> bool:
        """Check if the line is a table row."""
        return line.strip().startswith("|")

    def _is_table_separator(self, line: str) -> bool:
        """Check if the line is a table separator."""
        return bool(re.match(r"^[\|\-\s\+]+$", line.strip()))

    def _skip_until_table_end(self, lines: List[str], start_idx: int) -> int:
        """Skip through the lines until the end of the current table."""
        i = start_idx + 1

        # Skip until we find a line that is not part of the table
        while i < len(lines) and (
            self._is_table_row(lines[i]) or self._is_table_separator(lines[i])
        ):
            i += 1

        return i

    def _extract_content_block(
        self, lines: List[str], start_idx: int, source: str
    ) -> tuple:
        """Extract a regular content block that doesn't have special structure."""
        content_lines = []
        i = start_idx

        while i < len(lines):
            current_line = lines[i]

            # Check if this line starts a new structural element that should be handled separately
            if (
                self._is_code_block_start(current_line)
                or self._is_table_start(current_line)
                or current_line.strip().startswith("#")
            ):
                # If we found a header and have accumulated content, return the current block
                if current_line.strip().startswith("#") and content_lines:
                    break
                # If we found another structural element, return current block and let it be processed separately
                elif self._is_code_block_start(current_line) or self._is_table_start(
                    current_line
                ):
                    break
                # Otherwise, if it's a header and we haven't accumulated content yet, continue
                elif current_line.strip().startswith("#"):
                    content_lines.append(current_line)
                    i += 1
            else:
                content_lines.append(current_line)
                i += 1

        # Join the content lines and clean up
        content_block = "\n".join(content_lines).strip()
        if content_block:
            return {
                "content": content_block,
                "metadata": {"structure_type": "text", "source": source},
            }, i
        else:
            return None, i
