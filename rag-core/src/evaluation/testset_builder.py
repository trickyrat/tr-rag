import json
import logging
import random
from pathlib import Path
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class TestSetBuilder:
    """
    Helper to semi-automatically create a test set from existing chunks.
    """

    def __init__(self, chunks: List[Document], output_dir: str):
        """
        Args:
            chunks: List of chunk documents (must have chunk_id metadata).
            output_dir: Directory to save test set drafts.
        """
        self.chunks = chunks
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def sample_chunks_for_annotation(
        self, num_samples: int = 100, seed: int = 42
    ) -> None:
        """
        Randomly sample chunks and write them to a file for manual annotation.

        Output: sample_chunks.jsonl with chunk content and metadata.

        Args:
            num_samples: Number of chunks to sample.
            seed: Random seed for reproducibility.
        """
        rng = random.Random(seed)
        sampled = rng.sample(self.chunks, min(num_samples, len(self.chunks)))
        output_file = self.output_dir / "sample_chunks.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for chunk in sampled:
                record = {
                    "chunk_id": chunk.metadata.get("chunk_id"),
                    "content": chunk.page_content[:500],  # preview
                    "source": chunk.metadata.get("source"),
                    "path_hierarchy": chunk.metadata.get("path_hierarchy"),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(sampled)} sample chunks to {output_file}")

    def generate_query_suggestions(self, chunk: Document) -> List[str]:
        """
        Generate potential queries for a chunk using language-aware heuristics.
        Supports both Chinese and English documents.
        """
        import re

        content = chunk.page_content.strip()
        lines = content.split("\n")

        # Extract heading
        heading = next(
            (l.strip("# ").strip() for l in lines if l.startswith("#")), ""
        )

        # Detect language: if CJK characters present, treat as Chinese
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", content[:500]))

        suggestions = []

        if heading:
            if has_cjk:
                suggestions = [
                    f"什么是{heading}？",
                    f"{heading}的关键要点是什么？",
                    f"请解释{heading}",
                ]
            else:
                suggestions = [
                    f"What is {heading}?",
                    f"Explain {heading}",
                    f"What are the key points of {heading}?",
                ]
        else:
            # First meaningful sentence
            first_sentence = ""
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or stripped.startswith("```"):
                    continue
                # Split on common sentence terminators
                for sep in ["。", ". ", "?\n", "!\n", "\n\n"]:
                    parts = stripped.split(sep, 1)
                    candidate = parts[0].strip()
                    if len(candidate) > 10:
                        first_sentence = candidate
                        break
                if first_sentence:
                    break

            if first_sentence:
                if has_cjk:
                    suggestions = [
                        first_sentence + "？",
                        f"关于{first_sentence[:30]}...的更多信息是什么？",
                    ]
                else:
                    question = first_sentence.rstrip(".") + "?"
                    suggestions = [
                        question,
                        f"What more can you tell me about {first_sentence[:50]}...?",
                    ]
            else:
                # Fallback
                preview = content[:100]
                if has_cjk:
                    suggestions = [f"请描述以下内容：{preview}"]
                else:
                    suggestions = [f"Describe the following: {preview}"]

        return suggestions

    def create_annotation_template(self, output_path: str) -> None:
        """
        Create a template annotation file for manual labeling.
        Each line: {"chunk_id": "...", "suggested_queries": [...], "query": "", "relevant": true/false}
        """
        template_file = Path(output_path)
        existing = []
        if template_file.exists():
            with open(template_file, "r", encoding="utf-8") as f:
                existing = [json.loads(line) for line in f if line.strip()]
            existing_ids = {item["chunk_id"] for item in existing}
        else:
            existing_ids = set()

        new_annotations = []
        for chunk in self.chunks:
            cid = chunk.metadata.get("chunk_id")
            if cid in existing_ids:
                continue
            suggested = self.generate_query_suggestions(chunk)
            new_annotations.append(
                {
                    "chunk_id": cid,
                    "suggested_queries": suggested,
                    "query": "",  # to be filled manually
                    "relevant": False,
                }
            )
            if len(new_annotations) >= 50:  # batch size
                break

        with open(template_file, "a", encoding="utf-8") as f:
            for ann in new_annotations:
                f.write(json.dumps(ann, ensure_ascii=False) + "\n")
        logger.info(f"Added {len(new_annotations)} new entries to {output_path}")

    @staticmethod
    def convert_annotations_to_testset(
        annotation_file: str,
        output_testset: str,
        min_queries_per_chunk: int = 1,
    ) -> None:
        """
        Convert annotated file to final test set format.

        Annotation file lines should have either:
          - {"chunk_id": str, "query": str, "relevant": bool}
          - {"chunk_ids": [str, ...], "query": str, "relevant": bool}  (multi-chunk)
        Only entries with relevant=True and non-empty query are kept.

        Args:
            annotation_file: Path to annotation JSONL file.
            output_testset: Path to output test set JSONL.
            min_queries_per_chunk: Minimum queries per chunk (informational, not enforced).
        """
        test_cases = []
        seen_queries = set()
        with open(annotation_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                query = data.get("query", "").strip()
                if not (data.get("relevant") and query):
                    continue

                # Support both single-chunk and multi-chunk annotations
                if "chunk_ids" in data:
                    chunk_ids = data["chunk_ids"]
                elif "chunk_id" in data:
                    chunk_ids = [data["chunk_id"]]
                else:
                    logger.warning(
                        "Skipping annotation: no chunk_id or chunk_ids field"
                    )
                    continue

                # Deduplicate queries
                if query in seen_queries:
                    logger.warning(
                        f"Skipping duplicate query: '{query[:80]}...'"
                    )
                    continue
                seen_queries.add(query)

                entry = {"query": query, "relevant_chunk_ids": chunk_ids}

                # Preserve optional metadata fields
                for field in ("category", "difficulty", "query_type"):
                    if field in data:
                        entry[field] = data[field]

                test_cases.append(entry)

        with open(output_testset, "w", encoding="utf-8") as f:
            for case in test_cases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
        logger.info(
            f"Converted {len(test_cases)} test cases to {output_testset}"
        )
