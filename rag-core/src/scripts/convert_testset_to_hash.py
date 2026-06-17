"""Convert testset from chunk_id-based to content_hash-based matching.

Unlike the old approach (chunk_id → content_hash lookup, which breaks
when chunks are regenerated), this script uses keyword-based matching
between the query text and chunk content_preview to find the correct
chunks in the CURRENT chunk_mapping.jsonl.

Usage:
    # 1. Run rag_runner.py first to generate fresh chunk_mapping.jsonl
    uv run rag_runner.py

    # 2. Convert the testset (reads testset_new.jsonl, writes testset.jsonl)
    python src/scripts/convert_testset_to_hash.py

    # 3. Re-run evaluation — metrics should now be meaningful
    uv run rag_runner.py
"""
import json
import re
import sys
from pathlib import Path

CHUNK_MAPPING = "test/evaluation/chunk_mapping.jsonl"
TESTSET_IN = "test/evaluation/testset_new.jsonl"  # source (may have chunk_ids)
TESTSET_OUT = "test/evaluation/testset.jsonl"     # output (content_hash mode)


# ── tokenisation ──────────────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    """Extract meaningful tokens from text (Chinese + English)."""
    text = text.lower().strip()
    # Chinese characters as individual tokens + English words
    tokens = re.findall(r"[\u4e00-\u9fff]|[a-z0-9]{2,}", text)
    return tokens


def bigrams(text: str) -> list[str]:
    """Character bigrams for fuzzy Chinese matching."""
    chars = re.findall(r"[\u4e00-\u9fff]", text)
    return [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]


# ── stop words ────────────────────────────────────────────────────────
_STOP = {
    "的", "是", "在", "了", "和", "也", "就", "都", "而", "及", "与",
    "着", "或", "我们", "你们", "他们", "它们", "这个", "那个", "这些",
    "那些", "什么", "怎么", "如何", "为什么", "多少", "可以", "需要",
    "应该", "能够", "可能", "会", "要", "有", "不",
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "it", "its", "this", "that", "these", "those", "and", "or",
    "but", "not", "no", "what", "which", "who", "how", "when",
    "where", "why", "if", "then", "than", "about", "into",
}


def query_keywords(query: str) -> set[str]:
    """Extract meaningful keywords from a query."""
    tokens = set(tokenize(query))
    bigs = set(bigrams(query))
    keywords = (tokens | bigs) - _STOP
    return {k for k in keywords if len(k) >= 2}


# ── matching ──────────────────────────────────────────────────────────
def score_chunk(content_preview: str, query_kw: set[str]) -> float:
    """Score how well a chunk's content matches the query keywords."""
    content_lower = content_preview.lower()
    hits = sum(1 for kw in query_kw if kw in content_lower)
    return hits / max(len(query_kw), 1)


def find_best_chunks(
    query: str,
    chunks: list[dict],
    top_n: int = 3,
    min_score: float = 0.0,
) -> list[str]:
    """Find the top-N chunks whose content_preview best matches the query.
    Returns list of content_hashes.
    """
    query_kw = query_keywords(query)
    if not query_kw:
        return []

    scored = []
    for c in chunks:
        s = score_chunk(c.get("content_preview", ""), query_kw)
        if s > min_score:
            scored.append((s, c.get("content_hash")))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [ch for _, ch in scored[:top_n] if ch]


# ── main ──────────────────────────────────────────────────────────────
def main():
    # 1. Load current chunk mapping (with content_hash)
    if not Path(CHUNK_MAPPING).exists():
        print(f"ERROR: {CHUNK_MAPPING} not found. Run rag_runner.py first.")
        sys.exit(1)

    all_chunks: list[dict] = []
    with open(CHUNK_MAPPING, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_chunks.append(json.loads(line))

    print(f"Loaded {len(all_chunks)} chunks from {CHUNK_MAPPING}")

    # 2. Load testset
    if not Path(TESTSET_IN).exists():
        print(f"ERROR: {TESTSET_IN} not found.")
        sys.exit(1)

    cases = []
    with open(TESTSET_IN, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    print(f"Loaded {len(cases)} test cases from {TESTSET_IN}")

    # 3. For each query, find matching chunks via keyword overlap
    converted = []
    already_hash = 0
    matched = 0
    unmatched = 0

    for case in cases:
        query = case.get("query", "").strip()
        if not query:
            continue

        # Already content_hash mode — keep as-is
        if "relevant_content_hashes" in case:
            converted.append(case)
            already_hash += 1
            continue

        # Keyword match: find chunks whose content best matches this query
        hashes = find_best_chunks(query, all_chunks, top_n=3, min_score=0.05)

        if hashes:
            new_case = {"query": query, "relevant_content_hashes": hashes}
            for field in ("category", "difficulty", "query_type"):
                if field in case:
                    new_case[field] = case[field]
            converted.append(new_case)
            matched += 1
        else:
            unmatched += 1
            print(f"WARNING: No matching chunks found for: {query[:60]}")

    # 4. Write output
    with open(TESTSET_OUT, "w", encoding="utf-8") as f:
        for case in converted:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print("\nConversion complete:")
    print(f"  Already content-hash:  {already_hash}")
    print(f"  Matched by keywords:   {matched}")
    print(f"  Unmatched (skipped):   {unmatched}")
    print(f"  Output:                {TESTSET_OUT} ({len(converted)} cases)")


if __name__ == "__main__":
    main()
    print(f"  Output: {TESTSET_OUT}")


if __name__ == "__main__":
    main()
