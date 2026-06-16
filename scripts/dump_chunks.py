"""Dump chunk IDs per file with content previews for query-to-chunk mapping."""
import json
from collections import defaultdict

chunks = []
with open("test/evaluation/chunk_mapping.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

by_doc = defaultdict(list)
for c in chunks:
    by_doc[c["doc_name"]].append(c)

for doc_name in sorted(by_doc.keys()):
    items = by_doc[doc_name]
    cat = items[0]["primary_category"]
    sub = items[0].get("sub_category", "")
    path = f"{cat}/{sub}" if sub else cat
    print(f"\n=== {path}/{doc_name} ({len(items)} chunks) ===")
    for item in items:
        preview = item["content_preview"].replace("\n", " ")[:120]
        cid = item["chunk_id"]
        print(f"  [{cid[:16]}...] {preview}")
