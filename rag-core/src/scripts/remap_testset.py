"""Remap existing testset placeholder chunk IDs to real chunk IDs."""
import json
from collections import defaultdict

# Load chunk mapping
chunks = []
with open("test/evaluation/chunk_mapping.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

# Build lookup: doc_name -> list of chunk_ids
by_doc = defaultdict(list)
for c in chunks:
    by_doc[c["doc_name"]].append(c)

def find_chunk_id(doc_name, keywords):
    """Find chunk from doc_name whose content_preview contains all keywords."""
    for c in chunks:
        if c["doc_name"] == doc_name:
            if all(kw.lower() in c["content_preview"].lower() for kw in keywords):
                return c["chunk_id"]
    return None

# Load existing testset
existing = []
with open("test/evaluation/testset.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            existing.append(json.loads(line))

# Separate: existing placeholder entries (fake UUIDs) vs new entries (real UUIDs)
valid_ids = {c["chunk_id"] for c in chunks}
placeholders = []
real_entries = []

for tc in existing:
    ids = tc["relevant_chunk_ids"]
    if all(cid in valid_ids for cid in ids):
        real_entries.append(tc)
    else:
        placeholders.append(tc)

print(f"Real entries: {len(real_entries)}, Placeholders: {len(placeholders)}")

# Map placeholder queries to real chunk IDs using content matching
# Query -> (doc_name, [keywords]) mapping
query_map = {
    "SmartFlow 支持哪些渠道接入？": ("smartflow", [["SmartFlow 智能客服平台"]]),
    "部署 SmartFlow 需要什么硬件配置？": ("smartflow", [["硬件最低配置"]]),
    "如何用 Docker 安装 SmartFlow？": ("smartflow", [["快速安装"]]),
    "发送消息的 API 端点是什么？": ("smartflow", [["API 接口"]]),
    "ChromaDB 持久化路径如何配置？": ("smartflow", [["ChromaDB 持久化"]]),
    "容器启动失败，日志显示 chromadb 连接失败怎么办？": ("smartflow", [["故障排查"]]),
    "知识库更新的步骤是什么？": ("smartflow", [["知识库更新流程"]]),
    "如何提升检索速度？": ("smartflow", [["性能优化建议"]]),
    "RRF 常数 rrf_constant 设置多少合适？": ("smartflow", [["RRF 融合参数"]]),
    "召回准确率低的原因有哪些？": ("smartflow", [["召回准确率低于"]]),
    "如何配置 Webhook？": ("smartflow", [["Webhook 事件"]]),
    "BGE-large-zh 模型的召回率大概多少？": ("smartflow", [["嵌入模型对比"]]),
    "SmartFlow 有哪些核心模块？": ("smartflow", [["SmartFlow 智能客服平台"]]),
    "Docker 安装后如何访问？": ("smartflow", [["快速安装"]]),
    "API 返回 429 错误代表什么？": ("smartflow", [["API 接口"]]),
    "ChromaDB 数据目录权限不足如何解决？": ("smartflow", [["故障排查"]]),
    "知识库回滚怎么操作？": ("smartflow", [["知识库更新流程"]]),
    "百万级知识库建议如何部署？": ("smartflow", [["性能优化建议"]]),
    "RRF 小常数和大常数的区别？": ("smartflow", [["RRF 融合参数"]]),
    "重排序能提升多少准确率？": ("smartflow", [["召回准确率低于"]]),
    "Webhook 支持哪些事件？": ("smartflow", [["Webhook 事件"]]),
    "GTE-Qwen2-1.5B 显存占用多少？": ("smartflow", [["嵌入模型对比"]]),
    "是否支持微信渠道？": ("smartflow", [["SmartFlow 智能客服平台"]]),
    "推荐的分块大小是多少？": ("smartflow", [["性能优化建议"]]),
}

remapped = []
for tc in placeholders:
    q = tc["query"]
    if q in query_map:
        doc_name, kw_groups = query_map[q]
        ids = []
        for kws in kw_groups:
            cid = find_chunk_id(doc_name, kws)
            if cid:
                ids.append(cid)
        if ids:
            remapped.append({
                "query": q,
                "relevant_chunk_ids": ids,
                "category": "product",
                "difficulty": "medium",
                "query_type": "general",
            })
        else:
            print(f"WARNING: Could not remap: {q}")
    else:
        print(f"WARNING: No mapping for: {q}")

print(f"Remapped: {len(remapped)} queries")

# Merge: remapped placeholders + real new entries
# Also deduplicate against the new entries we already generated
final = remapped + real_entries

# Deduplicate by query
seen = set()
deduped = []
for tc in final:
    if tc["query"] not in seen:
        seen.add(tc["query"])
        deduped.append(tc)

print(f"Final after dedup: {len(deduped)} queries")

# Write
out_path = "test/evaluation/testset.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for tc in deduped:
        f.write(json.dumps(tc, ensure_ascii=False) + "\n")

# Validate
for i, tc in enumerate(deduped):
    for cid in tc["relevant_chunk_ids"]:
        assert cid in valid_ids, f"Case {i}: invalid chunk_id {cid} in '{tc['query'][:50]}'"

print("All chunk IDs valid!")
