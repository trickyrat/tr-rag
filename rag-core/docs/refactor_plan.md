# RAG 系统检索质量优化与重构方案

> **原始基线**：Hit@1 ≈ 0.50（harrier 模型 / 105 chunks / 60 查询）
> **当前状态**：Hit@1 = **0.7667** ✅（bge-m3 / 105 chunks / RRF rank-based 修复）

---

## 已完成变更

### ✅ 已实施

| 变更 | 文件 | 说明 |
|------|------|------|
| bge-m3 替代 harrier | `config.py`, `vector_store_builder.py` | `BGEM3Embeddings` 封装 `FlagEmbedding.BGEM3FlagModel`，支持 dense + sparse 双路检索 |
| BM25 移除 | `retrieval_optimization.py` | 原 `rank_bm25` + `_tokenize()` 替换为 bge-m3 `sparse_search_fn`，通过 RRF 融合 |
| Config 参数流修复 | `config.py`, `rag_runner.py`, `retrieval_optimization.py` | `vector_k/sparse_k/top_k/rrf_constant` 之前硬编码默认值，现已从 config 传入 `HybridRetriever` |
| Chunker 双模式开关 | `document_chunker.py`, `config.py` | `use_markdown_parser: bool` — False=旧 chunker(105块), True=新 MarkdownParser(186块) |
| `reset_collection()` 彻底清空 | `vector_store_builder.py` | 用 `shutil.rmtree` 替代 `delete_collection`，避免 ChromaDB UUID 目录残留 |
| Query instruction 设为空 | `vector_store_builder.py` | `BGEM3FlagModel` 内部已处理 query/passage 区分，不需要手动加前缀 |
| 下载模型脚本 | `src/scripts/download_model.py`, `pyproject.toml` | `uv run download-model bge-m3` 通过 ModelScope 下载 |
| **RRF 排序反转修复** | `retrieval_optimization.py` | 改为 rank-based RRF（标准公式），修复 ChromaDB L2 距离被当相似度用的 bug |
| **Cross-Encoder Reranker** | `retrieval_optimization.py` | `CrossEncoderReranker` + 集成进 `RetrievalOptimization.search()`，使用 `sentence_transformers.CrossEncoder`（避免 FlagReranker tokenizer 兼容问题） |

### ✅ 已验证

| 测试 | 结果 |
|------|------|
| bge-m3 裸模型相似度（query="云母水分测量公式" vs doc="## 云母的水分测量"） | **cosine=0.70** ✅ |
| RRF 排序修复后 (bge-m3 pure vector) | **Hit@1=0.7667, MRR=0.8306** ✅ |
| Cross-encoder 加载 | ✅ `sentence_transformers.CrossEncoder` 无 tokenizer 兼容问题 |
| MarkdownParser chunker (186块) | ❌ `convert-testset` 关键词匹配标错标签，暂不可用 |

---

## ⚠️ 根因分析 → ✅ 已修复

**根因：`HybridRetriever._rrf_rerank` 将 ChromaDB 的 L2 距离（低=好）直接当作相似度分数（高=好）使用，导致排序完全颠倒。**

修复后 bge-m3 pure vector 从 Hit@1≈0 提升到 **Hit@1=0.7667**。

---

## 待实施

| Phase | 内容 | 依赖 |
|-------|------|------|
| 2b | MarkdownParser chunker（需修复与测试集的兼容性） | 需测试集生成改为基于检索而非关键词 |
| 3b | Cross-Encoder 参数调优（模型已集成，待跑全量评估） | 需跑 `use_cross_encoder=True` |
| 5 | 参数网格搜索 | 需 baseline 稳定 |

---

## 当前配置速查

```python
# config.py 关键参数
embedding_model = "D:/models/BAAI/bge-m3"
top_k = 3
vector_k = 30
bm25_k = 30       # 实际是 sparse_k
rrf_constant = 60
chunk_size = 512
chunk_overlap = 64
use_markdown_parser = False   # 旧 chunker
use_parent_expansion = False
use_sparse = False            # pure vector 已经很好

# Cross-encoder (Phase 3) — 新增
use_cross_encoder = True
cross_encoder_model = "D:/models/BAAI/bge-reranker-v2-m3"
cross_encoder_top_k = 30
```

## 统一 API

```python
from rag import RetrievalOptimization

retriever = RetrievalOptimization(
    vectorstore=vs,
    chunks=chunks,
    cross_encoder_model="D:/models/BAAI/bge-reranker-v2-m3",  # None=禁用
    cross_encoder_candidate_k=30,
)

# 统一入口：hybrid → cross-encoder
docs = retriever.search("查询", top_k=3)

# Small-to-Big 模式
docs = retriever.search_with_parents("查询", top_k=3)

# 手动 rerank
docs = retriever.rerank("查询", docs, top_k=3)

# 检查状态
retriever.has_cross_encoder  # bool
```