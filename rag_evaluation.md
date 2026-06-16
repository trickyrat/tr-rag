# RAG 系统检索评估指南

本文档介绍如何对本项目的 RAG（检索增强生成）系统进行检索质量评估，包括测试集构建、单次评估、实验对比和参数网格搜索。

---

## 目录

1. [项目架构概述](#1-项目架构概述)
2. [环境准备](#2-环境准备)
3. [构建测试集](#3-构建测试集)
4. [运行检索评估](#4-运行检索评估)
5. [实验对比与网格搜索](#5-实验对比与网格搜索)
6. [评估指标说明](#6-评估指标说明)
7. [结果解读与优化建议](#7-结果解读与优化建议)

---

## 1. 项目架构概述

### RAG 核心模块 (`src/rag/`)

| 模块 | 类 | 功能 |
|------|-----|------|
| `document_chunker.py` | `DocumentChunker` | 加载 Markdown 文档，按标题层级分块，增强元数据（类别、路径） |
| `vector_store_builder.py` | `VectorStoreBuilder` | 基于 HuggingFace Embedding + ChromaDB 构建向量索引，支持增量索引 |
| `retrieval_optimization.py` | `RetrievalOptimization` / `HybridRetriever` | 向量检索 + BM25 关键词检索 + RRF 融合排序，支持元数据过滤 |
| `generation_integration.py` | `GenerationIntegration` | LLM 答案生成，支持查询重写、查询路由、多类型回答（基础/分步/列表） |

### 评估模块 (`src/evaluation/`)

| 模块 | 类 | 功能 |
|------|-----|------|
| `evaluator.py` | `RAGEvaluator` | 加载 JSONL 测试集，对检索函数进行评估 |
| `metrics.py` | — | 计算 Hit@K、Precision@K、Recall@K、MRR、NDCG@K |
| `experiment_runner.py` | `ExperimentRunner` | 批量运行多组检索配置，支持网格搜索，结果存 CSV |
| `testset_builder.py` | `TestSetBuilder` | 半自动构建测试集：采样 → 生成查询建议 → 人工标注 → 转换 |

### 配置 (`config.py`)

```python
@dataclass
class RAGConfig:
    knowledge_base_path: str = "./knowledge_base"       # 知识库路径
    index_save_path: str = "./knowledge_base_store"      # 向量索引持久化路径
    embedding_model: str = "..."                          # HuggingFace 嵌入模型路径
    llm_model: str = "kimi-k2.6"                          # LLM 模型名
    evaluation_testset_path: str = "./test/evaluation/testset.jsonl"  # 测试集路径
    top_k: int = 3                                        # 检索返回数量
    temperature: float = 0.1
    max_tokens: int = 2048
```

---

## 2. 环境准备

### 2.1 安装依赖

```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -r requirements.txt
```

核心依赖：
- `langchain` / `langchain-chroma` / `langchain-huggingface` — RAG 框架与向量存储
- `chromadb` — 向量数据库
- `sentence-transformers` + `torch` — 嵌入模型
- `rank-bm25` — BM25 关键词检索
- `langchain-deepseek` — LLM 接入（生成回答时使用）

### 2.2 环境变量

```bash
# 如果使用 LLM 生成回答，需要设置 API Key
export MOONSHOT_API_KEY="your-api-key"

# 可选：DeepSeek API Key（GenerationIntegration 使用）
export DEEPSEEK_API_KEY="your-deepseek-key"
```

### 2.3 准备测试集

确保测试集文件存在：
```
test/evaluation/testset.jsonl
```

该文件每行一个 JSON 对象。支持两种匹配模式：

**内容哈希模式（推荐，抗重新分块）：**
```json
{"query": "你的检索查询", "relevant_content_hashes": ["abc123...", "def456..."]}
```

**Chunk ID 模式（旧版，每次重新分块后会失效）：**
```json
{"query": "你的检索查询", "relevant_chunk_ids": ["uuid1", "uuid2"]}
```

> **重要**：由于 `rag_runner.py` 每次运行都会重新分块并生成新的 UUID，建议使用内容哈希模式。分块时会对每个 chunk 的 `page_content` 计算 MD5 哈希，存在 `content_hash` 元数据中。只要文档内容不变，哈希就不变，测试集可跨多次运行复用。

---

## 3. 构建测试集

### 3.1 整体流程

```mermaid
flowchart LR
    A[知识库 Markdown 文档] --> B[DocumentChunker 分块]
    B --> C[rag_runner.py<br/>自动导出 chunk_mapping.jsonl]
    C --> D[scripts/convert_testset_to_hash.py<br/>转换测试集为 content_hash 模式]
    D --> E[testset.jsonl<br/>使用 relevant_content_hashes]
    E --> F[rag_runner.py<br/>执行评估]
    F --> G[输出 Hit@1 / MRR 等指标]


### 3.2 推荐工作流（一键式）

`rag_runner.py` 已内置 chunk 映射导出功能。每次运行后都会自动生成 `test/evaluation/chunk_mapping.jsonl`。

```bash
# 步骤 1：运行 rag_runner.py，自动分块 & 导出 chunk_mapping.jsonl
uv run rag_runner.py

# 步骤 2：将测试集从 chunk_id 模式转换为 content_hash 模式（只需执行一次）
python scripts/convert_testset_to_hash.py

# 步骤 3：再次运行，这次指标就是正确的了
uv run rag_runner.py
```

> **注意**：步骤 2 只需在测试集首次创建或知识库内容变更后执行一次。之后即使重新分块，`content_hash` 也不会改变。

### 3.3 辅助脚本（按需使用）

| 脚本 | 用途 |
|------|------|
| `scripts/dump_chunks.py` | 按文档分组打印所有 chunk 的 ID 和内容预览，用于人工编写查询时参考 |
| `scripts/generate_testset.py` | 基于 `chunk_mapping.jsonl` 批量生成覆盖所有知识库文档的测试用例 |
| `scripts/remap_testset.py` | 将测试集中的旧 chunk_id 替换为当前 chunk_id（chunk_id 模式专用） |
| `scripts/convert_testset_to_hash.py` | **推荐**：将测试集从 `relevant_chunk_ids` 转换为 `relevant_content_hashes`（内容哈希模式） |

### 3.4 半自动构建测试集

使用 `TestSetBuilder` 类逐步构建，最终通过 `convert_testset_to_hash.py` 转为内容哈希模式：

```python
from src.evaluation.testset_builder import TestSetBuilder
from src.rag import DocumentChunker
from config import DEFAULT_CONFIG

# 1. 分块
chunker = DocumentChunker(DEFAULT_CONFIG.knowledge_base_path)
chunker.load_documents()
chunks = chunker.chunk_documents()

# 2. 创建 TestSetBuilder
builder = TestSetBuilder(chunks, output_dir="./test/evaluation")

# 3. 随机采样供人工参考
builder.sample_chunks_for_annotation(num_samples=100)

# 4. 生成标注模板（含自动查询建议）
builder.create_annotation_template("test/evaluation/annotations.jsonl")

# 5. 人工标注：编辑 annotations.jsonl，填写 query 字段，设置 relevant=true/false

# 6. 转换为 chunk_id 格式的测试集
TestSetBuilder.convert_annotations_to_testset(
    annotation_file="test/evaluation/annotations.jsonl",
    output_testset="test/evaluation/testset.jsonl",
)

# 7. 转为 content_hash 模式（抗重新分块）
# 在终端执行：python scripts/convert_testset_to_hash.py
```

---

## 4. 运行检索评估

### 4.1 使用 `rag_runner.py` 快速评估

项目根目录下的 `rag_runner.py` 是最简单的评估入口：

```bash
python rag_runner.py
```

该脚本会：
1. 初始化 `DocumentChunker` 并加载/分块知识库文档
2. 初始化 `VectorStoreBuilder` 并构建向量索引
3. 创建 `RetrievalOptimization` 检索器
4. 加载测试集并用 `RAGEvaluator` 进行评估
5. 输出 Hit@1 等聚合指标

### 4.2 使用 `RAGEvaluator` 自定义评估

```python
import logging
from config import DEFAULT_CONFIG
from src.rag import DocumentChunker, VectorStoreBuilder, RetrievalOptimization
from src.evaluation import RAGEvaluator

logging.basicConfig(level=logging.INFO)

config = DEFAULT_CONFIG

# 1. 加载文档与分块
chunker = DocumentChunker(config.knowledge_base_path)
chunker.load_documents()
chunks = chunker.chunk_documents()

# 2. 构建索引
builder = VectorStoreBuilder(
    model_name=config.embedding_model,
    index_save_path=config.index_save_path,
)
builder.index_documents(chunks)
vectorstore = builder.get_vectorstore()

# 3. 创建检索器
retriever = RetrievalOptimization(vectorstore=vectorstore, chunks=chunks)

# 4. 定义检索函数（供评估器调用）
def retrieve_fn(query: str, top_k: int = 5):
    return retriever.hybrid_search(query, top_k=top_k)

# 5. 创建评估器并执行评估
evaluator = RAGEvaluator(config.evaluation_testset_path)

# 完整评估（含每查询详细结果）
results = evaluator.evaluate(
    retrieve_fn=retrieve_fn,
    top_k=config.top_k,
    k_list=[1, 3, 5],
    per_query_results=True,  # 设为 True 可查看每个查询的详细信息
)

# 打印结果
print(f"总查询数: {results['total_queries']}")
print("\n聚合指标:")
for key, value in results["aggregate"].items():
    print(f"  {key}: {value:.4f}")

# 查看每个查询的详情
if "per_query" in results:
    for pq in results["per_query"]:
        print(f"\n查询: {pq['query'][:60]}...")
        print(f"  检索到: {len(pq['retrieved_ids'])} 相关: {len(pq['relevant_ids'])}")
        print(f"  MRR: {pq['metrics']['mrr']:.4f}  Hit@1: {pq['metrics']['hit@1']:.4f}")
```

### 4.3 子集快速评估（迭代调试）

当测试集较大时，可以先在子集上快速验证：

```python
# 仅在前 20 条查询上评估
results = evaluator.evaluate_on_subset(
    retrieve_fn=retrieve_fn,
    top_k=5,
    max_queries=20,
)
print(f"Hit@1 (subset): {results['aggregate']['hit@1']:.4f}")
print(f"MRR (subset): {results['aggregate']['mrr']:.4f}")
```

---

## 5. 实验对比与网格搜索

### 5.1 基本用法

`ExperimentRunner` 用于对比多种检索配置：

```python
from src.evaluation.experiment_runner import ExperimentRunner

runner = ExperimentRunner(
    vectorstore=vectorstore,
    chunks=chunks,
    testset_path="test/evaluation/testset.jsonl",
    results_dir="src/evaluation/results",
)

# 基线实验
def baseline_builder():
    return RetrievalOptimization(vectorstore, chunks)

baseline = runner.run_experiment("baseline", baseline_builder, top_k=5)
```

### 5.2 网格搜索

自动遍历参数组合，找到最优配置：

```python
param_grid = [
    {"rrf_constant": 30, "vector_k": 3, "bm25_k": 3},
    {"rrf_constant": 30, "vector_k": 5, "bm25_k": 5},
    {"rrf_constant": 60, "vector_k": 5, "bm25_k": 5},
    {"rrf_constant": 60, "vector_k": 8, "bm25_k": 8},
    {"rrf_constant": 90, "vector_k": 5, "bm25_k": 5},
]

grid_results = runner.run_grid_search(param_grid, {})

# 找出 Hit@1 最高的配置
best = max(grid_results, key=lambda x: float(x.get("hit@1", 0)))
print(f"最优 Hit@1: {best['hit@1']} → 配置: {best['experiment']}")
```

### 5.3 查看历史实验结果

```python
# 加载所有历史实验
history = ExperimentRunner.load_results("src/evaluation/results")
for row in history:
    print(f"{row['experiment']}: Hit@1={row.get('hit@1', 'N/A')}, MRR={row.get('mrr', 'N/A')}")
```

实验结果保存在 `src/evaluation/results/experiments.csv` 中，可直接用 Excel 或 Pandas 进行分析。

---

## 6. 评估指标说明

以下是本评估系统计算的所有指标：

| 指标 | 公式 | 含义 | 取值范围 |
|------|------|------|----------|
| **Hit@K** | $\text{Hit@K} = \begin{cases} 1 & \text{如果前 K 个结果中有相关文档} \\ 0 & \text{否则} \end{cases}$ | 前 K 个结果中是否至少命中一个相关文档 | 0 或 1 |
| **Precision@K** | $\text{P@K} = \frac{\vert \text{前 K 个结果中相关的数量} \vert}{K}$ | 前 K 个结果中相关文档的比例 | [0, 1] |
| **Recall@K** | $\text{R@K} = \frac{\vert \text{前 K 个结果中相关的数量} \vert}{\vert \text{全部相关文档数} \vert}$ | 全部相关文档中有多少被检索到（前 K） | [0, 1] |
| **MRR** | $\text{MRR} = \frac{1}{\text{第一个相关文档的排名}}$ | 第一个相关文档排名的倒数，衡量相关文档出现的位置 | (0, 1] |
| **NDCG@K** | $\text{NDCG@K} = \frac{\sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}}{\text{理想DCG@K}}$ | 归一化折损累计增益，考虑排名位置权重 | [0, 1] |

### 指标解读优先级

1. **Hit@1** — 最重要：衡量检索能否将正确答案排在第一，直接影响 RAG 答案质量
2. **MRR** — 衡量第一个相关文档的平均排名，值越高说明相关文档出现得越早
3. **Hit@3 / Hit@5** — 衡量前 3/5 名的覆盖率
4. **NDCG@K** — 综合排名位置权重的评估，更适合排序质量调优
5. **Precision@K / Recall@K** — 精度与召回，需要权衡

---

## 7. 结果解读与优化建议

### 7.1 典型输出示例

```
Evaluation complete: Hit@1=0.7234, MRR=0.8512
总查询数: 65

聚合指标:
  hit@1: 0.7234
  hit@3: 0.8912
  hit@5: 0.9456
  precision@1: 0.7234
  precision@3: 0.5123
  precision@5: 0.3876
  recall@1: 0.5123
  recall@3: 0.7234
  recall@5: 0.8345
  ndcg@1: 0.7234
  ndcg@3: 0.7654
  ndcg@5: 0.7891
  mrr: 0.8512
  total_queries: 65
```

### 7.2 常用优化方向

| 问题 | 可能的优化方向 |
|------|---------------|
| Hit@1 偏低（< 0.6） | 升级 Embedding 模型（如 BGE-large、GTE-Qwen2）；调整 RRF 融合参数 |
| MRR 偏低 | 减小 `rrf_constant`（如 30），强调排名靠前的结果 |
| Recall 偏低 | 增大 `top_k` 或 `vector_k`/`bm25_k`；检查分块是否合理 |
| 某些查询类别表现差 | 检查分块策略：`MarkdownHeaderTextSplitter` 是否正确保留了内容语义 |
| BM25 贡献不足 | 增大 `bm25_k`，确保 BM25 索引的文档与检索文档一致 |

### 7.3 关键参数调优

检索系统的可调参数（在 `HybridRetriever` 中）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vector_k` | 5 | 向量检索的候选数量 |
| `bm25_k` | 5 | BM25 检索的候选数量 |
| `top_k` | 3 | RRF 融合后最终返回的数量 |
| `rrf_constant` | 60 | RRF 平滑常数：越小越激进（强调第一），越大越平滑 |

推荐调优流程：

```mermaid
flowchart TD
    A[基线评估] --> B{Hit@1 >= 0.7?}
    B -->|是| C[调优 top_k 和 rrf_constant]
    B -->|否| D[检查分块质量/更换 Embedding 模型]
    C --> E[网格搜索 vector_k / bm25_k]
    E --> F[记录最优配置]
    D --> E
    F --> G[最终评估]
```

### 7.4 注意事项

1. **推荐使用 `content_hash` 模式**：`chunk_id` 在每次重新分块后会变化，导致测试集失效。`content_hash`（基于 `page_content` 的 MD5）只要文档内容不变就保持不变。
2. **嵌入模型路径**：`config.py` 中的 `embedding_model` 需指向本地有效的 HuggingFace 模型路径，否则向量检索会失败。
3. **内存占用**：`InMemoryRecordManager` 在重启后状态丢失，重建索引时会重新嵌入所有文档。
4. **GPU 加速**：如果可用 CUDA，嵌入模型会自动使用 GPU，否则回退 CPU。

---

## 快速参考

```bash
# 1. 首次运行（分块 + 导出映射）
uv run rag_runner.py

# 2. 转换测试集为 content_hash 模式（只需执行一次）
python scripts/convert_testset_to_hash.py

# 3. 再次运行评估（此后每次都能得到正确的指标）
uv run rag_runner.py

# 4. 查看实验对比结果
cat src/evaluation/results/experiments.csv
```
