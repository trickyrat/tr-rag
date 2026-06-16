# RAG 系统检索质量优化与重构方案

> 当前基线：Hit@1 ≈ 0.50（105 chunks / 9 文档 / 60 条测试查询）

## 诊断总览

| 层级      | 当前实现                                                             | 主要问题                                      | 预期收益 |
| :-------- | :------------------------------------------------------------------- | :-------------------------------------------- | :------- |
| Chunking  | `MarkdownHeaderTextSplitter`，仅按 h1/h2/h3 切分，无大小控制、无重叠 | 块粒度不可控，断头去尾丢失上下文              | ⭐⭐⭐ 高   |
| Embedding | `microsoft/harrier-oss-v1-0___6b`，无 query/passage 前缀             | 大模型未做检索指令适配，非专用 embedding 模型 | ⭐⭐ 中    |
| BM25 分词 | 正则 `[\u4e00-\u9fff][a-z0-9]+，中文按单字切                         | 中文单字级 BM25 几乎无效，关键词匹配完全退化  | ⭐⭐⭐ 高   |
| Rerank    | 仅 RRF 融合（向量+BM25），无 Cross-Encoder                           | 融合后无精排，排序质量有天花板                | ⭐⭐⭐ 高   |
| 检索参数  | top_k=3, vector_k=5, bm25_k=5, rrf_constant=60                       | 候选集太小、RRF 参数未调优                    | ⭐⭐ 中    |


## Phase 1：BM25 中文分词修复（投入产出比最高 🔥）

**问题**：`_tokenize()` 对中文按单字切分：

```python
# 当前："水解过程" → ["水","解","过","程"]
re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text)
```

单字级 BM25 完全无法捕捉中文词语语义。

**方案**：在 `_tokenize()` 中引入 jieba 分词，改动仅 10 行：

```bash
uv add jieba
```

```python
# src/rag/retrieval_optimization.py
import jieba

def _tokenize(text: str) -> List[str]:
    text = text.lower().strip()
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
    if has_cjk:
        tokens = list(jieba.cut(text))
        tokens = [t.strip() for t in tokens if t.strip() and len(t.strip()) >= 1]
        return tokens
    return re.findall(r"[a-z0-9]+", text)
```

**预期收益**：Hit@1 +5~10%

## Phase 2：Chunking 重构

**问题**：

1. 仅按 h1/h2/h3 切分，无大小控制（可能 30 字符或 2000+ 字符）
2. 无 chunk 重叠，边界信息丢失
3. `markdown_splitter.py` 中 `MarkdownParser` 已实现代码块保留 + 表格识别 + `RecursiveCharacterTextSplitter` 兜底，但 `DocumentChunker` 完全没使用

**方案**：

- `DocumentChunker._markdown_header_split()` 改为调用 `MarkdownParser.split_markdown_documents()`
- `config.py` 新增 `chunk_size=512`, `chunk_overlap=64`
- 分块后打印统计分布（min/max/mean）

**预期收益**：Hit@1 +10~15%

## Phase 3：Cross-Encoder Reranker

**问题**：RRF 是无监督启发式融合，没有精排能力。

**方案**：

新建 `src/rag/reranker.py`，使用 `BAAI/bge-reranker-v2-m3`（中英双语）：

```python
class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        from FlagEmbedding import FlagReranker
        self.reranker = FlagReranker(model_name, use_fp16=True)

    def rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.compute_score(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]
```

在 `HybridRetriever._get_relevant_documents()` 中 RRF 融合后增加精排步骤：

```bash
vector_search → BM25_search → RRF_merge → [CrossEncoder_rerank] → top_k
```

通过 `use_reranker: bool = False` 开关控制。

```bash
uv add FlagEmbedding
```

预期收益：Hit@1 +10~20%

## Phase 4：Embedding 模型切换

**问题**：`microsoft/harrier-oss-v1-0___6b` 是通用 LLM，不是专用 embedding 模型；未使用 query/passage 指令前缀。

**方案**：

切换到 `BAAI/bge-m3`（多语言 SOTA，1024 维，dense + sparse 双检）：

```python
# config.py
embedding_model: str = "BAAI/bge-m3"

# vector_store_builder.py 添加指令前缀
self.embeddings = HuggingFaceEmbeddings(
    model_name=self.model_name,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
    query_instruction="为这个句子生成表示以用于检索相关文章：",
)
```

**预期收益**：Hit@1 +5~15%

## Phase 5：参数网格搜索

使用 `ExperimentRunner` 对 `vector_k`、`bm25_k`、`rrf_constant`、`top_k` 做网格搜索，找到最优组合。同时改用 `similarity_search_with_score` 在 RRF 中使用真实分数。

**预期收益**：Hit@1 +3~8%


## 实施优先级

| 顺序 | Phase                  | 改动量       | 风险 | 预期提升 |
| :--- | :--------------------- | :----------- | :--- | :------- |
| 🔥 1  | BM25 jieba 分词        | 极小（10行） | 低   | +5~10%   |
| 2    | Chunking 重构          | 中（50行）   | 中   | +10~15%  |
| 3    | Cross-Encoder Reranker | 大（新文件） | 低   | +10~20%  |
| 4    | Embedding 切换         | 小（改配置） | 中   | +5~15%   |
| 5    | 参数调优               | 小           | 低   | +3~8%    |

**目标**：最终 Hit@1 ≥ 0.75，MRR ≥ 0.85。