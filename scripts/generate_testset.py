"""Generate the evaluation test set based on knowledge base chunk mapping.

This script reads chunk_mapping.jsonl and produces a comprehensive test set
with queries in both Chinese and English, covering all 9 knowledge base files.
"""

import json
from pathlib import Path
from collections import Counter

# Load chunk mapping
chunks = []
with open("test/evaluation/chunk_mapping.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))


# Build lookup: (doc_name, content_keyword) -> chunk_id
# For each doc, we need to find chunks by content keywords
def find_chunks(doc_name, keywords, n=1):
    """Find chunks from doc_name whose content_preview contains all keywords."""
    matches = []
    for c in chunks:
        if c["doc_name"] == doc_name:
            if all(kw.lower() in c["content_preview"].lower() for kw in keywords):
                matches.append(c["chunk_id"])
    return matches[:n] if n else matches


def find_chunk_ids(doc_name, keyword_groups):
    """For each keyword group, find the first matching chunk_id.
    keyword_groups: list of [keywords] where each group maps to one chunk_id.
    Returns list of chunk_ids (one per group).
    """
    result = []
    for kws in keyword_groups:
        ids = find_chunks(doc_name, kws, 1)
        if ids:
            result.append(ids[0])
    return result


def find_all_chunks(doc_name):
    """Get all chunk IDs for a document."""
    return [c["chunk_id"] for c in chunks if c["doc_name"] == doc_name]


test_cases = []

# ─── 1. pearl_powder.md (制造/云母) ─── Chinese ───
test_cases.extend(
    [
        {
            "query": "云母水分测量的公式是什么？",
            "relevant_chunk_ids": find_chunk_ids("pearl_powder", [["# 天然珠光"]]),
            "category": "manufacture/mica",
            "difficulty": "easy",
            "query_type": "factoid",
        },
        {
            "query": "水解过程中Sn液的配置需要多少量？不同系列有什么差别？",
            "relevant_chunk_ids": find_chunk_ids("pearl_powder", [["Sn液的配置"]]),
            "category": "manufacture/mica",
            "difficulty": "medium",
            "query_type": "factoid",
        },
        {
            "query": "TiCl4溶液稀释时需要注意哪些安全事项？",
            "relevant_chunk_ids": find_chunk_ids("pearl_powder", [["Ti液的配制"]]),
            "category": "manufacture/mica",
            "difficulty": "easy",
            "query_type": "procedural",
        },
        {
            "query": "Fe液的配制中，60g/L的浓度对应的是什么？",
            "relevant_chunk_ids": find_chunk_ids("pearl_powder", [["Fe液的配制"]]),
            "category": "manufacture/mica",
            "difficulty": "medium",
            "query_type": "factoid",
        },
        {
            "query": "银白和虹彩系列的水解包覆过程需要加入哪些溶液？金色系列和铁系列呢？",
            "relevant_chunk_ids": find_chunk_ids("pearl_powder", [["水解过程"]]),
            "category": "manufacture/mica",
            "difficulty": "medium",
            "query_type": "conceptual",
        },
        {
            "query": "金色系列加入Fe液时，温度和pH需要控制在什么范围？",
            "relevant_chunk_ids": find_chunk_ids("pearl_powder", [["水解过程"]]),
            "category": "manufacture/mica",
            "difficulty": "medium",
            "query_type": "factoid",
        },
        {
            "query": "云母水解完成后，取样洗涤、煅烧和刮卡的完整流程是什么？",
            "relevant_chunk_ids": find_chunk_ids("pearl_powder", [["取样洗涤"]]),
            "category": "manufacture/mica",
            "difficulty": "hard",
            "query_type": "procedural",
        },
        {
            "query": "刮卡时粉体与树脂的比例是多少？如何判断是否到样？",
            "relevant_chunk_ids": find_chunk_ids("pearl_powder", [["取样洗涤"]]),
            "category": "manufacture/mica",
            "difficulty": "medium",
            "query_type": "procedural",
        },
    ]
)

# ─── 2. smartflow.md (产品) ─── Chinese ───
test_cases.extend(
    [
        {
            "query": "SmartFlow 智能客服平台的核心模块有哪些？",
            "relevant_chunk_ids": find_chunk_ids(
                "smartflow", [["SmartFlow 智能客服平台"]]
            ),
            "category": "product",
            "difficulty": "easy",
            "query_type": "factoid",
        },
        {
            "query": "SmartFlow 部署的硬件最低配置和软件依赖是什么？",
            "relevant_chunk_ids": find_chunk_ids("smartflow", [["硬件最低配置"]])
            + find_chunk_ids("smartflow", [["软件依赖"]]),
            "category": "product",
            "difficulty": "easy",
            "query_type": "factoid",
        },
        {
            "query": "使用 Docker 部署 SmartFlow 的完整命令是什么？启动后如何访问？",
            "relevant_chunk_ids": find_chunk_ids("smartflow", [["快速安装"]]),
            "category": "product",
            "difficulty": "easy",
            "query_type": "procedural",
        },
        {
            "query": "SmartFlow 知识库更新的完整步骤是什么？如何回滚？",
            "relevant_chunk_ids": find_chunk_ids("smartflow", [["知识库更新流程"]]),
            "category": "product",
            "difficulty": "medium",
            "query_type": "procedural",
        },
        {
            "query": "SmartFlow 的 Webhook 支持哪些事件类型？如何配置？",
            "relevant_chunk_ids": find_chunk_ids("smartflow", [["Webhook 事件"]]),
            "category": "product",
            "difficulty": "medium",
            "query_type": "reference",
        },
        {
            "query": "RRF 融合参数 rrf_constant 的设置建议是什么？小常数和大常数有什么区别？",
            "relevant_chunk_ids": find_chunk_ids("smartflow", [["RRF 融合参数"]]),
            "category": "product",
            "difficulty": "medium",
            "query_type": "conceptual",
        },
        {
            "query": "召回准确率低于90%的可能原因有哪些？如何优化？",
            "relevant_chunk_ids": find_chunk_ids("smartflow", [["召回准确率低于"]]),
            "category": "product",
            "difficulty": "hard",
            "query_type": "conceptual",
        },
        {
            "query": "SmartFlow 百万级知识库的性能优化建议有哪些？",
            "relevant_chunk_ids": find_chunk_ids("smartflow", [["性能优化建议"]]),
            "category": "product",
            "difficulty": "medium",
            "query_type": "conceptual",
        },
    ]
)

# ─── 3. rag_system_concepts.md ─── Chinese/English mixed ───
test_cases.extend(
    [
        {
            "query": "RAG 系统的四个核心组件是什么？",
            "relevant_chunk_ids": find_chunk_ids(
                "rag_system_concepts", [["Core Components"]]
            )
            + find_chunk_ids("rag_system_concepts", [["Document Ingestion"]])
            + find_chunk_ids("rag_system_concepts", [["Vector Database"]])
            + find_chunk_ids("rag_system_concepts", [["Embedding Model"]])
            + find_chunk_ids("rag_system_concepts", [["Generation Model"]]),
            "category": "technologies/rag",
            "difficulty": "easy",
            "query_type": "factoid",
        },
        {
            "query": "RAG 的索引阶段包含哪些步骤？查询阶段呢？",
            "relevant_chunk_ids": find_chunk_ids(
                "rag_system_concepts", [["Indexing Phase"]]
            )
            + find_chunk_ids("rag_system_concepts", [["Query Phase"]]),
            "category": "technologies/rag",
            "difficulty": "medium",
            "query_type": "conceptual",
        },
        {
            "query": "RAG 系统中 Embedding Model 的作用是什么？常用的模型有哪些？",
            "relevant_chunk_ids": find_chunk_ids(
                "rag_system_concepts", [["Embedding Model"]]
            ),
            "category": "technologies/rag",
            "difficulty": "easy",
            "query_type": "conceptual",
        },
        {
            "query": "RAG 系统相比纯 LLM 有哪些优势？",
            "relevant_chunk_ids": find_chunk_ids(
                "rag_system_concepts", [["Benefits of RAG"]]
            ),
            "category": "technologies/rag",
            "difficulty": "easy",
            "query_type": "conceptual",
        },
        {
            "query": "文档分块策略如何影响 RAG 系统性能？有什么最佳实践？",
            "relevant_chunk_ids": find_chunk_ids(
                "rag_system_concepts", [["Document Chunking Strategies"]]
            )
            + find_chunk_ids("rag_system_concepts", [["Best Practices"]]),
            "category": "technologies/rag",
            "difficulty": "hard",
            "query_type": "conceptual",
        },
        {
            "query": "Hybrid Retrieval 混合检索是什么？它有什么优势？",
            "relevant_chunk_ids": find_chunk_ids(
                "rag_system_concepts", [["Hybrid Retrieval"]]
            ),
            "category": "technologies/rag",
            "difficulty": "medium",
            "query_type": "conceptual",
        },
        {
            "query": "Simple RAG 和 Advanced RAG 有什么区别？",
            "relevant_chunk_ids": find_chunk_ids(
                "rag_system_concepts", [["Simple RAG vs"]]
            ),
            "category": "technologies/rag",
            "difficulty": "medium",
            "query_type": "conceptual",
        },
    ]
)

# ─── 4. vector_database.md ─── Chinese ───
test_cases.extend(
    [
        {
            "query": "FAISS 是什么？主要用于什么场景？",
            "relevant_chunk_ids": find_chunk_ids("vector_database", [["FAISS"]]),
            "category": "technologies/rag",
            "difficulty": "easy",
            "query_type": "factoid",
        },
        {
            "query": "ChromaDB 是什么类型的数据库？有什么特点？",
            "relevant_chunk_ids": find_chunk_ids("vector_database", [["ChromaDB"]]),
            "category": "technologies/rag",
            "difficulty": "easy",
            "query_type": "factoid",
        },
        {
            "query": "Milvus 和 ChromaDB 有什么不同？",
            "relevant_chunk_ids": find_chunk_ids("vector_database", [["Milvus"]])
            + find_chunk_ids("vector_database", [["ChromaDB"]]),
            "category": "technologies/rag",
            "difficulty": "medium",
            "query_type": "conceptual",
        },
        {
            "query": "常用的向量数据库有哪些？请列举并简要说明。",
            "relevant_chunk_ids": find_all_chunks("vector_database"),
            "category": "technologies/rag",
            "difficulty": "easy",
            "query_type": "factoid",
        },
        {
            "query": "Pinecone 是开源的吗？它和 FAISS 有什么区别？",
            "relevant_chunk_ids": find_chunk_ids("vector_database", [["Pinecone"]])
            + find_chunk_ids("vector_database", [["FAISS"]]),
            "category": "technologies/rag",
            "difficulty": "medium",
            "query_type": "conceptual",
        },
    ]
)

# ─── 5. langchain_rag_implementation.md ─── English ───
test_cases.extend(
    [
        {
            "query": "What are the core concepts of LangChain?",
            "relevant_chunk_ids": find_chunk_ids(
                "langchain_rag_implementation", [["Core Concepts"]]
            ),
            "category": "technologies/langchain",
            "difficulty": "easy",
            "query_type": "conceptual",
        },
        {
            "query": "What is the role of Chains in LangChain?",
            "relevant_chunk_ids": find_chunk_ids(
                "langchain_rag_implementation", [["Chains"]]
            ),
            "category": "technologies/langchain",
            "difficulty": "easy",
            "query_type": "conceptual",
        },
        {
            "query": "What are Agents in LangChain and how do they work?",
            "relevant_chunk_ids": find_chunk_ids(
                "langchain_rag_implementation", [["Agents"]]
            ),
            "category": "technologies/langchain",
            "difficulty": "medium",
            "query_type": "conceptual",
        },
        {
            "query": "What are the four layers in the LangChain RAG architecture?",
            "relevant_chunk_ids": find_chunk_ids(
                "langchain_rag_implementation", [["RAG Architecture"]]
            ),
            "category": "technologies/langchain",
            "difficulty": "medium",
            "query_type": "factoid",
        },
        {
            "query": "What are the best practices for document chunking in RAG?",
            "relevant_chunk_ids": find_chunk_ids(
                "langchain_rag_implementation", [["Document Chunking"]]
            )
            + find_chunk_ids("langchain_rag_implementation", [["Best Practices"]]),
            "category": "technologies/langchain",
            "difficulty": "hard",
            "query_type": "conceptual",
        },
        {
            "query": "What advanced RAG techniques does LangChain support?",
            "relevant_chunk_ids": find_chunk_ids(
                "langchain_rag_implementation", [["Advanced RAG"]]
            )
            + find_chunk_ids("langchain_rag_implementation", [["Multi-Query"]])
            + find_chunk_ids(
                "langchain_rag_implementation", [["Contextual Compression"]]
            )
            + find_chunk_ids("langchain_rag_implementation", [["Self-Querying"]]),
            "category": "technologies/langchain",
            "difficulty": "hard",
            "query_type": "factoid",
        },
        {
            "query": "What are common challenges in RAG implementations and how to solve them?",
            "relevant_chunk_ids": find_chunk_ids(
                "langchain_rag_implementation", [["Common Challenges"]]
            )
            + find_chunk_ids("langchain_rag_implementation", [["Hallucinations"]])
            + find_chunk_ids(
                "langchain_rag_implementation", [["Outdated Information"]]
            ),
            "category": "technologies/langchain",
            "difficulty": "hard",
            "query_type": "conceptual",
        },
    ]
)

# ─── 6. python_basics.md ─── English ───
test_cases.extend(
    [
        {
            "query": "What are the key features of Python as a programming language?",
            "relevant_chunk_ids": find_chunk_ids("python_basics", [["Key Features"]]),
            "category": "technologies/python",
            "difficulty": "easy",
            "query_type": "factoid",
        },
        {
            "query": "What are Python's built-in data types? Give examples.",
            "relevant_chunk_ids": find_chunk_ids("python_basics", [["Data Types"]])
            + find_chunk_ids("python_basics", [["Numbers"]])
            + find_chunk_ids("python_basics", [["Strings"]]),
            "category": "technologies/python",
            "difficulty": "easy",
            "query_type": "factoid",
        },
        {
            "query": "How do conditional statements work in Python?",
            "relevant_chunk_ids": find_chunk_ids(
                "python_basics", [["Conditional Statements"]]
            ),
            "category": "technologies/python",
            "difficulty": "easy",
            "query_type": "conceptual",
        },
        {
            "query": "How do for loops and while loops work in Python?",
            "relevant_chunk_ids": find_chunk_ids("python_basics", [["Loops"]]),
            "category": "technologies/python",
            "difficulty": "easy",
            "query_type": "conceptual",
        },
        {
            "query": "How are functions defined in Python? What is the syntax?",
            "relevant_chunk_ids": find_chunk_ids("python_basics", [["Functions"]]),
            "category": "technologies/python",
            "difficulty": "easy",
            "query_type": "reference",
        },
        {
            "query": "What are Python lists and how do you create and use them?",
            "relevant_chunk_ids": find_chunk_ids("python_basics", [["Lists"]]),
            "category": "technologies/python",
            "difficulty": "easy",
            "query_type": "conceptual",
        },
    ]
)

# ─── 7. python_libraries_text_processing.md ─── English ───
test_cases.extend(
    [
        {
            "query": "What built-in string methods does Python provide for text processing?",
            "relevant_chunk_ids": find_chunk_ids(
                "python_libraries_text_processing", [["Built-in String"]]
            ),
            "category": "technologies/python",
            "difficulty": "easy",
            "query_type": "factoid",
        },
        {
            "query": "How do you use regular expressions in Python? Give examples of pattern matching.",
            "relevant_chunk_ids": find_chunk_ids(
                "python_libraries_text_processing", [["Regular Expressions"]]
            ),
            "category": "technologies/python",
            "difficulty": "medium",
            "query_type": "conceptual",
        },
        {
            "query": "What is NLTK and what NLP tasks can it perform?",
            "relevant_chunk_ids": find_chunk_ids(
                "python_libraries_text_processing", [["NLTK"]]
            ),
            "category": "technologies/python",
            "difficulty": "medium",
            "query_type": "conceptual",
        },
        {
            "query": "How does spaCy differ from NLTK? What are its advantages?",
            "relevant_chunk_ids": find_chunk_ids(
                "python_libraries_text_processing", [["spaCy"]]
            ),
            "category": "technologies/python",
            "difficulty": "medium",
            "query_type": "conceptual",
        },
        {
            "query": "How can Pandas be used for text processing tasks?",
            "relevant_chunk_ids": find_chunk_ids(
                "python_libraries_text_processing", [["Pandas for Text"]]
            ),
            "category": "technologies/python",
            "difficulty": "medium",
            "query_type": "conceptual",
        },
        {
            "query": "How do you select the right Python text processing library for your needs?",
            "relevant_chunk_ids": find_chunk_ids(
                "python_libraries_text_processing", [["Selecting the Right"]]
            ),
            "category": "technologies/python",
            "difficulty": "hard",
            "query_type": "conceptual",
        },
    ]
)

# ─── 8. text_processing_python.md ─── English ───
test_cases.extend(
    [
        {
            "query": "What is text processing and why is Python well-suited for it?",
            "relevant_chunk_ids": find_chunk_ids(
                "text_processing_python", [["Text Processing with Python"]]
            ),
            "category": "technologies/python",
            "difficulty": "easy",
            "query_type": "conceptual",
        },
        {
            "query": "What are Python's string methods for case conversion?",
            "relevant_chunk_ids": find_chunk_ids(
                "text_processing_python", [["Case Conversion"]]
            ),
            "category": "technologies/python",
            "difficulty": "easy",
            "query_type": "reference",
        },
        {
            "query": "How do you manage whitespace in Python strings using built-in methods?",
            "relevant_chunk_ids": find_chunk_ids(
                "text_processing_python", [["Whitespace"]]
            ),
            "category": "technologies/python",
            "difficulty": "easy",
            "query_type": "reference",
        },
        {
            "query": "How do you search and replace text in Python strings?",
            "relevant_chunk_ids": find_chunk_ids(
                "text_processing_python", [["Search and Replace"]]
            ),
            "category": "technologies/python",
            "difficulty": "easy",
            "query_type": "reference",
        },
        {
            "query": "How do you use regular expressions in Python? Give an example of pattern matching and substitution.",
            "relevant_chunk_ids": find_chunk_ids(
                "text_processing_python", [["Regular Expressions"]]
            ),
            "category": "technologies/python",
            "difficulty": "medium",
            "query_type": "conceptual",
        },
        {
            "query": "What common text cleaning operations can you perform in Python?",
            "relevant_chunk_ids": find_chunk_ids(
                "text_processing_python", [["Text Cleaning"]]
            ),
            "category": "technologies/python",
            "difficulty": "medium",
            "query_type": "conceptual",
        },
    ]
)

# ─── 9. api_cheatsheet.md ─── English ───
test_cases.extend(
    [
        {
            "query": "How do you write a basic SELECT query with WHERE clause in SQLAlchemy 2.0?",
            "relevant_chunk_ids": find_chunk_ids("api_cheatsheet", [["Basic query"]]),
            "category": "technologies/sqlalchemy",
            "difficulty": "easy",
            "query_type": "reference",
        },
        {
            "query": "How do you use the IN clause in SQLAlchemy?",
            "relevant_chunk_ids": find_chunk_ids("api_cheatsheet", [["IN clause"]]),
            "category": "technologies/sqlalchemy",
            "difficulty": "easy",
            "query_type": "reference",
        },
        {
            "query": "How do you perform JOIN queries in SQLAlchemy? What is the difference between join() and outerjoin()?",
            "relevant_chunk_ids": find_chunk_ids("api_cheatsheet", [["JOIN Query"]]),
            "category": "technologies/sqlalchemy",
            "difficulty": "medium",
            "query_type": "reference",
        },
        {
            "query": "How do you use AND and OR conditions with SQLAlchemy WHERE clauses?",
            "relevant_chunk_ids": find_chunk_ids("api_cheatsheet", [["Basic query"]]),
            "category": "technologies/sqlalchemy",
            "difficulty": "easy",
            "query_type": "reference",
        },
        {
            "query": "How do you perform sorting and pagination in SQLAlchemy?",
            "relevant_chunk_ids": find_chunk_ids(
                "api_cheatsheet", [["Sort and Pagination"]]
            ),
            "category": "technologies/sqlalchemy",
            "difficulty": "medium",
            "query_type": "reference",
        },
        {
            "query": "How do INSERT, UPDATE, and DELETE operations work in SQLAlchemy 2.0 style?",
            "relevant_chunk_ids": find_chunk_ids("api_cheatsheet", [["插入、更新"]]),
            "category": "technologies/sqlalchemy",
            "difficulty": "medium",
            "query_type": "reference",
        },
        {
            "query": "How do you use aggregate functions and GROUP BY in SQLAlchemy?",
            "relevant_chunk_ids": find_chunk_ids("api_cheatsheet", [["Aggregate"]]),
            "category": "technologies/sqlalchemy",
            "difficulty": "medium",
            "query_type": "reference",
        },
    ]
)


# ─── Flatten chunk IDs (multi-chunk queries may return nested lists) ───
def flatten_chunk_ids(ids):
    """Flatten nested list of chunk IDs into a single list."""
    result = []
    for item in ids:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result


# ─── Validate and write ───
valid_ids = {c["chunk_id"] for c in chunks}
issues = []
final_cases = []

for i, tc in enumerate(test_cases):
    ids = flatten_chunk_ids(tc["relevant_chunk_ids"])
    # Remove empty strings and deduplicate
    ids = list(dict.fromkeys([x for x in ids if x]))

    if not ids:
        issues.append(f"#{i}: NO matching chunks for query: {tc['query'][:60]}...")
        continue

    invalid = [x for x in ids if x not in valid_ids]
    if invalid:
        issues.append(
            f"#{i}: INVALID chunk IDs {invalid} for query: {tc['query'][:60]}..."
        )
        continue

    final_cases.append(
        {
            "query": tc["query"],
            "relevant_chunk_ids": ids,
            "category": tc.get("category", ""),
            "difficulty": tc.get("difficulty", "medium"),
            "query_type": tc.get("query_type", "general"),
        }
    )

# Print issues
if issues:
    print(f"WARNING: {len(issues)} test cases have issues:")
    for issue in issues:
        print(f"  {issue}")
    print()

# Write new test set
out_path = "test/evaluation/testset_new.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for tc in final_cases:
        f.write(json.dumps(tc, ensure_ascii=False) + "\n")

# ─── Merge with existing testset ───
existing_path = "test/evaluation/testset.jsonl"
existing_queries = set()
existing_cases = []
if Path(existing_path).exists():
    with open(existing_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tc = json.loads(line)
                existing_queries.add(tc["query"])
                existing_cases.append(tc)

# Deduplicate against existing
new_unique = [tc for tc in final_cases if tc["query"] not in existing_queries]
merged = existing_cases + new_unique

merged_path = "test/evaluation/testset.jsonl"
with open(merged_path, "w", encoding="utf-8") as f:
    for tc in merged:
        f.write(json.dumps(tc, ensure_ascii=False) + "\n")

print(f"New queries generated: {len(final_cases)}")
print(f"Unique new (not in existing): {len(new_unique)}")
print(
    f"Merged total: {len(merged)} (existing: {len(existing_cases)} + new: {len(new_unique)})"
)
print(f"Written to: {merged_path}")

# ─── Summary by category ───

by_cat = Counter(tc.get("category", "unknown") for tc in merged)
by_diff = Counter(tc.get("difficulty", "medium") for tc in merged)
by_type = Counter(tc.get("query_type", "general") for tc in merged)
print(f"\nBy category: {dict(by_cat)}")
print(f"By difficulty: {dict(by_diff)}")
print(f"By query type: {dict(by_type)}")
