# TR-RAG

这是一个使用 LangChain、Sentence Transformers 和 ChromaDB 构建的检索增强生成（Retrieval-Augmented Generation, RAG）系统。

## 项目概述

RAG（检索增强生成）是一种结合了信息检索和文本生成的技术，允许大型语言模型在生成文本时访问外部知识库，从而产生更准确、更相关的回答。

## 技术栈

- **LangChain**: 用于构建语言模型应用程序的框架
- **Sentence Transformers**: 用于生成句子嵌入向量
- **ChromaDB**: 向量数据库，用于高效存储和检索文档向量
- **OpenAI API (可选)**: 用于实际的问答生成（需要API密钥）

## 项目结构

- [rag_system.py](./rag_system.py): 完整的RAG系统实现
- [simple_rag.py](./simple_rag.py): 简化版RAG系统（无需API密钥即可运行）
- [bbrag.ipynb](./bbrag.ipynb): Jupyter Notebook格式的实现
- [pyproject.toml](./pyproject.toml): 项目依赖配置文件

## 安装依赖

此项目使用 uv 包管理器，可以通过以下命令安装依赖：

```bash
uv sync
```

或者使用 pip:

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 简单运行演示

直接运行简化版的RAG系统（不需要OpenAI API密钥）：

```bash
python simple_rag.py
```

### 2. 使用Jupyter Notebook

启动Jupyter Lab并打开 [bbrag.ipynb](./bbrag.ipynb) 笔记本：

```bash
jupyter lab
```

### 3. 自定义使用

如果你想使用自己的文档，可以参考 [rag_system.py](./rag_system.py) 中的实现方式：

```python
from rag_system import RAGSystem

# 创建RAG系统实例
rag_system = RAGSystem()

# 加载你的文档
rag_system.load_from_texts(["你的文档内容"])

# 执行查询
result = rag_system.query("你的问题")
print(result)
```

## 配置OpenAI API（可选）

如果要使用真实的语言模型而非模拟响应，需要设置OpenAI API密钥：

```bash
export OPENAI_API_KEY='your-api-key'
```

然后在代码中调用 `setup_qa_chain()` 方法来启用真实的LLM。

## 功能特性

- 支持从文本列表或文件加载文档
- 文档自动分块处理
- 基于语义相似性的文档检索
- 支持自定义嵌入模型
- 提供多种查询接口