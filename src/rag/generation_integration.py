import logging
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

BASIC_PROMPT = ChatPromptTemplate.from_template("""
You are a knwnledge technical assistant. Answer the user's question based on the provided reference materials.

User question: {question}

Reference materials:
{context}

Provide a clear, accurate, and practical answer. If the information is insufficient, honestly state what is missing.

Answer:""")

STEP_BY_STEP_PROMPT = ChatPromptTemplate.from_template("""
You are a senior techenical mentor. Based on the reference materials, provide a detailed step-by-step explanation for the user.

User question: {question}

Reference materials:
{context}

Provide origanize your answer flexibly, suggested structure (adjust based on actual content):

## Overview
[Brief introduction of the concept or topic]

## Key Concepts
[List core concepts, components, or prerequisites]

## Step-by-Step Guide
[Detailed step-by-step explanation, with code examples where applicable]

## Tips & Best Practices
[Include only if there are genuinely useful tips from the reference materials]

Notes:
- Adjust structure based on the actual content.
- Do not pad with irrelevant content
- Focus on practical and actionability
- Include code examples where helpful

Answer:""")

LIST_PROMPT = ChatPromptTemplate.from_template("""
You are a technical assistant. Based on the reference materials, provide a concise list or summary answering the user's question.

User question: {question}

Reference materials:
{context}

Provide a concise, well-organized answer. Use bullet points or numbered lists where appropriate.

Answer:""")

REWRITE_PROMPT = PromptTemplate(
    template="""
You are an intelligent query analysis assistant. Analyze the user's query and determine if it needs rewriting to improve technical documentation search results."

Original query: {query}

Analysis rules:
1. **Specific and clear queries** (return original query directly):
   - Contains specific technology names: e.g., "How to use LangChain RAG", "SQLAlchemy session management"
   - Clear technical questions: e.g., "How to implement vector search", "Python text splitting methods"
   - Specific API usage: e.g., "ChromaDB collection creation", "HuggingFace embeddings setup"

2. **Vague or unclear queries** (need rewriting):
   - Too broad: e.g., "database", "AI", "Python"
   - Lacks specifics: e.g., "how to code", "best practices"
   - Colloquial expressions: e.g., "how do I do this thing"

Rewriting principles:
- Keep the original meaning intact
- Add relevant technical terminology
- Make it more specific and searchable
- Keep it concise

Examples:
- "database" -> "vector database concepts and implementation"
- "RAG" -> "RAG system architecture and retrieval pipeline"
- "How to use LangChain RAG" -> "How to use LangChain RAG" (keep original)
- "SQLAlchemy session" -> "SQLAlchemy session" (keep original)

Output the final query (return original if no rewrite needed):""",
    input_variables=["query"],
)

ROUTER_PROMPT = ChatPromptTemplate.from_template("""
Classify the user's question into one of the following four types:

1. 'concept' - User wants to understand a concept or theory
    Examples: What is RAG, explain vector databases, how does embedding work

2. 'how-to' - User wants step-by-step instructions or implementation guidance
    Examples: How to build a RAG system, how to use LangChain, implement vector search

3. 'reference' - User wants specific API references, code snippets, or configurations
    Examples: SLQAlchemy session API, ChromaDB collection methods, Python text splitter parameters

4. 'general' - Other general questions
    Examples: Best practices comparison, pros and cons, howto, reference, or general

User question: {query}

Classification:""")


class GenerationIntegration:
    """生成集成模块 - 负责LLM集成和回答生成"""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.setup_llm()

    def setup_llm(self):
        logger.info(f"Initializing LLM: {self.model_name}")

        if not self.api_key:
            raise ValueError(
                "API key is required. Set API_KEY environment variable or pass it in RAGConfig."
            )

        try:
            from langchain_deepseek import ChatDeepSeek
            self.llm = ChatDeepSeek(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.api_key,
            )
            logger.info("LLM initialized via langchain_deepseek.ChatDeepSeek")
        except ImportError:
            logger.warning(
                "langchain_deepseek not installed; falling back to stub. "
                "Install with: pip install langchain-deepseek"
            )
            self.llm = None

        parser = StrOutputParser()

        if self.llm is not None:
            self.basic_chain = BASIC_PROMPT | self.llm | parser
            self.step_by_step_chain = STEP_BY_STEP_PROMPT | self.llm | parser
            self.list_chain = LIST_PROMPT | self.llm | parser
            self.rewrite_chain = REWRITE_PROMPT | self.llm | parser
            self.router_chain = ROUTER_PROMPT | self.llm | parser
        else:
            logger.warning("LLM not initialized; generation chains are None")

    def generate_basic_answer(self, query: str, context_docs: List[Document]) -> str:
        context = self._build_context(context_docs)
        return self.basic_chain.invoke({"question": query, "context": context})

    def generate_step_by_step_answer(
        self, query: str, context_docs: List[Document]
    ) -> str:
        context = self._build_context(context_docs)
        return self.step_by_step_chain.invoke({"question": query, "context": context})

    def generate_list_answer(self, query: str, context_docs: List[Document]) -> str:
        context = self._build_context(context_docs)
        return self.list_chain.invoke({"question": query, "context": context})

    def query_rewrite(self, query: str) -> str:
        response = self.rewrite_chain.invoke({"query": query}).strip()
        if response != query:
            logger.info(f"Query rewritten: `{query}` -> '{response}'")
        else:
            logger.info(f"Query unchanged: '{query}'")
        return response

    def query_router(self, query: str) -> str:
        result = self.router_chain.invoke({"query": query}).strip().lower()
        valid_types = ["concept", "howto", "reference", "general"]
        return result if result in valid_types else "general"

    def generate_basic_answer_stream(self, query: str, context_docs: List[Document]):
        context = self._build_context(context_docs)
        for chunk in self.basic_chain.stream({"question": query, "context": context}):
            yield chunk

    def generate_step_by_step_answer_stream(
        self, query: str, context_docs: List[Document]
    ):
        context = self._build_context(context_docs)
        for chunk in self.step_by_step_chain.stream(
            {"question": query, "context": context}
        ):
            yield chunk

    def _build_context(
        self, docs: List[Document], max_tokens: Optional[int] = None
    ) -> str:
        if not docs:
            return "No relevant information available."

        if max_tokens is None:
            max_tokens = int(self.max_tokens * 0.7)

        max_chars = max_tokens * 4
        context_parts = []
        current_length = 0

        for i, doc in enumerate(docs, 1):
            metadata_info = f"[Document {i}]"
            if "doc_name" in doc.metadata:
                metadata_info += f" {doc.metadata['doc_name']}"
            if "primary_category" in doc.metadata and doc.metadata["primary_category"]:
                metadata_info += f" | Category: {doc.metadata['primary_category']}"
            if "sub_category" in doc.metadata and doc.metadata["sub_category"]:
                metadata_info += f" > {doc.metadata['sub_category']}"

            doc_text = f"{metadata_info}\n{doc.page_content}\n"

            if current_length + len(doc_text) > max_chars:
                remaining = max_chars - current_length
                if remaining > 200:
                    doc_text = doc_text[:remaining] + "\n[...truncated]"
                    context_parts.append(doc_text)
                    logger.warning(
                        f"Context truncated at document {i}/{len(docs)}, "
                        f"budget: {max_chars} characters (~{max_tokens} tokens)."
                    )
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        return ("\n" + "=" * 50 + "\n").join(context_parts)
