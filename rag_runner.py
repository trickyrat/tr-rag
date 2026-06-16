import json
import sys
from src.evaluation import RAGEvaluator
from src.rag import RetrievalOptimization, DocumentChunker, VectorStoreBuilder
import logging
from config import RAGConfig, DEFAULT_CONFIG


logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


logger.info("🚀 Initializing RAG system...")

config: RAGConfig = DEFAULT_CONFIG

logger.info("Loading and chunking docuements")

chunker = DocumentChunker(config.knowledge_base_path)
chunker.load_documents()
chunks = chunker.chunk_documents()

# Export chunk mapping (content_hash + chunk_id) for testset sync
mapping_path = "test/evaluation/chunk_mapping.jsonl"
with open(mapping_path, "w", encoding="utf-8") as f:
    for chunk in chunks:
        record = {
            "chunk_id": chunk.metadata.get("chunk_id"),
            "content_hash": chunk.metadata.get("content_hash"),
            "source": chunk.metadata.get("source"),
            "doc_name": chunk.metadata.get("doc_name"),
            "primary_category": chunk.metadata.get("primary_category"),
            "sub_category": chunk.metadata.get("sub_category"),
            "content_preview": chunk.page_content[:500],
            "chunk_size": len(chunk.page_content),
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
logger.info(f"Exported {len(chunks)} chunk mappings to {mapping_path}")


builder = VectorStoreBuilder(
    model_name=config.embedding_model,
    index_save_path=config.index_save_path,
)
builder.index_documents(chunks)
logger.info(f"Indexing {len(chunks)} chunks...")

vectorstore = builder.get_vectorstore()
retriever = RetrievalOptimization(vectorstore=vectorstore, chunks=chunks)
logger.info("Knowledge base ready")


def retriev_fn(query: str, top_k: int = 5):
    return retriever.hybrid_search(query, top_k=top_k)


evaluator = RAGEvaluator(config.evaluation_testset_path)
results = evaluator.evaluate(
    retrieve_fn=retriev_fn, top_k=config.top_k, k_list=[1, 3, 5], per_query_results=False
)

agg = results["aggregate"]
logger.info("=" * 50)
logger.info("📊 Evaluation Results")
logger.info("=" * 50)
logger.info(f"  Total queries:        {agg.get('total_queries', 'N/A')}")
logger.info(f"  Hit@1:                {agg.get('hit@1', 0):.4f}")
logger.info(f"  Hit@3:                {agg.get('hit@3', 0):.4f}")
logger.info(f"  Hit@5:                {agg.get('hit@5', 0):.4f}")
logger.info(f"  MRR:                  {agg.get('mrr', 0):.4f}")
logger.info(f"  Precision@1:          {agg.get('precision@1', 0):.4f}")
logger.info(f"  Precision@3:          {agg.get('precision@3', 0):.4f}")
logger.info(f"  Precision@5:          {agg.get('precision@5', 0):.4f}")
logger.info(f"  Recall@1:             {agg.get('recall@1', 0):.4f}")
logger.info(f"  Recall@3:             {agg.get('recall@3', 0):.4f}")
logger.info(f"  Recall@5:             {agg.get('recall@5', 0):.4f}")
logger.info(f"  NDCG@1:               {agg.get('ndcg@1', 0):.4f}")
logger.info(f"  NDCG@3:               {agg.get('ndcg@3', 0):.4f}")
logger.info(f"  NDCG@5:               {agg.get('ndcg@5', 0):.4f}")
logger.info("=" * 50)
