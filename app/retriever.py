import os
import time
from dotenv import load_dotenv
import chromadb
import numpy as np

# LlamaIndex importları
from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.schema import TextNode, NodeWithScore
 
# .env yükle
load_dotenv()
embed_model_path = os.getenv("EMBEDDING_MODEL", "./models/embeddings/e5-small")
reranker_model_path = os.getenv("RERANKER_MODEL", "./models/embeddings/bge-reranker-base")
collection_name = os.getenv("CHROMA_COLLECTION", "hr_docs")
top_k = int(os.getenv("TOP_K", 10))
rrf_k = int(os.getenv("RRF_K", 60)) 
 
def get_dense_retriever():
  """Dense retriever (Chroma + local embeddings)."""
  embed_model = HuggingFaceEmbedding(model_name=embed_model_path)
  chroma_client = chromadb.PersistentClient(path="./index/chroma")
  collection = chroma_client.get_or_create_collection(collection_name)
  vector_store = ChromaVectorStore(chroma_collection=collection)
  storage_context = StorageContext.from_defaults(vector_store=vector_store)

  index = VectorStoreIndex.from_vector_store(
      vector_store,
      storage_context=storage_context,
      embed_model=embed_model
  )
  return index.as_retriever(similarity_top_k=top_k)

 
def get_bm25_retriever(documents, top_k):
  """BM25 retriever (bag-of-words baseline)."""
  # Document → Node çevir
  nodes = [TextNode(text=doc.get_content()) for doc in documents]
  retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)
  return retriever
 
 
def hybrid_retrieve(dense_retriever, bm25_retriever, query, top_k, rrf_k=rrf_k):
    """Hybrid retrieval with Reciprocal Rank Fusion (RRF)."""
    dense_results = dense_retriever.retrieve(query)
    bm25_results = bm25_retriever.retrieve(query)

    scores = {}

    def add_results(results, weight=1.0):
        for rank, r in enumerate(results):
            text = r.node.get_content()
            meta = r.node.metadata.get("file_name", "")  # kaynak dosya
            key = (text, meta)
            scores[key] = scores.get(key, 0) + weight * (1 / (rrf_k + rank + 1))

    add_results(dense_results, weight=1.0)
    add_results(bm25_results, weight=1.0)

    fused_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Geriye (text, source, score) dönelim
    return [{"text": text, "source": meta, "score": score} 
            for (text, meta), score in fused_results[:top_k]]



if __name__ == "__main__":
    # Dokümanları yükle
    documents = SimpleDirectoryReader("data").load_data()

    dense_retriever = get_dense_retriever()
    bm25_retriever = get_bm25_retriever(documents, top_k=top_k)

    # Reranker yükle
    reranker = FlagEmbeddingReranker(model=reranker_model_path, top_n=top_k)

    query = "Fazla mesai ücretleri ne zaman ödenir?"
    print("\n🔹 Query:", query)

    # Hybrid RRF test
    t0 = time.perf_counter()
    hybrid_results = hybrid_retrieve(dense_retriever, bm25_retriever, query, top_k=top_k)
    t1 = time.perf_counter()

    print(f"\nHybrid (RRF) sonuçları (top_k={top_k}, latency={t1-t0:.3f}s):")
    for item in hybrid_results:
        print(f"- [{item['score']:.4f}] ({item['source']}) {item['text'][:100]} ...")

    # Hybrid + Reranker test
    hybrid_nodes = [
        NodeWithScore(node=TextNode(text=item["text"]), score=item["score"])
        for item in hybrid_results
    ]
    reranked_hybrid_results = reranker.postprocess_nodes(hybrid_nodes, query_str=query)

    print(f"\nHybrid + Reranker sonrası sonuçlar (top_k={top_k}):")
    for r in reranked_hybrid_results:
        print("-", r.node.get_content()[:100], "...")
