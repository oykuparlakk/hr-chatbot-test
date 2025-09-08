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

 
def get_bm25_retriever(documents, top_k=10):
  """BM25 retriever (bag-of-words baseline)."""
  # Document → Node çevir
  nodes = [TextNode(text=doc.get_content()) for doc in documents]
  retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)
  return retriever
 
 
def hybrid_retrieve(dense_retriever, bm25_retriever, query, top_k=10):
  """Dense + BM25 retriever sonuçlarını birleştir."""
  dense_results = [(r.node.get_content(), 1.0) for r in dense_retriever.retrieve(query)]
  bm25_results = [(r.node.get_content(), r.score) for r in bm25_retriever.retrieve(query)]

  # Node içeriği bazlı birleştirme, duplicate önleme
  all_results = {text: score for text, score in dense_results + bm25_results}
  return list(all_results.items())[:top_k]
 
 
if __name__ == "__main__":
  # Dokümanları yükle
  documents = SimpleDirectoryReader("data").load_data()

  dense_retriever = get_dense_retriever()
  bm25_retriever = get_bm25_retriever(documents, top_k=top_k)

  # Reranker yükle
  reranker = FlagEmbeddingReranker(model=reranker_model_path, top_n=top_k)

  query = "Fazla mesai ücretleri ne zaman ödenir?"
  print("\n🔹 Query:", query)

  # Dense retriever test
  t0 = time.perf_counter()
  dense_results = dense_retriever.retrieve(query)
  t1 = time.perf_counter()
  print(f"\nDense sonuçları (top_k={top_k}, latency={t1-t0:.3f}s):")
  for r in dense_results:
    print("-", r.node.get_content()[:100], "...")

  # BM25 retriever test
  t0 = time.perf_counter()
  bm25_results = bm25_retriever.retrieve(query)
  t1 = time.perf_counter()
  print(f"\nBM25 sonuçları (top_k={top_k}, latency={t1-t0:.3f}s):")
  for r in bm25_results:
    print("-", r.node.get_content()[:100], "...")

  # Hybrid test
  t0 = time.perf_counter()
  hybrid_results = hybrid_retrieve(dense_retriever, bm25_retriever, query, top_k=top_k)
  t1 = time.perf_counter()
  print(f"\nHybrid sonuçları (top_k={top_k}, latency={t1-t0:.3f}s):")
  for text, score in hybrid_results:
    print("-", text[:100], "...")

  # Hybrid + Reranker test
  hybrid_nodes = [
    NodeWithScore(node=TextNode(text=text), score=score)
    for text, score in hybrid_results
  ]
  reranked_hybrid_results = reranker.postprocess_nodes(hybrid_nodes, query_str=query)

  print(f"\nHybrid + Reranker sonrası sonuçlar (top_k={top_k}):")
  for r in reranked_hybrid_results:
    print("-", r.node.get_content()[:100], "...")