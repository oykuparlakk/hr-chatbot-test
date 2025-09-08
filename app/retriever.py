import os
import time
from dotenv import load_dotenv
import chromadb

# LlamaIndex importları
from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever

# .env yükle
load_dotenv()
embed_model_path = os.getenv("EMBEDDING_MODEL", "./models/embeddings/e5-small")
collection_name = os.getenv("CHROMA_COLLECTION", "hr_docs")


def get_dense_retriever():
    """Dense retriever (Chroma + embeddings)."""
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
    return index.as_retriever(similarity_top_k=10)


def get_bm25_retriever(documents):
    """BM25 retriever (keyword-based)."""
    # ✅ dokümanları doğrudan from_defaults'a veriyoruz
    return BM25Retriever.from_defaults(documents=documents, similarity_top_k=10)


def hybrid_retrieve(dense_retriever, bm25_retriever, query, top_k=10):
    """Dense + BM25 sonuçlarını birleştir."""
    dense_results = dense_retriever.retrieve(query)
    bm25_results = bm25_retriever.retrieve(query)

    all_results = {n.node_id: n for n in dense_results + bm25_results}
    return list(all_results.values())[:top_k]


if __name__ == "__main__":
    documents = SimpleDirectoryReader("data").load_data()

    dense_retriever = get_dense_retriever()
    bm25_retriever = get_bm25_retriever(documents)

    query = "Fazla mesai ücretleri ne zaman ödenir?"
    print("\n🔹 Query:", query)

    # Dense test
    t0 = time.perf_counter()
    dense_results = dense_retriever.retrieve(query)
    t1 = time.perf_counter()
    print(f"\nDense sonuçları ({len(dense_results)} adet, latency={t1-t0:.3f}s):")
    for r in dense_results:
        print("-", r.node.get_content()[:100], "...")

    # BM25 test
    t0 = time.perf_counter()
    bm25_results = bm25_retriever.retrieve(query)
    t1 = time.perf_counter()
    print(f"\nBM25 sonuçları ({len(bm25_results)} adet, latency={t1-t0:.3f}s):")
    for r in bm25_results:
        print("-", r.node.get_content()[:100], "...")

    # Hybrid test
    t0 = time.perf_counter()
    hybrid_results = hybrid_retrieve(dense_retriever, bm25_retriever, query)
    t1 = time.perf_counter()
    print(f"\nHybrid sonuçları ({len(hybrid_results)} adet, latency={t1-t0:.3f}s):")
    for r in hybrid_results:
        print("-", r.node.get_content()[:100], "...")
