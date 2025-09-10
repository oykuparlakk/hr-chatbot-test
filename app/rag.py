import time
from dotenv import load_dotenv
import os

# Retriever ve LLM importları
from retriever import get_dense_retriever, get_bm25_retriever, hybrid_retrieve
from llm import get_local_llm, generate_answer
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core import SimpleDirectoryReader

load_dotenv()
TOP_K = int(os.getenv("TOP_K", 5))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "./models/embeddings/bge-reranker-base")


def rag_pipeline(query: str, stream: bool = True):
    """Hybrid retrieval + reranker + LLM cevabı döndürür."""

    # --- Retrieval ---
    dense_retriever = get_dense_retriever(top_k = TOP_K)

    # BM25 retriever için dokümanları yükle
    documents = SimpleDirectoryReader("data").load_data()
    bm25_retriever = get_bm25_retriever(documents, top_k=TOP_K)

    # Hybrid retrieval (dense + bm25 + rrf)
    hybrid_results = hybrid_retrieve(dense_retriever, bm25_retriever, query, top_k=TOP_K)

    # --- Reranker ---
    reranker = FlagEmbeddingReranker(model=RERANKER_MODEL, top_n=TOP_K)
    hybrid_nodes = [
        NodeWithScore(
            node=TextNode(
                text=item["text"],
                metadata={"source": item.get("source", "data")}
            ),
            score=item["score"]
        )
        for item in hybrid_results
    ]
    reranked = reranker.postprocess_nodes(hybrid_nodes, query_str=query)

    # --- LLM ---
    llm = get_local_llm()

    # Reranked sonuçları generate_answer'e uygun hale getir
    sources_for_llm = [
        {
            "text": r.node.get_content(),
            "source": r.node.metadata.get("source", "data")
        }
        for r in reranked
    ]

    t0 = time.perf_counter()
    answer = generate_answer(llm, query, sources_for_llm, stream=stream)
    t1 = time.perf_counter()

    return answer, sources_for_llm, t1 - t0


if __name__ == "__main__":
    query = input("\n💬 Kullanıcı: ")
    answer, sources, latency = rag_pipeline(query, stream=True)

    if answer:  # sadece stream=False olduğunda içerik döner
        print("\n✅ Cevap:", answer)

    print("\n📚 Kaynaklar:")
    for i, s in enumerate(sources, 1):
        print(f"[{i}] ({s['source']}) {s['text'][:80]}...")

    print(f"\n⏱️ Toplam süre: {latency:.2f}s")
