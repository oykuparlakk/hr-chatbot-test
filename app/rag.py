import time
from dotenv import load_dotenv

# Retriever ve LLM importları
from app.retriever import get_dense_retriever, get_bm25_retriever, hybrid_retrieve
from app.llm import get_local_llm
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.llms import ChatMessage

import os

load_dotenv()
TOP_K = int(os.getenv("TOP_K", 5))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "./models/embeddings/bge-reranker-base")

def rag_pipeline(query: str, stream: bool = False):
    """Hybrid retrieval + reranker + LLM cevabı döndürür."""
    # --- Retrieval ---
    documents = SimpleDirectoryReader("data").load_data()
    dense_retriever = get_dense_retriever()
    bm25_retriever = get_bm25_retriever(documents, top_k=TOP_K)

    # Hybrid retrieval
    hybrid_results = hybrid_retrieve(dense_retriever, bm25_retriever, query, top_k=TOP_K)

    # Reranker
    reranker = FlagEmbeddingReranker(model=RERANKER_MODEL, top_n=TOP_K)
    hybrid_nodes = [
        NodeWithScore(node=TextNode(text=text), score=score)
        for text, score in hybrid_results
    ]
    reranked = reranker.postprocess_nodes(hybrid_nodes, query_str=query)

    # --- LLM ---
    llm = get_local_llm()
    context = "\n".join([r.node.get_content() for r in reranked])

    messages = [
      ChatMessage(role="system", content=(
          "Sen bir İK asistanısın. "
          "Yalnızca aşağıdaki dokümanlardan alıntı yaparak cevap ver. "
          "Eğer cevap dokümanlarda yoksa 'Bilmiyorum' de. "
          "Asla yeni bilgi uydurma."
      )),
      ChatMessage(role="user", content=f"Soru: {query}\n\nDokümanlar:\n{context}")
    ]

    if stream:
        print("\n✅ Cevap (stream):\n")
        t0 = time.perf_counter()
        for chunk in llm.stream_chat(messages):
            print(chunk.delta, end="", flush=True)
        t1 = time.perf_counter()
        print("\n===================")
        return None, reranked, t1 - t0
    else:
        t0 = time.perf_counter()
        response = llm.chat(messages)
        t1 = time.perf_counter()
        return response.message.content, reranked, t1 - t0


if __name__ == "__main__":
    query = "istifa prosedürü nedir?"
    # stream=True dersek token token çıktıyı göreceğiz
    answer, sources, latency = rag_pipeline(query, stream=True)

    if answer:  # sadece stream=False iken olacak
        print("\n✅ Cevap:", answer)

    print("\n📚 Kaynaklar:")
    for r in sources:
        print("-", r.node.get_content()[:100], "...")

    print(f"\n⏱️ LLM latency: {latency:.2f}s")