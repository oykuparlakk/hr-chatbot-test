import os
import chromadb
from dotenv import load_dotenv

# LlamaIndex importları
from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

# .env yükle
load_dotenv()
embed_model_path = os.getenv("EMBEDDING_MODEL", "./models/embeddings/e5-small")
reranker_model_path = os.getenv("RERANKER_MODEL", "./models/embeddings/bge-reranker-base")
collection_name = os.getenv("CHROMA_COLLECTION", "hr_docs")


def get_dense_retriever(top_k: int):
    """Dense retriever (Chroma + local embeddings)."""
    embed_model = HuggingFaceEmbedding(model_name=embed_model_path)
    chroma_client = chromadb.PersistentClient(path="./index/chroma")
    collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    return index.as_retriever(similarity_top_k=top_k)


def get_bm25_retriever(documents, top_k: int):
    """BM25 retriever (bag-of-words baseline)."""
    nodes = [
        TextNode(
            text=doc.get_content(),
            metadata={
                "file_name": doc.metadata.get("file_name", "unknown"),
                "section": doc.metadata.get("section", None),
            },
        )
        for doc in documents
    ]
    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)
    return retriever


def hybrid_retrieve(dense_retriever, bm25_retriever, query: str, top_k: int, rrf_k: int = 60, use_reranker: bool = True):
    """Hybrid retrieval with Reciprocal Rank Fusion (RRF) + optional reranker."""
    dense_results = dense_retriever.retrieve(query)
    bm25_results = bm25_retriever.retrieve(query)

    scores = {}

    def add_results(results, weight=1.0):
        for rank, r in enumerate(results):
            text = r.node.get_content()
            meta = r.node.metadata.get("file_name", "")
            key = (text, meta)
            scores[key] = scores.get(key, 0) + weight * (1 / (rrf_k + rank + 1))

    add_results(dense_results, weight=1.0)
    add_results(bm25_results, weight=1.0)

    fused_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    hybrid_results = [
        {"text": text, "source": meta, "score": score}
        for (text, meta), score in fused_results[:top_k]
    ]

    if not use_reranker:
        return hybrid_results

    # Reranker uygula
    reranker = FlagEmbeddingReranker(model=reranker_model_path, top_n=top_k)
    hybrid_nodes = [
        NodeWithScore(
            node=TextNode(text=item["text"], metadata={"source": item["source"]}),
            score=item["score"],
        )
        for item in hybrid_results
    ]
    reranked = reranker.postprocess_nodes(hybrid_nodes, query_str=query)

    return [
        {"text": r.node.get_content(), "source": r.node.metadata.get("source"), "score": r.score}
        for r in reranked
    ]


if __name__ == "__main__":
    documents = SimpleDirectoryReader("data").load_data()
    dense = get_dense_retriever(top_k=10)
    bm25 = get_bm25_retriever(documents, top_k=10)

    query = "Fazla mesai ücretleri ne zaman ödenir?"
    results = hybrid_retrieve(dense, bm25, query, top_k=10, use_reranker=True)
    print("\nSonuçlar:")
    for r in results:
        print(f"- ({r['source']}) {r['text'][:80]} ...")
 