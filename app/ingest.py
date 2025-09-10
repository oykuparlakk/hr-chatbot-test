from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import TokenTextSplitter
import chromadb
from dotenv import load_dotenv
import os


def run_ingest(chunk_size: int, chunk_overlap: int):
    """Verilen chunk parametreleriyle ingestion yapar ve index döner."""

    # .env yükle
    load_dotenv()
    embed_model_path = os.getenv("EMBEDDING_MODEL", "./models/embeddings/e5-small")
    collection_name = os.getenv("CHROMA_COLLECTION", "hr_docs")

    print(f"\n📂 Data klasöründen dokümanlar yükleniyor...")
    documents = SimpleDirectoryReader("data").load_data()
    for d in documents:
        if "source" not in d.metadata:
            file_name = d.metadata.get("file_name") or os.path.basename(d.doc_id)
            d.metadata["source"] = file_name
    print(f" 🔹 Doküman sayısı: {len(documents)}")

    # Embedding modeli
    embed_model = HuggingFaceEmbedding(model_name=embed_model_path)

    # ChromaDB client
    chroma_client = chromadb.PersistentClient(path="./index/chroma")

    # Koleksiyonu sıfırla
    try:
        chroma_client.delete_collection(collection_name)
        print(f" 🔄 Eski '{collection_name}' koleksiyonu silindi.")
    except Exception:
        print(f" ℹ️ Önceki '{collection_name}' koleksiyonu bulunamadı, yeni oluşturulacak.")

    collection = chroma_client.get_or_create_collection(collection_name)

    # Vector store ve storage context
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Chunk splitter
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Index oluşturma
    print(f"\n⚙️ Index oluşturuluyor... (chunk_size={chunk_size}, overlap={chunk_overlap})")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
    )
    print("✅ LlamaIndex ile ingestion tamamlandı!")
    print("🔢 Toplam kayıt sayısı:", collection.count())

    return index


if __name__ == "__main__":
    load_dotenv()
    chunk_size = int(os.getenv("CHUNK_SIZE", 500))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 50))
    run_ingest(chunk_size, chunk_overlap) 