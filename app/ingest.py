from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import TokenTextSplitter
import chromadb
from dotenv import load_dotenv
import os


if __name__ == "__main__":
    # .env dosyasını yükle
    load_dotenv()

    # Parametreleri .env'den oku
    embed_model_path = os.getenv("EMBEDDING_MODEL", "./models/embeddings/e5-small")
    collection_name = os.getenv("CHROMA_COLLECTION", "hr_docs")
    chunk_size = int(os.getenv("CHUNK_SIZE", 500))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 50))

    print("\nData klasöründen dokümanlar yükleniyor...")
    documents = SimpleDirectoryReader("data").load_data()
    for d in documents:
      if "source" not in d.metadata:
        # Eğer SimpleDirectoryReader "file_name" vermediyse fallback olarak basename
        file_name = d.metadata.get("file_name") or os.path.basename(d.doc_id)
        d.metadata["source"] = file_name
    print(f" Doküman sayısı: {len(documents)}")
    # 1. Embedding modeli
    print("\n Embedding modeli yükleniyor...")
    embed_model = HuggingFaceEmbedding(model_name=embed_model_path)
    print(f" Embedding modeli hazır! ({embed_model_path})")

    # 2. ChromaDB client (lokal, dosya tabanlı)
    print("\n ChromaDB başlatılıyor...")
    chroma_client = chromadb.PersistentClient(path="./index/chroma")

    # Önce koleksiyon varsa sil sonra sıfırdan oluştur
    try:
        chroma_client.delete_collection(collection_name)
        print(f" Eski '{collection_name}' koleksiyonu silindi.")
    except Exception:
        print(f"Önceki '{collection_name}' koleksiyonu bulunamadı, yeni oluşturulacak.")

    collection = chroma_client.get_or_create_collection(collection_name)

    # Vector store ve storage context
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    print("ChromaDB client hazır!")

    # 3. Chunk splitter 
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 4. Index oluşturma (chunking + embedding + kayıt)
    print(f"\nIndex oluşturuluyor... (chunk_size={chunk_size}, overlap={chunk_overlap})")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter]
    )
    print("LlamaIndex ile ingestion tamamlandı!")

    print("Toplam kayıt sayısı:", collection.count())


    print("Toplam kayıt sayısı:", index)
