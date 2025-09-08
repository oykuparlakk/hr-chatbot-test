############

# pip3 install llama-index
# pip3 install llama-index-embeddings-huggingface
# pip install chromadb llama-index sentence-transformers

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Embedding modeli
# embed_model = HuggingFaceEmbedding(
#     model_name="intfloat/e5-small",   # HF Hub modeli
#     trust_remote_code=True
# )

embed_model_path = "./models/embeddings/e5-small"
embed_model = HuggingFaceEmbedding(
    model_name=embed_model_path,
    trust_remote_code=True
)


# Dökümanları oku
documents = SimpleDirectoryReader("data").load_data()

# Chroma istemcisi oluştur
chroma_client = chromadb.Client()

# Chroma koleksiyonu oluştur
chroma_collection = chroma_client.get_or_create_collection("hr_docs")

# ChromaVectorStore ile vektör veritabanı oluştur
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)


# Index oluştur (chunking + embedding) ve ChromaDB’ye bağla
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
    vector_store=vector_store,
    storage_context=storage_context
)

# DB’yi diske kaydet
vector_store.persist(persist_path="./index/chroma")
print("Yüklenen koleksiyondaki doküman sayısı:", len(vector_store._collection.get()["ids"]))

print("ChromaDB'ye kaydetme tamamlandı ✅")

# Embedding'lere erişim
results = chroma_collection.get(include=["documents", "metadatas"])
for i, chunk_text in enumerate(results["documents"]):
    print("Chunk:", chunk_text[:200])  # ilk 200 karakter
    print("Metadata:", results["metadatas"][i])
    print("------------------")