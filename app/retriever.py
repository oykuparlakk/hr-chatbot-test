import chromadb
from sentence_transformers import SentenceTransformer

# Embedding modelimizin yüklendiği yer
embed_model_path = "./models/embeddings/e5-small"
model = SentenceTransformer(embed_model_path)

# ChromaDB'ye bağlandığımız yer
client = chromadb.PersistentClient(path="./index/chroma")
collection = client.get_collection(name="hr_docs")

def retrieve(query, top_k=3):
    # Soruyu embedding'e çevir
    query_embedding = model.encode([query])[0]

    #ChromaDB'den en yakın chunk'ları getir
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    return results

if __name__ == "__main__":
    # örnek soru
    question = "Maaşlar hangi tarihte yatırılıyor?"
    results = retrieve(question, top_k=3)

    print("🔎 Soru:", question)
    print("📌 En yakın chunklar:")
    for i, doc in enumerate(results["documents"][0]):
        print(f"{i+1}. {doc}")
