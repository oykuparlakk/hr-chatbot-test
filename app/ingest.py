from pypdf import PdfReader
import os
from sentence_transformers import SentenceTransformer
import chromadb

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    file_path = os.path.join("data", "Fake_Soru_Cevap_Verisi.pdf")
    text = load_pdf(file_path)

    print("PDF başarıyla yüklendi!")
    print("Karakter sayısı:", len(text))

    chunks = chunk_text(text)
    print("Chunk sayısı:", len(chunks))

    # Embedding Modelinin Yüklenmesi
    embed_model_path = "./models/embeddings/e5-small"
    model = SentenceTransformer(embed_model_path)

    embeddings = model.encode(chunks)
    print("Embedding tamamlandı!")
    print("Vektör boyutu:", embeddings.shape)

    # ChromaDB'ye Kaydedilmesi
    client = chromadb.PersistentClient(path="./index/chroma")
    collection = client.get_or_create_collection(name="hr_docs")

    # Her chunk için ID üretimi
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=ids
    )

    print("ChromaDB'ye kaydedildi!")
    print("Toplam kayıt sayısı:", collection.count())
