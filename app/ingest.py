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
  pdf_folder = "data"
  pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

  # Embedding Modeli ve ChromaDB client sadece bir kez yükleniyor
  embed_model_path = "./models/embeddings/e5-small"
  model = SentenceTransformer(embed_model_path)
  client = chromadb.PersistentClient(path="./index/chroma")
  collection = client.get_or_create_collection(name="hr_docs")

  for pdf_file in pdf_files:
      file_path = os.path.join(pdf_folder, pdf_file)
      print(f"\nPDF yükleniyor: {pdf_file}")
      text = load_pdf(file_path)
      print("Karakter sayısı:", len(text))

      chunks = chunk_text(text)
      print("Chunk sayısı:", len(chunks))

      embeddings = model.encode(chunks)
      print("Embedding tamamlandı!")
      print("Vektör boyutu:", embeddings.shape)

      # Her chunk için ID üretimi, dosya adını da ekliyoruz
      base_name = os.path.splitext(pdf_file)[0]
      ids = [f"{base_name}_chunk_{i}" for i in range(len(chunks))]

      collection.add(
          documents=chunks,
          embeddings=embeddings.tolist(),
          ids=ids
      )

      print(f"{pdf_file} ChromaDB'ye kaydedildi!")

  print("\nToplam kayıt sayısı:", collection.count())