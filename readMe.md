# HR-Chatbot-Test

RAG (Retrieval-Augmented Generation) tabanlı basit İnsan Kaynakları Chatbot projesi.

---

## 📖 Proje Hakkında

Bu proje, **dokümanlardan bilgi alıp** kullanıcı sorularına cevap veren bir chatbot geliştirmek için hazırlandı.  
Çalışma adımları:

1. **Ingestion (Hazırlık)** → Dokümanları alır, parçalara böler, embedding’e çevirir ve ChromaDB’ye kaydeder.  
2. **Retrieval (Sorgu)** → Kullanıcı sorusunu embedding’e çevirir, Vector DB’den en yakın parçaları bulur, reranker ile sıralar.  
3. **LLM + RAG (Cevaplama)** → En iyi parçaları LLM’e verir ve anlamlı yanıt üretir.  
4. **UI** → Streamlit ile web arayüzünden soru-cevap yapılabilir.

---

## 📂 Dosya Yapısı

- `models/embeddings/e5-small/` → Embedding modeli (doküman ve soruları sayıya çevirir)  
- `models/embeddings/bge-reranker-base/` → Reranker modeli (embedding sonuçlarını yeniden sıralar)  
- `data/` → Kaynak dokümanlar  
- `index/` → ChromaDB veritabanı  
- `app/ingest.py` → Ingestion işlemleri  
- `app/retriever.py` → Retrieval & Reranker işlemleri  
- `app/llm.py` + `app/rag.py` → LLM çağrısı ve RAG cevabı  
- `app/ui.py` → Streamlit tabanlı arayüz

---

## ⚙️ Kurulum

1. Gerekli paketleri yükleyin
   ``` pip3 install llama-index chromadb FlagEmbedding llama-cpp-python streamlit ```

## 🚀 Çalıştırma
  ``` streamlit run app/ui.py ```

## 🛠️ Hatalar ve Çözümleri

1. from llama_index.llms.llama_cpp import LlamaCPP hatası:
    ``` pip install llama-index-llms-llama-cpp  ```

2. ModuleNotFoundError: No module named 'llama_index.retrievers':
    ``` pip3 install llama-index-retrievers-bm25  ```

3. ModuleNotFoundError: No module named 'llama_index.postprocessor':
    ``` pip3 install llama-index-postprocessor-flag-embedding-reranker ```