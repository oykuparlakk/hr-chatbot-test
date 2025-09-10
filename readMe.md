hr-chatbot-test/models/embeddings/e5-small/ -> Embedding modeli dokümanı ve soruyu sayıya çeviren motor.
hr-chatbot-test/models/embeddings/bge-reranker-base/ -> Reranker, embedding’den gelen sonuçları yeniden sıralar, en alakalıyı seçer.



#Ingestion (hazırlık aşaması)

Dokümanları alıyoruz (data/).

Parçalara bölüyoruz (chunking).

Embedding modelleriyle sayılara çeviriyoruz.

Çıkan embedding’leri ChromaDB’ye kaydediyoruz (index/).
Bu kısım → app/ingest.py

#Retrieval (sorgu aşaması)

Kullanıcı bir soru soruyor.

Soruyu embedding’e çeviriyoruz.

Vector DB’den en yakın parçaları buluyoruz.

#Reranker varsa sıralıyoruz.
Bu kısım → app/retriever.py

#LLM + RAG (cevaplama aşaması)

En iyi parçaları LLM’e veriyoruz.

LLM parçaları kullanarak düzgün bir cümle döndürüyor.
Bu kısım → app/llm.py + app/rag.py

UI → app/ui.py


pip3 install pypdf llama-index chromadb sentence-transformers FlagEmbedding llama-cpp-python streamlit

source venv/bin/activate

llmdeki from llama_index.llms.llama_cpp import LlamaCPP hatası bu komutla çözüldü ->  pip install llama-index-llms-llama-cpp

