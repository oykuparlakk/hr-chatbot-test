from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.llms import ChatMessage
from dotenv import load_dotenv
import os
import time
import traceback

# .env yükle
load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL", "./models/llm/mistral-7b-instruct-q4_k_m.gguf")


def get_local_llm():
    """Yerel LLM instance döner (llama.cpp)."""
    llm = LlamaCPP(
        model_path=LLM_MODEL,
        temperature=0.1,
        max_new_tokens=256,         # daha uzun cevaplar için
        context_window=4096,       # model metadata max
        model_kwargs={
            "n_threads": os.cpu_count(),  # tüm CPU çekirdekleri
            "n_batch": 128,               # dengeli batch
            "n_gpu_layers": 33            # GPU'ya offload
        }
    )
    return llm


def generate_answer(llm, query: str, sources: list, stream: bool = True) -> str:
    """Retriever’dan gelen pasajları kullanarak cevap üretir (streaming loglu)."""

    # Kaynakları sınırla ve truncate et
    limited_sources = sources[:5]
    context_str = "\n\n".join([
        f"[Kaynak Dosya: {doc['source']}] {doc['text'][:500]}"
        if isinstance(doc, dict) else f"{str(doc)[:500]}"
        for doc in limited_sources
    ])

    system_prompt = (
        "Sen bir İK asistanısın. "
        "Sadece aşağıdaki kaynaklara dayanarak cevap ver. "
        "Her iddiadan sonra mutlaka (Kaynak Dosya: <dosya_adı>) ekle. "
        "Dosya adı bu formatta yukarıda verilmiştir. "
        "Eğer cevap dokümanlarda yoksa sadece 'Bilmiyorum' yaz. "
        "Cevabını kısa ve öz tut."
    )
    user_prompt = f"Soru: {query}\n\nKaynaklar:\n{context_str}"

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt),
    ]

    if stream:
        print("\n--- Streaming Başladı ---\n")
        chunks = []
        try:
            for chunk in llm.stream_chat(messages):
                delta = chunk.delta or ""
                print(delta, end="", flush=True)
                chunks.append(delta)
            print("\n\n--- Streaming Bitti ---\n")
            return "".join(chunks)
        except Exception:
            print("\n❌ Hata oluştu:\n", traceback.format_exc())
            return ""
    else:
        try:
            response = llm.chat(messages)
            return response.message.content
        except Exception:
            print("\n❌ Hata oluştu:\n", traceback.format_exc())
            return ""



if __name__ == "__main__":
    llm = get_local_llm()

    query = "Yıllık izin kaç parçaya bölünebilir?"
    sources = [
        {"text": "Yıllık izin işçi ile işverenin anlaşması halinde bölünebilir.", "source": "izin_politikasi.pdf"},
        {"text": "Yıllık izin en az 10 gün kesintisiz kullanılmalıdır.", "source": "iş_kanunu.docx"}
    ]

    start = time.time()
    answer = generate_answer(llm, query, sources, stream=True)
    elapsed = time.time() - start

    print("\n===================")
    print("Final Answer:\n", answer)
    print(f"\n⏱️ Süre: {elapsed:.2f} saniye")
    print("===================\n")
