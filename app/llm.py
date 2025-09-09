from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.llms import ChatMessage
from dotenv import load_dotenv
import os

load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL", "./models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf")

def get_local_llm():
    llm = LlamaCPP(
        model_path=LLM_MODEL,
        temperature=0.1,
        max_new_tokens=256,       # çıkış uzunluğu
        context_window=5000,      # daha küçük → hız artar
        model_kwargs={
            "n_threads": os.cpu_count() // 2,
            "n_batch": 256,       # 512 yerine 256 → stabil
            "n_gpu_layers": 33    # tüm katman GPU’ya
        }
    )
    return llm

if __name__ == "__main__":
    llm = get_local_llm()

    messages = [
        ChatMessage(role="user", content="Merhaba! Fazla mesai ücreti ne zaman ödenir?")
    ]

    print("\n===================")
    print("LLM Streaming Output:\n")

    for chunk in llm.stream_chat(messages):
        print(chunk.delta, end="", flush=True)

    print("\n===================\n")