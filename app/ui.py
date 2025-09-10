import streamlit as st
from rag import rag_pipeline  # RAG pipeline importu

st.set_page_config(page_title="AI Chat Assistant", layout="centered")

# Sohbet geçmişi
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Merhaba 👋 Ben senin AI asistanınım. Size nasıl yardımcı olabilirim?"}
    ]

st.title("AI Chat Assistant")

# Sohbet balonları
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Kullanıcıdan input al
if prompt := st.chat_input("Mesajınızı yazın..."):
    # Kullanıcı mesajı ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG pipeline çağır
    with st.chat_message("assistant"):
        with st.spinner("Yanıt üretiliyor..."):
            answer, sources, latency = rag_pipeline(prompt, stream=False)

        st.markdown(answer if answer else "Bilmiyorum")
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Kaynakları göster
        with st.expander("📚 Kaynaklar"):
            for i, s in enumerate(sources, 1):
                st.markdown(f"[{i}] ({s['source']}) {s['text'][:200]}...")

        st.caption(f"⏱️ {latency:.2f} saniye")