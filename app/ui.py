import streamlit as st

st.set_page_config(page_title="AI Chat Assistant", layout="centered")

# Basit sohbet gecmişi için session state
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
    # Kullanıcı mesajı ekle ve görüntüle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Dummy cevap şimdilik
    response = f"'{prompt}' mesajını aldım"
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
