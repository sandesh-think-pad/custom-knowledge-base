import time
import sys
from pathlib import Path

import streamlit as st

root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from rag_pipeline.pipeline import get_data

st.write("Welcome to policy knowledge base assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Let's start asking questions! 👇"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your company policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = get_data(prompt)
        displayed = ""
        for word in response.split():
            displayed += word + " "
            time.sleep(0.05)
            message_placeholder.markdown(displayed + "▌")
        message_placeholder.markdown(displayed.strip())

    st.session_state.messages.append({"role": "assistant", "content": displayed.strip()})
