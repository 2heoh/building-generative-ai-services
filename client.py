# client.py

import requests
import streamlit as st

st.title("FastAPI ChatBot") 

if "messages" not in st.session_state:
    st.session_state.messages = [] 

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True) 

st.write("Upload a file to FastAPI")
file = st.file_uploader("Choose a file", type=["pdf"])

if st.button("Submit"):
    if file is not None:
        files = {"file": (file.name, file, file.type)}
        response = requests.post("http://localhost:8000/upload", files=files)
        if response.ok:
            st.success("File uploaded. Wait ~10 seconds for indexing before asking questions.")
        else:
            st.error(response.text)
    else:
        st.write("No file uploaded.")

if prompt := st.chat_input("Write your prompt in this input field"): 
    st.session_state.messages.append({"role": "user", "content": prompt}) 

    with st.chat_message("user"):
        st.text(prompt) 

    response = requests.post(
        "http://localhost:8000/generate/text",
        json={
            "model": "tinyLlama",
            "prompt": prompt,
            "temperature": 0.7,
        },
    )
    response.raise_for_status()

    content = response.json()["content"]

    with st.chat_message("assistant"):
        st.markdown(content, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": content})
