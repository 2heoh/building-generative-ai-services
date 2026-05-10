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
        st.write(response.text)
    else:
        st.write("No file uploaded.")

if prompt := st.chat_input("Write your prompt in this input field"): 
    st.session_state.messages.append({"role": "user", "content": prompt}) 

    with st.chat_message("user"):
        st.text(prompt) 

    response = requests.get(
        f"http://localhost:8000/generate/text", params={"prompt": prompt}
    ) 
    response.raise_for_status() 

    with st.chat_message("assistant"):
        st.markdown(response.text, unsafe_allow_html=True) 
