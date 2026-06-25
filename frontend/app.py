import streamlit as st 
import requests 
import uuid

st.set_page_config(page_title="PDF Chat Assistant", layout="wide", page_icon="💬")
st.title("PDF Chat Assistant")
st.caption("A decoupled, production-grade RAG architecture running via FastAPI and Docker.")

with st.sidebar : 
    st.header("⚙️ System Configurations")
    
    top_k = st.slider(
        label="Top_k", 
        min_value=1, 
        max_value=8,
        value=3,
        step=1 
    )
    st.session_state["top_k"] = top_k
    st.markdown("---")
    st.header("📂 Document Ingestion")
    
    uploaded_file = st.file_uploader("Upload 1 PDF Document", type=["pdf"])
    print(type(uploaded_file))
    if uploaded_file is not None : 
        file_name = uploaded_file.name 
        print(f"File Name : {file_name}")
        if file_name.lower().endswith(".pdf"):
            st.success(f"File '{uploaded_file.name}' received! Ready to transmit to backend service.")

            files = {
                "file" : (
                    uploaded_file.name, uploaded_file.read() , "application/pdf"
                )
            }
            data = {
                    "session_id" : st.session_state["unique_id"]
                }
            requests.post("http://localhost:8000/api/upload", files=files, data=data)
        else :
            st.error("Only PDF files are allowed.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "unique_id" not in st.session_state:
    st.session_state["unique_id"] = str(uuid.uuid4())


if user_query := st.chat_input("Ask a question..."):
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.chat_history.append({"role": "user", "text": user_query})
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing resutls..."):
            data = {
                "query" : user_query,
                "session_id" : st.session_state["unique_id"],
                "top_k" : st.session_state["top_k"],
                "history" : st.session_state["chat_history"]
            }
            ans = requests.post("http://localhost:8000/chat/api", json=data, stream=True)
            text = st.write_stream(ans.iter_content(decode_unicode=True))
            print(text) 

    st.session_state.chat_history.append({
        "role": "assistant", 
        "text": text
    })
