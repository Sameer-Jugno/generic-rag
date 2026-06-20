import streamlit as st 

st.set_page_config(page_title="PDF Chat Assistant", layout="wide", page_icon="💬")
st.title("PDF Chat Assistant")
st.caption("A decoupled, production-grade RAG architecture running via FastAPI and Docker.")

with st.sidebar : 
    st.header("⚙️ System Configurations")
    
    tok_k = st.slider(
        label="Top_k", 
        min_value=1, 
        max_value=8,
        value=3,
        step=1 
    )

    st.markdown("---")
    st.header("📂 Document Ingestion")
    
    uploaded_file = st.file_uploader("Upload 1 PDF Document", type=["pdf"])
    print(type(uploaded_file))
    if uploaded_file is not None : 
        file_name = uploaded_file.name 
        print(f"File Name : {file_name}")
        if file_name.lower().endswith(".pdf"):
            st.success(f"File '{uploaded_file.name}' received! Ready to transmit to backend service.")
        else :
            st.error("Only PDF files are allowed.")