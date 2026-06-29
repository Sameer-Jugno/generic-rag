import streamlit as st 
import requests 
import uuid

st.set_page_config(page_title="PDF Chat Assistant", layout="wide", page_icon="💬")
st.title("PDF Chat Assistant")
st.caption("A decoupled, production-grade RAG architecture running via FastAPI and Docker.")

# 1. Initialize global states safely
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "unique_id" not in st.session_state:
    st.session_state["unique_id"] = str(uuid.uuid4())

if "file_uploaded" not in st.session_state:
    st.session_state["file_uploaded"] = False

with st.sidebar: 
    st.header("⚙️ System Configurations")
    
    top_k = st.slider(
        label="Top_k", min_value=1, max_value=8, value=3, step=1 
    )
    st.session_state["top_k"] = top_k
    st.markdown("---")
    st.header("📂 Document Ingestion")
    
    uploaded_file = st.file_uploader("Upload 1 PDF Document", type=["pdf"])
    
    # NEW FLAG INITIALIZATION: Tracks if the collection was just explicitly destroyed
    if "just_deleted" not in st.session_state:
        st.session_state["just_deleted"] = False

    # If the user manually clicks the 'X' on the widget to remove the file, reset our deletion flag automatically
    if uploaded_file is None:
        st.session_state["just_deleted"] = False
        st.session_state["file_uploaded"] = False

    if uploaded_file is not None: 
        file_name = uploaded_file.name 
        if file_name.lower().endswith(".pdf"):
            # UPDATED GUARD LAYER: Only uploads if it hasn't been uploaded AND wasn't just explicitly deleted
            if not st.session_state["file_uploaded"] and not st.session_state["just_deleted"]:
                with st.spinner("Transmitting knowledge base..."):
                    files = {"file": (uploaded_file.name, uploaded_file.read(), "application/pdf")}
                    data = {"session_id": st.session_state["unique_id"]}
                    requests.post("http://localhost:8000/api/upload", files=files, data=data)
                    st.session_state["file_uploaded"] = True
            
            # Dynamically adjusts status notifications based on active states
            if st.session_state["file_uploaded"]:
                st.success(f"File '{uploaded_file.name}' integrated and active in cloud vault.")
            elif st.session_state["just_deleted"]:
                st.warning("Collection purged. Please clear or re-upload the document widget to resume.")
        else:
            st.error("Only PDF files are allowed.")

    # The clean-up action button executor block
    if st.session_state["file_uploaded"]: 
        st.markdown("---")
        if st.button(label="🗑️ Delete Collection", use_container_width=True): 
            session_id = st.session_state["unique_id"]
            if session_id: 
                ans = requests.delete(f"http://localhost:8000/api/collection/{session_id}")
                if ans.status_code == 200: 
                    # UPDATED RE-ROUTE STATE TRACKING: Blocks subsequent accidental uploads
                    st.session_state["file_uploaded"] = False 
                    st.session_state["just_deleted"] = True 
                    st.session_state["chat_history"] = []
                    st.rerun()

st.markdown("---")

# FIXED: Re-render the historical conversation messages so they persist on screen
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["text"])

# Handle the new user input stream
if user_query := st.chat_input("Ask a question..." if st.session_state["file_uploaded"] else "Please upload pdf to start conversation",
        disabled=not st.session_state["file_uploaded"]
    ):

    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.chat_history.append({"role": "user", "text": user_query})
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing data vectors..."):
            data = {
                "query": user_query,
                "session_id": st.session_state["unique_id"],
                "top_k": st.session_state["top_k"],
                "history": st.session_state["chat_history"]
            }
            ans = requests.post("http://localhost:8000/chat/api", json=data, stream=True)
            text = st.write_stream(ans.iter_content(decode_unicode=True))

    st.session_state.chat_history.append({"role": "assistant", "text": text})