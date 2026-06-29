from fastapi import FastAPI, UploadFile, File, Form 
from dotenv import load_dotenv
from pydantic import BaseModel 
from fastapi.responses import StreamingResponse 
from backend.services.vector_db import (
    extract_text_from_pdf, 
    chunk_text, 
    generate_embeddings, 
    upsert_vectors,
    query_vector_db, 
    delete_qdrant_collection
)
from backend.services.llm import (
    rewrite_query_with_history,
    generate_stream_response
)

load_dotenv() 

class ChatRequest(BaseModel):
    query: str
    session_id: str
    top_k: int = 4
    history: list = []

app = FastAPI() 

@app.get("/")
def check_initialization() : 
    return {
        "status" : "Healthy"
    }

@app.post("/api/upload") 
async def implement_ingestion_pipeline(file : UploadFile = File(...), session_id : str = Form(...) ) :
    print("-"*100)
    print("Receiving File bytes...")
    print("-"*100)
    
    fileBytes = await file.read() 
    
    print("-"*100)
    print("Extracting text from received file...")
    print("-"*100)
    chunks = extract_text_from_pdf(fileBytes)
    
    print("-"*100)
    print("Chunking the documents...")
    print("-"*100)
    documents = chunk_text(chunks)

    print("-"*100)
    print("Generating embeddings...")
    print("-"*100)
    points = generate_embeddings(documents)

    print("-"*100)
    print("Uploading embeddings...")
    print("-"*100)
    result = upsert_vectors(session_id, points)  
    if result : 
        return {
            "status": "success", 
            "message": "Knowledge base integrated"
        }
    else : 
        return {
            "status": "Failed", 
            "message": "Failed in Knowledge base integration"
        }

@app.post("/chat/api") 
async def implement_chat_api(response : ChatRequest) : 
    
    rewritten_query = rewrite_query_with_history(response.query, response.history)
    context = query_vector_db(response.session_id, rewritten_query, response.top_k, )

    return StreamingResponse(
        generate_stream_response(response.query, context, response.history),
        media_type="text/plain"
    )

@app.delete("/api/collection/{session_id}") 
def implementing_chat_deletion(session_id : str) : 

    status = delete_qdrant_collection(session_id)
    if status : 
        return {
            "Status" : status, 
            "Response" : "Current Conversation's Collection is removed."
        }
    else : 
        return {
            "Status" : status, 
            "Response" : "Current Conversation's Collection is unabled to remove."
        }