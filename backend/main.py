from fastapi import FastAPI, UploadFile, File, Form 
from pydantic import BaseModel 
from fastapi.responses import StreamingResponse 
from backend.services.vector_db import (
    extract_text_from_pdf, 
    chunk_text, 
    generate_embeddings, 
    upsert_vectors,
    query_vector_db
)
from backend.services.llm import (
    rewrite_query_with_history,
    generate_stream_response
)

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
    fileBytes = await file.read() 
    chunks = extract_text_from_pdf(fileBytes)
    documents = chunk_text(chunks)
    points = generate_embeddings(documents)
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
