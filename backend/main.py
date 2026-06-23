from fastapi import FastAPI, UploadFile, File, Form 
from backend.services.vector_db import (
    extract_text_from_pdf, 
    chunk_text, 
    generate_embeddings, 
    upsert_vectors
)
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
        