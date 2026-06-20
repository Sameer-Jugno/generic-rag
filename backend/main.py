from fastapi import FastAPI 

app = FastAPI() 


@app.get("/")
def check_initialization() : 
    return {
        "status" : "Healthy"
    }