import os 
import time
# from google import genai
# from google.genai.errors import APIError
from groq import Groq 
from groq import GroqError 
from backend.services.vector_db import query_vector_db

def rewrite_query_with_history(query: str, history: list) -> str:

    system_prompt = (
        "Given a conversation history and a new user query, re-write the user query into a "
        "completely standalone search query that includes all necessary context (resolving pronouns like 'it', 'they'). "
        "Output ONLY the raw re-written query string. Do NOT include explanations, markdown, or conversational filler."
    )

    messages = [{"role" : "system", "content" : system_prompt}] 
    for msg in history : 
        messages.append({"role" : msg["role"], "content" : msg["text"]})
    messages.append({"role":"user", "content":query}) 

    client = get_groq_client() 
    for attempt in range(3) : 
        try : 
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant", 
                messages=messages, 
                temperature=0.0
            )
            return response.choices[0].message.content.strip() 
        except GroqError as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise e
    return response.text

def generate_stream_response(query: str, contexts: list, history: list):
    """Orchestrates retrieved content and logs into a structured system prompt, yielding text tokens in real-time."""
    client = get_groq_client()
    system_prompt = f"""
        You are a highly capable enterprise PDF Chat Assistant. Your job is to answer the user's question accurately using ONLY the provided reference document context segments below.

        Retrieved Document Context:
        {contexts}

        Conversation History:
        {history}

        User Question: {query}

        Strict Execution Instructions:
        1. Base your answer completely on the provided context text snippets.
        2. If the answer cannot be found or reasonably inferred from the context, state clearly and politely that the document does not contain enough information.
        3. Do NOT invent, assume, or hallucinate facts that are outside the provided reference text boundaries.
    """
    messages = [{"role":"system", "content":system_prompt}]
    
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["text"]})
    messages.append({"role": "user", "content": query})
    
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant", 
                messages=messages,
                temperature=0.3,
                stream=True
            ) 
            for chunk in response : 
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            return 
        except GroqError as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise e


# def get_gemini_client() : 
#     key = os.getenv("GEMINI_API_KEY")
#     client = genai.Client(api_key=key)
#     return client 

def get_groq_client() : 
    key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=key) 