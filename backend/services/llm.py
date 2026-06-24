import os 
from google import genai
from backend.services.vector_db import query_vector_db

def rewrite_query_with_history(query: str, history: list) -> str:
    """Uses gemini-2.5-flash to re-write ambiguous queries into standalone search strings based on history."""
    prompt = f"""
        Given the following conversation history and a new user query, re-write the user query into a completely standalone search query that includes all necessary context (resolving pronouns like 'it', 'they', 'he', 'she'). 

        History: {history}
        User Query: {query}

        Output ONLY the raw re-written query string. Do NOT include any conversational filler, explanations, markdown formatting, or quotes.
    """
    client = get_gemini_client() 
    response = client.models.generate_content(
        model="gemini-2.5-flash",   
        contents=prompt
    )
    return response.text

def generate_stream_response(query: str, contexts: list, history: list):
    """Orchestrates retrieved content and logs into a structured system prompt, yielding text tokens in real-time."""
    client = get_gemini_client()
    prompt = f"""
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

    response = client.models.generate_content_stream(
        model="gemini-2.5-flash", 
        contents=prompt
    ) 
    for chunk in response: 
        yield chunk.text

def get_gemini_client() : 
    key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=key)
    return client 