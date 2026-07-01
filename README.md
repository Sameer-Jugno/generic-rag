# 💬 Decoupled Full-Stack Hybrid RAG Assistant

A production-grade, decoupled Retrieval-Augmented Generation (RAG) platform that enables lightning-fast streaming conversations over any text-based PDF document. 
The architecture completely isolates front-end UI delivery from heavy back-end vector parsing, orchestrates dual-server workloads concurrently inside a single Docker 
container environment, and features automated, self-healing metadata lifecycles.

---

## 🏗️ Architectural Topology & Core Pipelines

```text
            +--------------------------------------------------------+

            |               Streamlit Frontend UI                    |
            |        (Port 7860 - Hugging Face / iframe Secure)      |
            +---------------------------+----------------------------+
                                        |
                 REST API (HTTP)        |    Chunked Byte-Stream (SSE)
           Inference Data payloads      |    Real-Time Token Rendering
                                        v
            +--------------------------------------------------------+

            |               FastAPI Backend Engine                   |
            |                 (Port 8000 Internal)                   |
            +---------------------------+----------------------------+
                                        |
                 +----------------------+----------------------+

                 |                                             |
  Vector Dense + Sparse Extraction               Context-Aware Multi-Turn History
                 v                                             v
+----------------------------------+         +-----------------------------------+

|          Qdrant Cloud            |         |            Groq Cloud             |
|   (RRF Hybrid Search Engine)     |         |    (Llama 3.1-8b Inference API)   |
+----------------------------------+         +-----------------------------------+
```

### 1. The High-Performance Hybrid Ingestion Pipeline
* **Document Extraction**: Employs structural token matching via `PyMuPDF` to stream file byte streams safely in memory, eliminating brittle system-level tmp directory read locks.
* **Document Constraints**: Optimized for processing standard text-based PDFs up to 5–10 MB. Highly comprehensive or heavily dense documents are supported but may require additional processing
*   time during the initial embedding extraction step.
* **Dual-Vector Generation**: Utilizes FastEmbed loops to simultaneously create high-density vectors (semantic representation) and sparse BM25 structures (exact keyword token mapping).
* **Qdrant Cloud Ingestion**: Transmits multi-vector point structures into cloud clusters using an automated collection lifecycle management system.

### 2. The Context-Restrained Retrieval & Streaming Pipeline
* **Query Re-Writing Loop**: A multi-turn history parsing service intercepts conversational queries and runs them through a deterministic Llama 3.1 layer to dynamically resolve relative pronouns (e.g., "that business unit") before querying the database.
* **Reciprocal Rank Fusion (RRF)**: Merges sparse keyword matches and dense embeddings inside Qdrant Cloud to deliver context matching across complex data sets.
* **Server-Sent Events (SSE)**: Bypasses heavy HTTP memory overheads by opening a streaming reader loop, delivering real-time assistant responses token-by-token.

---

## 🛠️ Tech Stack & Production Tooling

* **Core Language Frameworks**: Python 3.10, FastAPI (Asynchronous Web Gateway), Streamlit (Real-time Canvas Engine).
* **Vector Engine & Search Infrastructure**: Qdrant Cloud Cluster, FastEmbed (Dense & Sparse Embedding Matrices), Reciprocal Rank Fusion (RRF).
* **LLM Foundation Layer**: Groq Cloud Platform, Meta Llama-3.1-8b-Instant (Advanced Reasoning Context Node).
* **DevOps & Infrastructure Containment**: Docker Slim Linux, Multi-Process Bash Orchestrator (`run.sh`), Custom AppArmor & CORS Network Gateway Overrides.

---

## 🐳 Self-Contained Local Execution (Docker Setup)

This entire application is completely containerized. You do not need to install Python, configure library files, or maintain virtual environments to execute it locally.

### 1. Clone the Space and Configure Your Environment Keys
Create a `.env` file right inside your root directory folder layout:
```env
QDRANT_CLOUD_URL=https://qdrant.io
QDRANT_API_KEY=your_private_qdrant_api_key
GROQ_API_KEY=your_private_groq_api_key
```

### 2. Build and Execute the Container Environment
```bash
# Compile the production image layer stack using Docker caching flags
docker build -t generic-rag-app .

# Spin the full-stack multi-process sandbox container online instantly
docker run -d -p 7860:7860 --env-file .env --name running-rag generic-rag-app
```
Access the application's user interface directly by loading the local port routing address inside your web browser.

---

## 🚀 Try It Out

Test the production-grade deployment live right now without any local machine configuration setup:

👉 **[Launch Live RAG Application on Hugging Face Spaces]([https://huggingface.co](https://huggingface.co/spaces/SameerJugno/pdf-chat-assistant))**
