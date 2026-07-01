# 💬 Decoupled Full-Stack Hybrid RAG Assistant

A production-grade, decoupled Retrieval-Augmented Generation (RAG) platform that enables lightning-fast streaming conversations over text-based PDF documents. This architecture isolates front-end UI delivery from heavy back-end vector parsing, orchestrates dual-server workloads concurrently inside a single Docker container environment, and features automated, self-healing metadata lifecycles.

---

## 🏗️ Architectural Topology & Core Pipelines

```text
            +--------------------------------------------------------+

            |               Streamlit Frontend UI                    |
            |        (Hugging Face Space / iframe Responsive)        |
            +---------------------------+----------------------------+
                                        |
                 REST API (HTTP)        |    Chunked Byte-Stream (SSE)
           Inference Data payloads      |    Real-Time Token Rendering
                                        v
            +--------------------------------------------------------+

            |               FastAPI Backend Engine                   |
            |                 (Internal API Router)                  |
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

### 1. High-Performance Hybrid Ingestion Pipeline
* **Document Extraction**: Employs structural token matching via `PyMuPDF` to stream file byte streams safely in memory, eliminating brittle system-level tmp directory read locks.
* **Document Constraints**: Optimized for processing standard text-based PDFs up to 5–10 MB. High-density documents are fully supported, with processing time dynamically adapting during the initial embedding extraction step.
* **Dual-Vector Generation**: Utilizes FastEmbed loops to simultaneously create high-density vectors (semantic representation) and sparse BM25 structures (exact keyword token mapping).
* **Qdrant Cloud Ingestion**: Transmits multi-vector point structures into the cloud cluster using an automated collection lifecycle management system.

### 2. Context-Restrained Retrieval & Streaming Pipeline
* **Query Re-Writing Loop**: A multi-turn history parsing service intercepts conversational queries and runs them through a deterministic Llama 3.1 layer to dynamically resolve relative pronouns (e.g., "that company's challenges") before querying the database.
* **Reciprocal Rank Fusion (RRF)**: Merges sparse keyword matches and dense embeddings inside Qdrant Cloud to deliver accurate context matching across complex data sets.
* **Server-Sent Events (SSE)**: Bypasses heavy HTTP memory overheads by opening a streaming reader loop, delivering real-time assistant responses token-by-token.

---

## 🛠️ Tech Stack & Production Tooling

* **Core Language Frameworks**: Python 3.10, FastAPI (Asynchronous Web Gateway), Streamlit (Real-time Canvas Engine).
* **Vector Engine & Search Infrastructure**: Qdrant Cloud Cluster, FastEmbed (Dense & Sparse Embedding Matrices), Reciprocal Rank Fusion (RRF).
* **LLM Foundation Layer**: Groq Cloud Platform, Meta Llama-3.1-8b-Instant (Advanced Reasoning Context Node).
* **DevOps & Infrastructure Containment**: Docker Slim Linux, Multi-Process Bash Orchestrator (`run.sh`), Custom AppArmor & CORS Network Gateway Overrides.

---

## 🚀 Try It Out

Test the production-grade deployment live right now without any local machine configuration setup:

👉 **Launch Live RAG Application on Hugging Face Spaces : https://huggingface.co/spaces/SameerJugno/pdf-chat-assistant**
