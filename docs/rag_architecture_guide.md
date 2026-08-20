# Document Chatbot (RAG System) Architecture

This document explains the architecture of the **Retrieval-Augmented Generation (RAG) Document Chatbot**. While the Database Chatbot turns questions into SQL, this engine allows users to upload documents, vectorizes them, and answers natural language questions based strictly on the uploaded text.

## 1. High-Level Architecture Diagram

```mermaid
graph TD
    %% User Inputs
    User_Upload[User Uploads Document]
    User_Question[User Asks Question]
    
    %% API Endpoints (fastapi_app.py)
    API_Upload[/upload-file]
    API_Ask[/ask-your-query]
    API_Stream[/ask-your-query-stream]
    
    %% RAG Engine (v2_rag_engine.py)
    subgraph RAG_Engine [RAG Engine - v2_rag_engine.py]
        Chunker[Document Parser & Chunker]
        Embedder[Embedding Model: all-MiniLM-L6-v2]
        Reranker[Cross-Encoder Reranker]
        Translator[Language Translator]
    end
    
    %% Storage
    subgraph Local_Storage [File System & Databases]
        Folder_Upload[upload_file_RAG/]
        ChromaDB[(ChromaDB Vector Store)]
    end
    
    %% LLM Backend
    Ollama[Ollama Client / Local LLMs]
    
    %% Ingestion Flow
    User_Upload --> API_Upload
    API_Upload --> Folder_Upload
    API_Upload --> Chunker
    Chunker --> Embedder
    Embedder -->|Save Embeddings| ChromaDB
    
    %% Query Flow
    User_Question --> API_Ask
    API_Ask --> Translator
    Translator -->|Search| ChromaDB
    ChromaDB -->|Top N Matches| Reranker
    Reranker -->|Best Matches| API_Ask
    API_Ask --> Ollama
    Ollama -->|Final Answer| User_Question
```

---

## 2. Component Breakdown

### A. The Endpoints (`fastapi_app.py`)
* **`/upload-file`**: Accepts file uploads (like PDFs, TXT). It saves the raw file to the local directory `upload_file_RAG/` and immediately passes it to the RAG engine for processing.
* **`/ask-your-query` & `/ask-your-query-stream`**: Receives a natural language question. It searches the uploaded documents, builds an AI prompt with the search results, and returns the answer (either fully processed or streaming).
* **`/is-file-present`**: Checks if there is an active document currently loaded in the system.

### B. The RAG Engine (`v2_rag_engine.py`)
This is the core processor for documents. It performs several heavy-lifting tasks:

1. **Document Ingestion (`ingest_file_v2`)**: 
   When a file is uploaded, the engine reads the text and splits it into smaller, manageable "chunks." 
2. **Vector Embeddings**: 
   These chunks are converted into numerical vectors using a lightweight, lightning-fast embedding model (like `all-MiniLM-L6-v2`) and stored in **ChromaDB**. 
3. **Retrieval & Reranking (`retrieve_and_rerank`)**: 
   When a question is asked, it converts the question to a vector, finds the top matching chunks in ChromaDB, and then passes them through a **Cross-Encoder**. The cross-encoder meticulously scores how relevant each chunk is to the specific question and sorts them so only the absolute best information reaches the AI.
4. **Multilingual Support**: 
   It detects if the question was asked in a language other than English. If so, it translates the question to English to search the documents, and then translates the final AI answer back to the user's native language.

### C. Local Storage
* **`upload_file_RAG/`**: The physical directory where the uploaded documents are temporarily stored. When you upload a new document, the previous document is cleared to maintain context.
* **ChromaDB**: An advanced, locally-hosted vector database. It stores the mathematical representations of your documents so the AI can search through hundreds of pages in milliseconds.

### D. The LLM (`ollama_client.py`)
* The text chunks extracted by the RAG Engine are sent to a local LLM running via Ollama (e.g., `llama3:latest`). 
* The LLM is strictly prompted to **only** use the provided document context to answer the question, ensuring it does not hallucinate or guess answers based on external knowledge.

---

## 3. Data Flow: "What happens when you upload a document?"

1. **Upload**: You submit a PDF via the frontend.
2. **Save**: `/upload-file` saves it to the `upload_file_RAG/` folder, clearing out old files.
3. **Chunk & Embed**: `v2_rag_engine` reads the PDF, splits it into paragraphs, turns them into vectors, and saves them in ChromaDB.

## 4. Data Flow: "What happens when you ask the Document Chatbot?"

1. **Question**: You ask "What is the total budget mentioned in the report?".
2. **Translate**: If the question is in Hindi, it automatically translates it to English for better search accuracy.
3. **Vector Search**: It searches ChromaDB and pulls out the top 5 paragraphs that contain budget numbers.
4. **Rerank**: The Cross-Encoder reviews those 5 paragraphs and picks the 3 that actually answer the question.
5. **Prompt AI**: Those 3 paragraphs are sent to Ollama with the instruction: *"Using ONLY this document text, answer the user's question."*
6. **Answer**: Ollama generates the final text and streams it back to your screen.
