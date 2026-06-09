# AI-Powered NL Data Profiling Q&A Assistant

A production-grade, full-stack AI application designed to automatically profile CSV datasets, index column metadata, and provide a conversational Q&A interface using a hybrid Retrieval-Augmented Generation (RAG) architecture powered by Ollama and ChromaDB.

---

## 🛠 Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python FastAPI, Uvicorn
- **Data Profiling**: `ydata-profiling` (formerly pandas-profiling), Pandas
- **Vector Database**: ChromaDB (Semantic column-schema matching)
- **Relational Database**: SQLite via SQLAlchemy (Persistent datasets, metadata, and Q&A history)
- **Orchestration**: LangChain
- **LLM**: Ollama (`qwen2.5:7b-instruct-q4_K_M`)

---

## 🏗️ System Architecture

The application is structured as a decoupled full-stack system consisting of a Streamlit frontend client, a FastAPI backend server, local vector & relational databases, and a local LLM client:

```mermaid
flowchart TB
    %% Nodes
    subgraph UI ["Client Layer (Streamlit)"]
        App["app.py (Web Dashboard)"]
    end

    subgraph API ["Backend API Layer (FastAPI)"]
        direction TB
        Main["main.py (App Lifespan)"]
        Router["router.py (API Routing)"]
        DS_EP["datasets.py (Endpoints)"]
        Chat_EP["chat.py (Endpoints)"]
    end

    subgraph Services ["Service Layer"]
        Profiler["profiler.py (ydata-profiling)"]
        PParser["profiler_parser.py (JSON Parser Cache)"]
        RAG["rag_engine.py (Hybrid RAG Router)"]
        VS["vector_store.py (ChromaDB Client)"]
        LLM["llm_client.py (LangChain Ollama Wrapper)"]
    end

    subgraph DB ["Data & Persistence Layer"]
        SQLite[("SQLite (sqlite.db)\n- Column Metadata\n- Chat History\n- Dataset State")]
        Chroma[("ChromaDB\n(Vector Embeddings)")]
        Disk[("Local Storage\n- Uploaded CSVs\n- HTML/JSON Profiles")]
    end

    %% Ingestion Flow
    App -->|"1. Upload CSV"| DS_EP
    DS_EP -->|"2. Write File"| Disk
    DS_EP -->|"3. Start Background Profiler"| Profiler
    Profiler -->|"4. Save HTML & JSON"| Disk
    Profiler -->|"5. Extract Stats"| SQLite
    Profiler -->|"6. Create Summary & Embed"| VS
    VS -->|"7. Index Metadata"| Chroma

    %% Query / Chat Flow
    App -->|"8. Send Chat Prompt"| Chat_EP
    Chat_EP -->|"9. Route & Query"| RAG
    RAG -->|"10. Keyword Match (Factual)"| PParser
    RAG -->|"10. Vector Similarity (Semantic)"| VS
    PParser -->|"Fetch metrics"| Disk
    VS -->|"Search docs"| Chroma
    RAG -->|"11. Load Context & Memory"| SQLite
    RAG -->|"12. Format Prompt & Predict"| LLM
    LLM -->|"13. Query Ollama"| Ollama[("Local Ollama\n(Qwen 2.5)")]
    RAG -->|"14. Save Chat History"| SQLite
    Chat_EP -->|"15. Return Answer & Citations"| App

    %% Styling
    classDef ui fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef api fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef svc fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;
    classDef db fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px;
    classDef ext fill:#eceff1,stroke:#37474f,stroke-width:2px;
    class App ui;
    class Main,Router,DS_EP,Chat_EP api;
    class Profiler,PParser,RAG,VS,LLM svc;
    class SQLite,Chroma,Disk db;
    class Ollama ext;
```

---

## 📁 Project Structure

```text
d:/NLP/
├── app.py                      # Streamlit frontend client application
├── backend/
│   ├── api/
│   │   ├── endpoints/
│   │   │   ├── chat.py         # Chat, query, and message history routes
│   │   │   └── datasets.py     # Upload, download, profiling, and metadata routes
│   │   └── router.py           # Primary endpoint aggregator
│   ├── config.py               # Environmental configuration (Pydantic-Settings)
│   ├── logging_config.py       # Centralized system logger configs
│   ├── main.py                 # FastAPI application entrypoint and lifespan events
│   └── services/
│       ├── llm_client.py       # Ollama LangChain Client wrapper
│       ├── profiler.py         # ydata-profiling analytics runner
│       ├── profiler_parser.py  # JSON statistics parser & caching loader
│       ├── rag_engine.py       # Router & LangChain Prompt template coordinator
│       └── vector_store.py     # ChromaDB collection & embedding service
├── db/
│   ├── chat_history_db.py      # Relational Q&A transaction helpers
│   ├── models.py               # SQLAlchemy schema definitions
│   └── session.py              # SQLite session provider
├── utils/
│   └── helpers.py              # Common string, file, and JSON decorators
├── requirements.txt            # System dependencies
├── .env                        # Local configurations override
└── .gitignore                  # Git ignore patterns
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- **Python 3.10+**
- **Ollama**: Download and install [Ollama](https://ollama.com/).
- Pull the instruction model locally:
  ```bash
  ollama pull qwen2.5:7b-instruct-q4_K_M
  ```

### 2. Installation Steps
1. Navigate to the project root directory:
   ```bash
   cd d:/NLP
   ```
2. Create and activate a python virtual environment:
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   ```
3. Install all python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Environment Variables Configuration
The application automatically creates directories and handles default setups. If you need to customize settings, copy `.env.example` to `.env` or edit the existing `.env` file in the project root:
```ini
APP_ENV=development
LOG_LEVEL=INFO
DATABASE_URL=sqlite:///./data/sqlite.db
CHROMA_DB_PATH=./data/chromadb
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct-q4_K_M
UPLOAD_DIR=./data/uploads
PROFILES_DIR=./data/profiles
MAX_FILE_SIZE_MB=50
```

---

## 🏃 Running the Application

To run the complete system, you must start both the **FastAPI Backend Server** and the **Streamlit Web UI**.

### 1. Start the FastAPI Backend
Ensure your virtual environment is active and execute:
```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```
- The backend API docs will be available at: http://127.0.0.1:8000/docs
- Relational SQLite tables are auto-created in `./data/sqlite.db` on start.

### 2. Start the Streamlit Frontend Client
Open a second terminal window, activate the virtual environment, and run:
```bash
streamlit run app.py
```
- The frontend dashboard will launch in your browser at: http://localhost:8501
