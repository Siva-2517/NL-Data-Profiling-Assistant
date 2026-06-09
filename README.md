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
└── .env                        # Local configurations override
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

---

## 🧪 Running Validation & Tests

The system comes with test scripts to verify the functionality of the system components.

### Run Performance & Caching Tests:
Checks the JSON parser caching, ChromaDB singleton connection reuse, and LLM Client singleton configurations.
```bash
python C:\Users\HP\.gemini\antigravity-ide\brain\80b1b07a-000f-4729-aa8e-985d802e30cc\scratch\test_performance.py
```

### Run Chat History Integration Tests:
Verifies Q&A persistence, formatting, history retrieval, and database clearing.
```bash
python C:\Users\HP\.gemini\antigravity-ide\brain\80b1b07a-000f-4729-aa8e-985d802e30cc\scratch\test_chat_history.py
```
