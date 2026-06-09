"""
Streamlit Web UI Client for AI-powered NL Data Profiling Q&A Assistant.
Provides interfaces for uploading CSVs, displaying generated ydata-profiling reports,
and interacting with datasets through a conversational RAG chat panel.
"""

import logging
import streamlit as st
import pandas as pd
import httpx
from typing import Dict, List, Any

# Configure logging for the frontend application
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("streamlit_frontend")

# Configure wide layout and custom title
st.set_page_config(
    page_title="NL Data Profiling Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend URL base config
API_BASE_URL = "http://localhost:8000/api/v1"

def init_session_states():
    """Initialize session variables if they don't exist."""
    if "selected_dataset_id" not in st.session_state:
        st.session_state.selected_dataset_id = None
    if "active_chat_session_id" not in st.session_state:
        st.session_state.active_chat_session_id = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "datasets_list" not in st.session_state:
        st.session_state.datasets_list = []

def fetch_datasets():
    """Fetches the list of all datasets from the backend API."""
    try:
        response = httpx.get(f"{API_BASE_URL}/datasets/", timeout=10.0)
        if response.status_code == 200:
            st.session_state.datasets_list = response.json()
        else:
            st.error(f"Failed to fetch datasets: Status {response.status_code}")
    except Exception as e:
        st.error(f"Could not connect to backend API: {e}. Make sure the backend server is running.")

def fetch_dataset_details(dataset_id: str) -> Dict[str, Any]:
    """Retrieves metadata properties for a given dataset."""
    try:
        response = httpx.get(f"{API_BASE_URL}/datasets/{dataset_id}", timeout=10.0)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logger.error(f"Error fetching dataset details for {dataset_id}: {e}")
    return {}

def fetch_chat_history(dataset_id: str):
    """Fetches the message logs from the backend API for a given dataset."""
    try:
        response = httpx.get(f"{API_BASE_URL}/chat/history/{dataset_id}", timeout=10.0)
        if response.status_code == 200:
            st.session_state.chat_history = response.json()
        else:
            st.error(f"Failed to load chat history: Status {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to fetch chat history: {e}")

def render_sidebar():
    """Renders the sidebar navigation, dataset upload form, and list of datasets."""
    st.sidebar.title("📁 Datasets Manager")
    
    # Refresh datasets on sidebar render
    if not st.session_state.datasets_list:
        fetch_datasets()

    # Upload component
    st.sidebar.subheader("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size_mb > 50:
            st.sidebar.error("File exceeds 50MB maximum size limit.")
        else:
            if st.sidebar.button("Upload & Profile", use_container_width=True):
                with st.sidebar.spinner("Uploading and starting profiling task..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                        res = httpx.post(f"{API_BASE_URL}/datasets/", files=files, timeout=60.0)
                        if res.status_code == 201:
                            st.sidebar.success("Upload complete! Profiling queued.")
                            fetch_datasets()
                            st.rerun()
                        else:
                            detail = res.json().get("detail", "Server upload rejection.")
                            st.sidebar.error(f"Upload failed: {detail}")
                    except Exception as e:
                        st.sidebar.error(f"Upload error: {e}")

    st.sidebar.markdown("---")
    
    # Active datasets selector
    st.sidebar.subheader("Active Datasets")
    if not st.session_state.datasets_list:
        st.sidebar.info("No datasets loaded yet. Upload a CSV file to begin.")
    else:
        # Build selection dictionary mapping human label with status to full dataset object
        options = {}
        for d in st.session_state.datasets_list:
            status_emoji = "⏳" if d["status"] == "PENDING" else "✅" if d["status"] == "PROFILED" else "❌"
            label = f"{d['filename']} ({status_emoji} {d['status']})"
            options[label] = d
            
        select_list = list(options.keys())
        # Keep index matched to active dataset state if present
        selected_index = 0
        if st.session_state.selected_dataset_id:
            for idx, item in enumerate(select_list):
                if options[item]["id"] == st.session_state.selected_dataset_id:
                    selected_index = idx
                    break
                    
        selected_label = st.sidebar.selectbox("Choose active dataset", options=select_list, index=selected_index)
        active_ds = options[selected_label]
        
        # If the active dataset ID changes, clear the active chat session context
        if st.session_state.selected_dataset_id != active_ds["id"]:
            st.session_state.selected_dataset_id = active_ds["id"]
            st.session_state.active_chat_session_id = None
            st.session_state.chat_history = []
            st.rerun()
        
        # Display dataset contextual tools
        if active_ds["status"] == "PENDING":
            st.sidebar.warning("Profiling in progress. Please refresh periodically.")
            if st.sidebar.button("🔄 Refresh Status", use_container_width=True):
                fetch_datasets()
                st.rerun()
        elif active_ds["status"] == "FAILED":
            st.sidebar.error("Profiling failed. View logs or delete dataset.")
            
        # Delete dataset trigger
        if st.sidebar.button("🗑️ Delete Dataset", use_container_width=True):
            with st.sidebar.spinner("Purging files and data collections..."):
                try:
                    res = httpx.delete(f"{API_BASE_URL}/datasets/{active_ds['id']}", timeout=15.0)
                    if res.status_code == 200:
                        st.sidebar.success("Dataset deleted.")
                        st.session_state.selected_dataset_id = None
                        st.session_state.active_chat_session_id = None
                        st.session_state.chat_history = []
                        fetch_datasets()
                        st.rerun()
                    else:
                        st.sidebar.error(f"Delete failed: {res.json().get('detail')}")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")

def render_dashboard_tab():
    """Displays ydata-profiling HTML reports and core metrics of the dataset."""
    st.header("📊 Dataset Profiling Dashboard")
    if not st.session_state.selected_dataset_id:
        st.warning("Please upload or select a dataset from the sidebar to view profiling statistics.")
        return

    # Find dataset record from list
    active_ds = next((d for d in st.session_state.datasets_list if d["id"] == st.session_state.selected_dataset_id), None)
    if not active_ds:
        st.error("Selected dataset not found in cached datasets list. Try refreshing.")
        return

    if active_ds["status"] == "PENDING":
        st.info("Dataset profiling is currently running in the background. Check back in a few seconds!")
        st.spinner("Executing pandas profile analyzers...")
        return
    elif active_ds["status"] == "FAILED":
        st.error("Data profiling failed for this dataset. The file might contain invalid formatting, empty rows, or incorrect delimiting characters.")
        return

    # Fetch statistical properties from backend details endpoint
    details = fetch_dataset_details(st.session_state.selected_dataset_id)
    if details:
        row_count = details.get("row_count", 0)
        col_count = details.get("col_count", 0)
        cols = details.get("columns", [])
        
        # Calculate average missing pct across variables
        if cols:
            avg_missing = sum(c.get("missing_pct", 0.0) for c in cols) / len(cols)
            avg_missing_str = f"{avg_missing:.2f}%"
        else:
            avg_missing_str = "0.00%"
            
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Rows", value=f"{row_count:,}")
        with col2:
            st.metric(label="Total Columns", value=col_count)
        with col3:
            st.metric(label="Average Missing Values", value=avg_missing_str)

    # Embed pre-compiled HTML ydata-profiling report via iframe
    st.markdown("### Detailed Variables Profile Report")
    profile_html_url = f"{API_BASE_URL}/datasets/{st.session_state.selected_dataset_id}/profile"
    try:
        st.components.v1.iframe(profile_html_url, height=850, scrolling=True)
    except Exception as iframe_err:
        st.error(f"Failed to embed profiling dashboard iframe: {iframe_err}")

def render_chat_tab():
    """Renders the conversational interface, source citations, and question input."""
    st.header("💬 AI Data Q&A Assistant")
    if not st.session_state.selected_dataset_id:
        st.warning("Please select a dataset to start asking questions.")
        return

    # Auto-initialize chat session for this dataset if not already present
    if not st.session_state.active_chat_session_id:
        try:
            with st.spinner("Initializing AI conversational workspace..."):
                res = httpx.post(
                    f"{API_BASE_URL}/chat/sessions",
                    json={
                        "dataset_id": st.session_state.selected_dataset_id,
                        "title": f"Chat session for dataset ID: {st.session_state.selected_dataset_id[:8]}"
                    },
                    timeout=10.0
                )
                if res.status_code == 201:
                    st.session_state.active_chat_session_id = res.json()["session_id"]
                    st.session_state.chat_history = []
                else:
                    st.error(f"Failed to start chat session: {res.json().get('detail')}")
                    return
        except Exception as e:
            st.error(f"Failed to connect to backend chat endpoints: {e}")
            return

    # Trigger loading message history if it's empty
    if not st.session_state.chat_history:
        fetch_chat_history(st.session_state.selected_dataset_id)

    # Use columns to position explanation and the clear history button neatly
    col1, col2 = st.columns([5, 1.2])
    with col1:
        st.write("Ask natural language questions about your dataset's columns, shapes, statistics, and categories.")
    with col2:
        if st.button("🧹 Clear Chat History", use_container_width=True):
            with st.spinner("Clearing chat history..."):
                try:
                    res = httpx.delete(f"{API_BASE_URL}/chat/history/{st.session_state.selected_dataset_id}", timeout=10.0)
                    if res.status_code == 200:
                        st.session_state.chat_history = []
                        st.rerun()
                    else:
                        st.error("Failed to clear chat history.")
                except Exception as e:
                    st.error(f"Error clearing history: {e}")
    
    # Display chat logs in ChatGPT style
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                # Display citations as visual pills
                badges = "".join([f'<span class="citation-pill">🏷️ {src}</span>' for src in message["sources"] if src])
                if badges:
                    st.markdown(f'<div style="margin-top: 8px;"><strong>Citations:</strong> {badges}</div>', unsafe_allow_html=True)

    # Prompt text area
    user_input = st.chat_input("Ask about the columns, null ratios, averages, etc...")
    if user_input:
        # 1. Immediately render User message locally
        with st.chat_message("user"):
            st.markdown(user_input)
            
        # 2. Call backend query in loading spinner block
        with st.spinner("Assistant is analyzing dataset stats..."):
            try:
                res = httpx.post(
                    f"{API_BASE_URL}/chat/query",
                    json={
                        "dataset_id": st.session_state.selected_dataset_id,
                        "session_id": st.session_state.active_chat_session_id,
                        "message": user_input
                    },
                    timeout=120.0 # Allow time for Ollama to run locally
                )
                if res.status_code == 200:
                    # Refresh history and rerun
                    fetch_chat_history(st.session_state.selected_dataset_id)
                    st.rerun()
                else:
                    st.error(f"Failed to get response: {res.json().get('detail', 'Inference failed.')}")
            except Exception as e:
                st.error(f"Communication error: {e}")

def main():
    # Inject Premium CSS Styling
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

        /* Set premium font */
        html, body, [class*="css"], .stApp {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        /* Glowing text gradient main header */
        .main-header {
            background: linear-gradient(135deg, #6366f1, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            font-size: 2.2rem;
            margin-top: 1rem;
            margin-bottom: 1.5rem;
            display: inline-block;
        }

        /* Premium glassmorphism cards for metrics */
        div[data-testid="metric-container"] {
            background-color: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 15px 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
            transition: all 0.3s ease;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(99, 102, 241, 0.08);
            border-color: rgba(99, 102, 241, 0.2);
        }

        /* Rounded buttons with hover scales */
        .stButton>button {
            border-radius: 8px !important;
            transition: all 0.2s ease;
            font-weight: 500;
        }
        .stButton>button:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15);
        }

        /* Chat input custom focus borders */
        .stChatInput textarea {
            border-radius: 12px !important;
            border: 1px solid rgba(99, 102, 241, 0.2) !important;
            transition: all 0.3s ease;
        }
        .stChatInput textarea:focus {
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
        }

        /* Visual pill styling for citations */
        .citation-pill {
            display: inline-block;
            padding: 3px 10px;
            margin: 4px 4px 4px 0px;
            font-size: 0.76rem;
            font-weight: 500;
            color: #3b82f6;
            background-color: rgba(59, 130, 246, 0.08);
            border: 1px solid rgba(59, 130, 246, 0.2);
            border-radius: 12px;
            text-decoration: none;
            transition: all 0.2s ease-in-out;
            cursor: default;
        }
        .citation-pill:hover {
            color: #ffffff;
            background-color: #3b82f6;
            transform: translateY(-1px);
            box-shadow: 0 4px 10px rgba(59, 130, 246, 0.3);
        }
        
        /* Tab items styling */
        button[data-baseweb="tab"] {
            font-size: 1rem !important;
            font-weight: 600 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">💡 Natural Language Data Profiling & Q&A Assistant</h1>', unsafe_allow_html=True)
    init_session_states()
    render_sidebar()
    
    tab1, tab2 = st.tabs(["📊 Data Profiling Report", "💬 Chat Q&A Workspace"])
    
    with tab1:
        render_dashboard_tab()
        
    with tab2:
        render_chat_tab()

if __name__ == "__main__":
    main()

