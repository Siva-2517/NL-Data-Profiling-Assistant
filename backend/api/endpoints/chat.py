"""
Chat API Endpoints.
Manages conversations, Q&A interactions, history retrieval,
and triggers the hybrid RAG engine pipeline.
Uses SQLite persistent chat_history schema.
"""

import uuid
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel

from db.session import get_db
from db.models import Dataset, ChatSession, ChatHistory
from db.chat_history_db import insert_chat, fetch_history, clear_history
from backend.services.rag_engine import RAGEngine

logger = logging.getLogger("backend_chat")
router = APIRouter()

# Pydantic schemas for request validation
class ChatSessionCreate(BaseModel):
    dataset_id: str
    title: str = "New Chat Session"

class ChatQueryRequest(BaseModel):
    dataset_id: str
    message: str
    session_id: Optional[str] = None  # Kept for backwards compatibility

@router.post("/sessions", status_code=status.HTTP_201_CREATED)
async def create_chat_session(payload: ChatSessionCreate, db: Session = Depends(get_db)):
    """
    Creates a new conversational chat session linked to a specific dataset.
    Kept for backwards compatibility.
    """
    logger.info(f"Creating chat session for dataset: {payload.dataset_id}")
    dataset = db.query(Dataset).filter(Dataset.id == payload.dataset_id).first()
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {payload.dataset_id} not found."
        )

    try:
        session_id = str(uuid.uuid4())
        new_session = ChatSession(
            id=session_id,
            dataset_id=payload.dataset_id,
            title=payload.title
        )
        db.add(new_session)
        db.commit()
        db.refresh(new_session)
        return {
            "session_id": session_id,
            "dataset_id": new_session.dataset_id,
            "title": new_session.title,
            "created_at": new_session.created_at
        }
    except Exception as e:
        logger.error(f"Failed to create chat session: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register chat session in database."
        )

@router.get("/sessions/{session_id}/history")
async def get_session_chat_history(session_id: str, db: Session = Depends(get_db)):
    """
    Retrieves history by session ID.
    Backwards compatibility: maps to fetching by the dataset associated with the session.
    """
    logger.info(f"Retrieving session-based history for: {session_id}")
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        # If session not found, return empty history
        return []
    return await get_dataset_chat_history(session.dataset_id, db)

@router.get("/history/{dataset_id}")
async def get_dataset_chat_history(dataset_id: str, db: Session = Depends(get_db)):
    """
    Retrieves the chronological list of Q&A interactions for a given dataset,
    formatting them as user/assistant messages for the Streamlit UI.
    """
    logger.info(f"Retrieving chat history for dataset: {dataset_id}")
    try:
        records = fetch_history(db, dataset_id)
        history = []
        for r in records:
            sources = []
            if r.citation:
                try:
                    sources = json.loads(r.citation)
                except Exception:
                    sources = [r.citation]
            # Split each Q&A record into user and assistant message bubbles
            history.append({
                "id": f"{r.id}_user",
                "role": "user",
                "content": r.question,
                "sources": [],
                "timestamp": r.timestamp
            })
            history.append({
                "id": f"{r.id}_assistant",
                "role": "assistant",
                "content": r.answer,
                "sources": sources,
                "timestamp": r.timestamp
            })
        return history
    except Exception as e:
        logger.error(f"Failed to query chat history: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to query chat history."
        )

@router.delete("/history/{dataset_id}", status_code=status.HTTP_200_OK)
async def clear_dataset_chat_history(dataset_id: str, db: Session = Depends(get_db)):
    """
    Clears all persistent chat history logs scoped to the target dataset.
    """
    logger.info(f"Clearing chat history for dataset: {dataset_id}")
    try:
        clear_history(db, dataset_id)
        return {"message": f"Chat history for dataset {dataset_id} cleared successfully."}
    except Exception as e:
        logger.error(f"Failed to clear chat history: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear chat history."
        )

@router.post("/query")
async def query_dataset(payload: ChatQueryRequest, db: Session = Depends(get_db)):
    """
    Core RAG orchestrator endpoint.
    Retrieves chat context, invokes hybrid routing services, compiles prompts,
    invokes LLM, records interaction in ChatHistory SQLite table, and returns response.
    """
    logger.info(f"Received query for dataset: {payload.dataset_id} - query: {payload.message}")
    
    # 1. Validate dataset existence in SQLite
    dataset = db.query(Dataset).filter(Dataset.id == payload.dataset_id).first()
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {payload.dataset_id} not found."
        )

    # 2. Invoke RAG engine to select route and generate answer
    try:
        rag_engine = RAGEngine()
        result = rag_engine.execute_query(
            dataset_id=payload.dataset_id,
            query=payload.message,
            db=db
        )
    except Exception as e:
        logger.error(f"Error during RAG execution: {e}", exc_info=True)
        result = {
            "answer": f"RAG inference execution error: {e}",
            "sources": ["system.error"]
        }

    # 3. Save Q&A transaction in ChatHistory table
    try:
        insert_chat(
            db=db,
            dataset_id=payload.dataset_id,
            question=payload.message,
            answer=result["answer"],
            citations=result["sources"]
        )
    except Exception as e:
        logger.error(f"Failed to save persistent chat history row: {e}", exc_info=True)
        # Non-fatal: return the result anyway

    return {
        "message_id": str(uuid.uuid4()),
        "dataset_id": payload.dataset_id,
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "timestamp": datetime.utcnow()
    }


