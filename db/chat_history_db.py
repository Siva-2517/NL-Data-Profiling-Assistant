"""
Database transactions for persistent QA chat histories.
Provides functional helpers:
- insert_chat: Saves a Q&A interaction and citations to SQLite.
- fetch_history: Returns historical logs sorted chronologically.
- clear_history: Deletes history entries scoped to the target dataset.
"""

import uuid
import json
import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from db.models import ChatHistory

logger = logging.getLogger("backend_chat_history_db")

def insert_chat(
    db: Session, 
    dataset_id: str, 
    question: str, 
    answer: str, 
    citations: List[str]
) -> ChatHistory:
    """
    Creates and persists a new ChatHistory record in SQLite.
    
    Args:
        db (Session): Database transaction connection.
        dataset_id (str): UUID referencing the active dataset.
        question (str): Prompt text.
        answer (str): Response text.
        citations (List[str]): Citations list.
        
    Returns:
        ChatHistory: Saved model entity.
    """
    logger.info(f"Saving chat interaction to database for dataset: {dataset_id}")
    try:
        chat_id = str(uuid.uuid4())
        serialized_citations = json.dumps(citations)
        
        db_chat = ChatHistory(
            id=chat_id,
            dataset_id=dataset_id,
            question=question,
            answer=answer,
            citation=serialized_citations
        )
        db.add(db_chat)
        db.commit()
        db.refresh(db_chat)
        return db_chat
    except Exception as e:
        logger.error(f"Failed to insert chat history record: {e}", exc_info=True)
        db.rollback()
        raise RuntimeError(f"Database insertion failure: {e}")

def fetch_history(db: Session, dataset_id: str) -> List[ChatHistory]:
    """
    Queries and returns all persistent chat history logs scoped to a dataset.
    
    Args:
        db (Session): Database session.
        dataset_id (str): Target dataset filter key.
        
    Returns:
        List[ChatHistory]: Chronological history entries.
    """
    logger.info(f"Querying chat history records for dataset: {dataset_id}")
    try:
        history = (
            db.query(ChatHistory)
            .filter(ChatHistory.dataset_id == dataset_id)
            .order_by(ChatHistory.timestamp.asc())
            .all()
        )
        return history
    except Exception as e:
        logger.error(f"Failed to query chat history: {e}", exc_info=True)
        raise RuntimeError(f"Database query failure: {e}")

def clear_history(db: Session, dataset_id: str) -> None:
    """
    Deletes all persistent chat history entries matching the dataset scope.
    
    Args:
        db (Session): Database transaction session.
        dataset_id (str): Target dataset filter key.
    """
    logger.info(f"Clearing chat history records for dataset: {dataset_id}")
    try:
        db.query(ChatHistory).filter(ChatHistory.dataset_id == dataset_id).delete()
        db.commit()
        logger.info(f"Successfully cleared chat history for dataset {dataset_id}")
    except Exception as e:
        logger.error(f"Failed to clear chat history: {e}", exc_info=True)
        db.rollback()
        raise RuntimeError(f"Database deletion failure: {e}")
