"""
SQLAlchemy relational schemas mapping database tables to Python objects.
Tables:
- datasets: Details of uploaded CSV data files.
- column_metadata: Profiling characteristics extracted for each column.
- chat_sessions: Conversational context scopes.
- chat_messages: History transcript entries.
"""

import datetime
from sqlalchemy import Column, String, Integer, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from db.session import Base

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    json_report_path = Column(String, nullable=True)
    html_report_path = Column(String, nullable=True)
    row_count = Column(Integer, nullable=True)
    col_count = Column(Integer, nullable=True)
    status = Column(String, default="PENDING")  # PENDING, PROFILED, FAILED
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    columns = relationship("ColumnMetadata", back_populates="dataset", cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", back_populates="dataset", cascade="all, delete-orphan")


class ColumnMetadata(Base):
    __tablename__ = "column_metadata"

    id = Column(String, primary_key=True, index=True)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=False)
    name = Column(String, nullable=False)
    data_type = Column(String, nullable=False)  # Numeric, Categorical, Datetime, etc.
    missing_count = Column(Integer, default=0)
    missing_pct = Column(Float, default=0.0)
    unique_count = Column(Integer, default=0)
    mean = Column(Float, nullable=True)
    min = Column(String, nullable=True)  # Store min as string to handle dates/categories
    max = Column(String, nullable=True)  # Store max as string to handle dates/categories
    correlation_summary = Column(Text, nullable=True)  # JSON-serialized correlation metrics

    # Relationships
    dataset = relationship("Dataset", back_populates="columns")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, index=True)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=False)
    title = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    dataset = relationship("Dataset", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)  # user, assistant
    content = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)  # JSON-serialized list of sources/citations
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    session = relationship("ChatSession", back_populates="messages")


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(String, primary_key=True, index=True)
    dataset_id = Column(String, nullable=True)  # Optional dataset scope association
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    citation = Column(Text, nullable=True)     # JSON-serialized list of sources/citations
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

