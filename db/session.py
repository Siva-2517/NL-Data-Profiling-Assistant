"""
Database connection, session management, and generator dependencies.
Uses SQLAlchemy to link to SQLite DB defined in environmental config.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from backend.config import settings

DATABASE_URL = settings.DATABASE_URL

# SQLite connection requirements (specifically check for threading parameters)
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

# Create SQLite Database Engine
engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False
)

# Thread-local database session maker
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Declarative base class for SQLAlchemy models
Base = declarative_base()

def get_db():
    """
    FastAPI dependency generator.
    Yields active database session and ensures proper close-down.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
