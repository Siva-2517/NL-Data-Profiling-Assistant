"""
FastAPI Server Entrypoint.
Responsible for:
- Initializing database schemas.
- Configuring request routers and logging structures.
- Registering middleware (CORS) and error handling blocks.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.logging_config import setup_logging
from backend.api.router import api_router
from db.session import engine
from db.models import Base

# Setup logger before starting the app
setup_logging()
logger = logging.getLogger("backend_app")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Life-cycle context manager handling application setup and shutdown tasks.
    """
    logger.info("Starting up FastAPI application...")
    
    # Auto-initialize database tables in SQLite
    try:
        logger.info("Initializing relational database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database schemas created successfully.")
    except Exception as e:
        logger.error(f"Database initialization failure: {e}", exc_info=True)
        
    yield
    
    logger.info("Shutting down FastAPI application...")

app = FastAPI(
    title="Data Profiling Assistant API",
    description="FastAPI Backend for NL Data Profiling Q&A Assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS Middleware allowing local frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for specific security requirements in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API v1 router endpoints
app.include_router(api_router, prefix="/api/v1")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catches all unhandled exceptions and formats a clean JSON error response.
    """
    logger.error(f"Unhandled error occurred: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please contact the administrator."}
    )

@app.get("/health", tags=["System"])
async def health_check():
    """Simple API health check endpoint."""
    return {"status": "ok", "environment": settings.APP_ENV}
