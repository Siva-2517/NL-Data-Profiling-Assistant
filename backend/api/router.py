"""
Primary routing file aggregating endpoint-specific sub-routers.
Includes routes for:
- datasets: Upload, status listing, deletes
- chat: Query execution, session tracking
"""

from fastapi import APIRouter
from backend.api.endpoints.datasets import router as datasets_router
from backend.api.endpoints.chat import router as chat_router

api_router = APIRouter()

# Register sub-routers
api_router.include_router(datasets_router, prefix="/datasets", tags=["Datasets"])
api_router.include_router(chat_router, prefix="/chat", tags=["Chat & Inference"])
