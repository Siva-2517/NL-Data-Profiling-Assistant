"""
Datasets API Endpoints.
Handles CSV file uploads, background profiling trigger registration, list operations,
and generated profiling HTML reports delivery.
"""

import os
import uuid
import shutil
import logging
from typing import List
from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from db.session import get_db, SessionLocal
from db.models import Dataset, ColumnMetadata
from backend.config import settings
from backend.services.profiler import DataProfiler
from backend.services.vector_store import VectorStoreService
from utils.helpers import sanitize_filename

logger = logging.getLogger("backend_datasets")
router = APIRouter()

def run_profiling_and_indexing_task(dataset_id: str, file_path: str):
    """
    Background worker task.
    Runs ydata-profiling, extracts variables statistics, updates SQLite,
    and indexes schemas into ChromaDB.
    """
    logger.info(f"Starting background profiling for dataset_id: {dataset_id}")
    db = SessionLocal()
    try:
        # Generate target output paths
        html_filename = f"{dataset_id}.html"
        json_filename = f"{dataset_id}.json"
        html_path = os.path.join(settings.PROFILES_DIR, html_filename)
        json_path = os.path.join(settings.PROFILES_DIR, json_filename)

        # 1. Run ydata-profiling report generation
        report_meta = DataProfiler.generate_profile_report(file_path, html_path, json_path)
        
        # 2. Extract column metrics
        columns_stats = DataProfiler.extract_column_statistics(json_path)

        # 3. Update Dataset object in SQLite database
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            logger.error(f"Dataset {dataset_id} not found in database during background task.")
            return

        dataset.row_count = report_meta["row_count"]
        dataset.col_count = report_meta["col_count"]
        dataset.html_report_path = html_path
        dataset.json_report_path = json_path
        
        # 4. Insert Column statistics
        db_cols = []
        for col in columns_stats:
            col_id = str(uuid.uuid4())
            db_col = ColumnMetadata(
                id=col_id,
                dataset_id=dataset_id,
                name=col["name"],
                data_type=col["data_type"],
                missing_count=col["missing_count"],
                missing_pct=col["missing_pct"],
                unique_count=col["unique_count"],
                mean=col["mean"],
                min=col["min"],
                max=col["max"],
                correlation_summary=None
            )
            db_cols.append(db_col)
            
        db.bulk_save_objects(db_cols)
        
        # 5. Populate ChromaDB Vector Store with schemas
        try:
            vector_store = VectorStoreService()
            vector_store.index_dataset_metadata(dataset_id, columns_stats)
        except Exception as vec_err:
            logger.warning(f"Vector indexing failed: {vec_err}", exc_info=True)
            # We don't fail the whole job if vector store indexing fails, but let's log it

        dataset.status = "PROFILED"
        db.commit()
        logger.info(f"Background profiling successfully finished for dataset: {dataset_id}")
        
    except Exception as err:
        logger.error(f"Error occurred during profiling background job for {dataset_id}: {err}", exc_info=True)
        db.rollback()
        # Mark dataset as FAILED in database
        try:
            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if dataset:
                dataset.status = "FAILED"
                db.commit()
        except Exception as update_err:
            logger.error(f"Failed to update dataset status to FAILED: {update_err}")
    finally:
        db.close()

@router.post("/", status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Accepts multipart CSV file upload, registers metadata records,
    and queues background tasks for profiling and indexing.
    """
    logger.info(f"Received dataset upload request for file: {file.filename}")
    
    # 1. Enforce file extension validation (.csv)
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file extension. Only .csv files are permitted."
        )

    # 2. Setup storage target paths
    dataset_id = str(uuid.uuid4())
    filename = sanitize_filename(file.filename)
    file_path = os.path.join(settings.UPLOAD_DIR, f"{dataset_id}_{filename}")
    
    # Save file contents
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file on server."
        )

    # 3. Create initial SQLite Dataset record with status = "PENDING"
    try:
        db_dataset = Dataset(
            id=dataset_id,
            filename=filename,
            file_path=file_path,
            status="PENDING"
        )
        db.add(db_dataset)
        db.commit()
        db.refresh(db_dataset)
    except Exception as e:
        logger.error(f"Database error registering upload: {e}", exc_info=True)
        # Attempt cleanup of saved file
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register dataset upload in database."
        )

    # 4. Trigger background profiling tasks
    background_tasks.add_task(run_profiling_and_indexing_task, dataset_id, file_path)
    
    return {
        "message": "Dataset uploaded successfully. Profiling job has been queued.",
        "dataset_id": dataset_id,
        "filename": filename,
        "status": "PENDING"
    }

@router.get("/")
async def list_datasets(db: Session = Depends(get_db)):
    """
    Lists all uploaded datasets registered in SQLite.
    """
    logger.info("Listing all datasets...")
    try:
        datasets = db.query(Dataset).order_by(Dataset.created_at.desc()).all()
        return datasets
    except Exception as e:
        logger.error(f"Failed to query datasets list: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database query error."
        )

@router.get("/{dataset_id}")
async def get_dataset_details(dataset_id: str, db: Session = Depends(get_db)):
    """
    Fetches detailed metadata and column statistics of a specific dataset.
    """
    logger.info(f"Fetching details for dataset_id: {dataset_id}")
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
        
    return {
        "id": dataset.id,
        "filename": dataset.filename,
        "row_count": dataset.row_count,
        "col_count": dataset.col_count,
        "status": dataset.status,
        "created_at": dataset.created_at,
        "columns": [
            {
                "name": col.name,
                "data_type": col.data_type,
                "missing_count": col.missing_count,
                "missing_pct": col.missing_pct,
                "unique_count": col.unique_count,
                "mean": col.mean,
                "min": col.min,
                "max": col.max
            } for col in dataset.columns
        ]
    }

@router.get("/{dataset_id}/profile")
async def get_dataset_profile_report(dataset_id: str, db: Session = Depends(get_db)):
    """
    Serves the pre-compiled HTML profile report generated by ydata-profiling.
    """
    logger.info(f"Serving ydata-profiling HTML report for dataset_id: {dataset_id}")
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
        
    if dataset.status != "PROFILED" or not dataset.html_report_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Profile report is not available for this dataset. Status: {dataset.status}"
        )
        
    if not os.path.exists(dataset.html_report_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="HTML report file not found on disk."
        )
        
    return FileResponse(dataset.html_report_path, media_type="text/html")

@router.delete("/{dataset_id}", status_code=status.HTTP_200_OK)
async def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """
    Deletes the dataset file, profiling reports, SQLite columns, and Chroma vector index.
    """
    logger.info(f"Request to delete dataset_id: {dataset_id}")
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
        
    # Remove files on disk (CSV, HTML, JSON report)
    for path in [dataset.file_path, dataset.html_report_path, dataset.json_report_path]:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                logger.warning(f"Failed to delete file {path}: {e}")

    try:
        # Purge SQLite records (cascading columns list)
        db.delete(dataset)
        db.commit()
    except Exception as db_err:
        logger.error(f"Failed to purge database records: {db_err}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete database records."
        )

    # Delete from vector store
    try:
        vector_store = VectorStoreService()
        vector_store.delete_dataset_collection(dataset_id)
    except Exception as e:
        logger.warning(f"Failed to clear vector store collection: {e}")
        
    return {"message": f"Dataset {dataset_id} and related artifacts deleted successfully."}

