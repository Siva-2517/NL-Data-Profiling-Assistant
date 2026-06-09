"""
General helper functions shared across frontend and backend modules.
Includes file helpers, data cleanup routines, and JSON decorators.
"""

import os
import re
import json
from typing import Any, Dict, Optional

def sanitize_filename(filename: str) -> str:
    """
    Remove potentially dangerous path traversal characters and replace spaces.
    
    Args:
        filename (str): Original upload file name.
        
    Returns:
        str: Sanitized, system-safe file name.
    """
    # Keep only letters, digits, dots, dashes, and underscores
    clean_name = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    return clean_name

def format_file_size(size_bytes: int) -> str:
    """
    Converts raw byte volumes into human-readable string formats (KB, MB, GB).
    
    Args:
        size_bytes (int): Size in bytes.
        
    Returns:
        str: Formatted file size string.
    """
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = 0
    while size_bytes >= 1024 and i < len(size_name) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.2f} {size_name[i]}"

def clean_column_name(column_name: str) -> str:
    """
    Normalizes column headers (lowercasing, replacing special characters and spaces with underscores).
    
    Args:
        column_name (str): Original column name.
        
    Returns:
        str: Normalized column name.
    """
    # Lowercase and replace non-alphanumeric characters with underscores
    cleaned = re.sub(r'[^a-zA-Z0-9]', '_', column_name.lower())
    # Remove duplicate underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    return cleaned.strip('_')

def safe_json_load(json_str: Optional[str]) -> Dict[str, Any]:
    """
    Safely deserializes database JSON text fields.
    
    Args:
        json_str (Optional[str]): Serialized JSON string.
        
    Returns:
        Dict[str, Any]: Decoded dictionary or empty dict on failure/null.
    """
    if not json_str:
        return {}
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}
