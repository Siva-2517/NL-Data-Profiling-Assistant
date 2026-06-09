"""
ydata-profiling JSON Parser Service.
Loads, parses, and normalizes statistical results into structured schemas.
Includes prompt converters for LLM (RAG) prompts and data query helpers.
"""

import os
import json
import logging
from typing import Dict, Any, List, Union, Optional, Tuple

logger = logging.getLogger("backend_profiler_parser")

class ProfilerJSONParser:
    """
    Robust parser for ydata-profiling JSON report outputs.
    Parses and extracts metadata, statistical variables, correlations, and duplicates.
    Provides RAG-friendly prompt descriptors and quick queries for structural data.
    """
    
    _cache = {}

    def __new__(cls, json_path_or_dict: Union[str, Dict[str, Any]], *args, **kwargs):
        if isinstance(json_path_or_dict, str):
            if json_path_or_dict in cls._cache:
                logger.info(f"Retrieving cached parsed profile for: {json_path_or_dict}")
                return cls._cache[json_path_or_dict]
        return super().__new__(cls)

    def __init__(self, json_path_or_dict: Union[str, Dict[str, Any]]):
        """
        Initializes parser with JSON report path or preloaded dictionary.
        
        Args:
            json_path_or_dict (str/dict): Path to file or raw report dict.
        """
        if hasattr(self, "raw_data") and self.raw_data:
            return

        self.raw_data = {}
        if isinstance(json_path_or_dict, str):
            if not os.path.exists(json_path_or_dict):
                logger.error(f"JSON report file does not exist: {json_path_or_dict}")
                raise FileNotFoundError(f"JSON report file not found: {json_path_or_dict}")
            try:
                with open(json_path_or_dict, "r", encoding="utf-8") as f:
                    self.raw_data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load JSON profiling file: {e}", exc_info=True)
                raise ValueError(f"Failed to parse profiling JSON: {e}")
        elif isinstance(json_path_or_dict, dict):
            self.raw_data = json_path_or_dict
        else:
            raise TypeError("Argument must be a file path string or dictionary.")

        # Trigger normalization parse operations
        self.normalized_data = self._normalize()

        # Cache instance if loaded from file path
        if isinstance(json_path_or_dict, str):
            self.__class__._cache[json_path_or_dict] = self

    def _normalize(self) -> Dict[str, Any]:
        """Runs parsers to compile unified structural profiles."""
        summary = self._parse_table_summary()
        columns = self._parse_columns(summary.get("row_count", 0))
        correlations = self._parse_correlations()
        duplicates = self._parse_duplicates(summary.get("row_count", 0))

        return {
            "summary": summary,
            "columns": columns,
            "correlations": correlations,
            "duplicates": duplicates
        }

    def _parse_table_summary(self) -> Dict[str, Any]:
        """Extracts dataset-level metrics."""
        table = self.raw_data.get("table", {})
        row_count = int(table.get("n", 0))
        col_count = int(table.get("n_var", 0))
        missing_cells = int(table.get("n_cells_missing", 0))
        p_cells_missing = float(table.get("p_cells_missing", 0.0))
        
        return {
            "row_count": row_count,
            "col_count": col_count,
            "missing_cells": missing_cells,
            "missing_cells_pct": p_cells_missing * 100.0
        }

    def _parse_columns(self, row_count: int) -> Dict[str, Dict[str, Any]]:
        """Extracts and normalizes column statistics."""
        variables = self.raw_data.get("variables", {})
        parsed_cols = {}

        for col_name, stats in variables.items():
            # Get missing values properties
            missing_count = int(stats.get("n_missing", 0))
            p_missing = stats.get("p_missing", 0.0)
            missing_pct = float(p_missing) * 100.0
            
            # Clean mean value
            mean_val = stats.get("mean", None)
            if mean_val is not None:
                try:
                    mean_val = float(mean_val)
                except (ValueError, TypeError):
                    mean_val = None

            # Get min/max
            min_val = stats.get("min", None)
            max_val = stats.get("max", None)
            
            parsed_cols[col_name] = {
                "name": col_name,
                "type": str(stats.get("type", "Unknown")),
                "missing": {
                    "count": missing_count,
                    "pct": missing_pct
                },
                "distinct": {
                    "count": int(stats.get("n_distinct", 0)),
                    "is_unique": bool(stats.get("is_unique", False))
                },
                "mean": mean_val,
                "min": min_val,
                "max": max_val
            }
        return parsed_cols

    def _parse_correlations(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Extracts correlation matrix structures from raw reports."""
        raw_correlations = self.raw_data.get("correlations", {})
        parsed_corr = {}
        for method, matrix in raw_correlations.items():
            if not isinstance(matrix, dict):
                continue
            parsed_corr[method] = matrix
        return parsed_corr

    def _parse_duplicates(self, row_count: int) -> Dict[str, Any]:
        """Extracts duplicates row information."""
        table = self.raw_data.get("table", {})
        dup_count = int(table.get("n_duplicates", 0))
        dup_pct = (dup_count / row_count * 100.0) if row_count > 0 else 0.0
        return {
            "count": dup_count,
            "pct": dup_pct
        }

    def get_normalized_profile(self) -> Dict[str, Any]:
        """Returns the normalized dictionary structure."""
        return self.normalized_data

    # ==========================================
    # RAG Optimization Prompts Generation
    # ==========================================

    def get_dataset_summary_text(self) -> str:
        """
        Creates a descriptive paragraph summarizing the general characteristics of the dataset.
        Optimized for prompt context insertion.
        """
        summ = self.normalized_data["summary"]
        dups = self.normalized_data["duplicates"]
        cols = list(self.normalized_data["columns"].keys())
        
        desc = (
            f"Dataset Summary Details:\n"
            f"- Variables / Columns: {summ['col_count']} columns\n"
            f"- Rows: {summ['row_count']} rows\n"
            f"- Total Cells Missing: {summ['missing_cells']} cells ({summ['missing_cells_pct']:.2f}% null ratio)\n"
            f"- Duplicate Rows: {dups['count']} duplicate records ({dups['pct']:.2f}% duplicate ratio)\n"
            f"- Column Names: {', '.join(cols)}\n"
        )
        return desc

    def get_column_summary_text(self, column_name: str) -> str:
        """
        Creates a descriptive statistical profile of a single column variable.
        Optimized for prompt context insertion.
        """
        col = self.normalized_data["columns"].get(column_name)
        if not col:
            return f"Column '{column_name}' was not found in this dataset."
            
        desc = (
            f"Column '{column_name}' Profile details:\n"
            f"- Data Type: {col['type']}\n"
            f"- Missing values: {col['missing']['count']} nulls ({col['missing']['pct']:.2f}% missing)\n"
            f"- Distinct values: {col['distinct']['count']} unique items (Is unique: {col['distinct']['is_unique']})\n"
        )
        if col["mean"] is not None:
            desc += f"- Statistical Mean average: {col['mean']:.4f}\n"
        if col["min"] is not None:
            desc += f"- Minimum value boundary: {col['min']}\n"
        if col["max"] is not None:
            desc += f"- Maximum value boundary: {col['max']}\n"
            
        return desc

    def get_top_correlations(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Identifies column pairs exhibiting absolute correlation values >= threshold.
        Avoids duplicate pairings (A-B vs B-A) and self-correlation.
        """
        top_corrs = []
        # Try to fetch Pearson or Auto correlations matrices
        corr_methods = ["auto", "pearson", "spearman"]
        matrix = {}
        selected_method = "unknown"
        
        for m in corr_methods:
            if m in self.normalized_data["correlations"]:
                matrix = self.normalized_data["correlations"][m]
                selected_method = m
                break
                
        if not matrix:
            return []

        processed_pairs = set()
        
        for var1, correlations in matrix.items():
            if not isinstance(correlations, dict):
                continue
            for var2, val in correlations.items():
                if var1 == var2:
                    continue
                if val is None:
                    continue
                try:
                    coeff = float(val)
                except (ValueError, TypeError):
                    continue

                if abs(coeff) >= threshold:
                    # Sort names alphabetically to prevent duplicate pairing entries (A-B vs B-A)
                    pair = tuple(sorted([var1, var2]))
                    if pair not in processed_pairs:
                        processed_pairs.add(pair)
                        top_corrs.append({
                            "var1": var1,
                            "var2": var2,
                            "coefficient": coeff,
                            "method": selected_method
                        })
                        
        # Sort descending by absolute strength
        top_corrs.sort(key=lambda x: abs(x["coefficient"]), reverse=True)
        return top_corrs

    def get_correlation_summary_text(self, threshold: float = 0.5) -> str:
        """
        Creates a list description of high correlations across variables.
        Optimized for prompt context insertion.
        """
        corrs = self.get_top_correlations(threshold)
        if not corrs:
            return f"No columns exhibit linear correlations above threshold absolute value {threshold}."

        desc = f"Correlated Variable Pairs (Absolute value >= {threshold}):\n"
        for c in corrs:
            strength = "Strong" if abs(c['coefficient']) >= 0.7 else "Moderate"
            direction = "positive" if c['coefficient'] > 0 else "negative"
            desc += f"- Columns '{c['var1']}' and '{c['var2']}' show a {strength} {direction} correlation ({c['coefficient']:.3f} using {c['method']})\n"
        return desc

    # ==========================================
    # Statistical Query Helpers
    # ==========================================

    def query_column_stat(self, column_name: str, stat_name: str) -> Any:
        """
        Retrieves a target stat attribute for a given column.
        
        Args:
            column_name (str): Column variable name.
            stat_name (str): Stat attribute (type, missing, mean, min, max, distinct).
            
        Returns:
            Any: Value of stat or None if column or stat is invalid.
        """
        col = self.normalized_data["columns"].get(column_name)
        if not col:
            return None
            
        # Support dotted keys (e.g. 'missing.pct' or 'distinct.count')
        if "." in stat_name:
            parent_key, child_key = stat_name.split(".", 1)
            parent = col.get(parent_key)
            if isinstance(parent, dict):
                return parent.get(child_key)
            return None
            
        return col.get(stat_name)

    def get_columns_by_type(self, data_type: str) -> List[str]:
        """
        Finds columns of matching data types.
        
        Args:
            data_type (str): Type to filter (e.g., 'Numeric', 'Categorical', 'Text', 'DateTime').
            
        Returns:
            List[str]: List of column names.
        """
        columns = self.normalized_data["columns"]
        return [name for name, val in columns.items() if val["type"].lower() == data_type.lower()]

    def get_high_missing_columns(self, threshold_pct: float = 10.0) -> List[Dict[str, Any]]:
        """
        Finds columns with missing rates exceeding threshold_pct.
        
        Args:
            threshold_pct (float): Percentage threshold boundary (0-100).
            
        Returns:
            List[Dict[str, Any]]: List containing col name and missing percentages.
        """
        columns = self.normalized_data["columns"]
        high_missing = []
        for name, val in columns.items():
            pct = val["missing"]["pct"]
            if pct >= threshold_pct:
                high_missing.append({
                    "column": name,
                    "missing_pct": pct,
                    "missing_count": val["missing"]["count"]
                })
        # Sort descending by missing percentage
        high_missing.sort(key=lambda x: x["missing_pct"], reverse=True)
        return high_missing
