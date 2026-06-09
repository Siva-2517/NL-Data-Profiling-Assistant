"""
Profiling Service.
Utilizes pandas and ydata-profiling to generate descriptive statistical dashboards.
Extracts metrics and correlation indexes from generated JSON reports to save to SQLite database columns.
"""

import json
import logging
import pandas as pd
from typing import Dict, Any, List

logger = logging.getLogger("backend_profiler_service")

class DataProfiler:
    @staticmethod
    def generate_profile_report(
        csv_path: str, 
        output_html_path: str, 
        output_json_path: str
    ) -> Dict[str, Any]:
        """
        Executes ydata-profiling analysis report over the target CSV file.
        Saves report as both visual HTML and data-interoperable JSON formats.
        
        Args:
            csv_path (str): Location of uploaded raw CSV file.
            output_html_path (str): Target path to write HTML report.
            output_json_path (str): Target path to write JSON report.
            
        Returns:
            Dict[str, Any]: Basic dataset shape information (row_count, col_count).
        """
        logger.info(f"Generating profiling report for {csv_path}...")
        
        # Load CSV using pandas with encoding fallback
        try:
            logger.info("Loading CSV file via pandas...")
            try:
                df = pd.read_csv(csv_path, encoding="utf-8")
            except UnicodeDecodeError:
                logger.warning("UTF-8 decoding failed, falling back to Latin-1 encoding.")
                df = pd.read_csv(csv_path, encoding="latin-1")
        except Exception as e:
            logger.error(f"Failed to read CSV file: {e}", exc_info=True)
            raise ValueError(f"Unable to read CSV file: {e}")

        if df.empty:
            logger.error("Uploaded CSV file is empty.")
            raise ValueError("The uploaded CSV file does not contain any data rows.")

        row_count = int(df.shape[0])
        col_count = int(df.shape[1])
        logger.info(f"Loaded DataFrame with shape: {row_count} rows, {col_count} columns.")

        # Run profiling using ydata-profiling (minimal=True for performance)
        try:
            logger.info("Invoking ydata-profiling Report...")
            from ydata_profiling import ProfileReport
            
            profile = ProfileReport(df, title="Dataset Profiling Report", minimal=True)
            
            logger.info(f"Saving HTML profile report to {output_html_path}...")
            profile.to_file(output_html_path)
            
            logger.info(f"Saving JSON profile report to {output_json_path}...")
            profile.to_file(output_json_path)
            
            logger.info("ydata-profiling report generation complete.")
        except Exception as e:
            logger.error(f"Error during ydata-profiling execution: {e}", exc_info=True)
            raise RuntimeError(f"Data profiling failed: {e}")
        
        return {
            "row_count": row_count,
            "col_count": col_count
        }

    @staticmethod
    def extract_column_statistics(json_report_path: str) -> List[Dict[str, Any]]:
        """
        Parses compiled ydata-profiling JSON report, extracting tabular variables details
        like averages, datatypes, unique categories, and missing percentages.
        
        Args:
            json_report_path (str): Path to generated JSON statistics file.
            
        Returns:
            List[Dict[str, Any]]: List of dictionary mappings representing column characteristics.
        """
        logger.info(f"Extracting column metrics from {json_report_path} using ProfilerJSONParser...")
        
        from backend.services.profiler_parser import ProfilerJSONParser
        
        try:
            parser = ProfilerJSONParser(json_report_path)
            normalized = parser.get_normalized_profile()
            columns = normalized.get("columns", {})
        except Exception as e:
            logger.error(f"Failed to parse column statistics: {e}", exc_info=True)
            raise ValueError(f"Unable to extract statistics: {e}")

        columns_stats = []

        for col_name, col in columns.items():
            col_meta = {
                "name": col_name,
                "data_type": col["type"],
                "missing_count": col["missing"]["count"],
                "missing_pct": col["missing"]["pct"],
                "unique_count": col["distinct"]["count"],
                "mean": col["mean"],
                # min and max are cast to string for database column boundaries matching
                "min": str(col["min"]) if col["min"] is not None else None,
                "max": str(col["max"]) if col["max"] is not None else None,
                "correlation_summary": None
            }
            columns_stats.append(col_meta)

        logger.info(f"Successfully parsed metrics for {len(columns_stats)} columns via parser service.")
        return columns_stats

