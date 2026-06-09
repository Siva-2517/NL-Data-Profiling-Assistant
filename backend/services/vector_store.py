"""
Vector Store Service.
Responsible for managing ChromaDB collections, computing embeddings of columns and schemas,
and performing similarity semantic searches over unstructured metadata.
"""

import logging
import chromadb
from typing import List, Dict, Any
from backend.config import settings

logger = logging.getLogger("backend_vector_store")

class VectorStoreService:
    _client = None

    def __init__(self, embedding_function: Any = None):
        """Initializes connection to the ChromaDB client."""
        if VectorStoreService._client is None:
            logger.info(f"Connecting to ChromaDB at: {settings.CHROMA_DB_PATH}")
            try:
                VectorStoreService._client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB Persistent Client: {e}", exc_info=True)
                raise RuntimeError(f"ChromaDB startup failure: {e}")
        self.client = VectorStoreService._client
        self.embedding_function = embedding_function

    def index_dataset_metadata(self, dataset_id: str, columns: List[Dict[str, Any]]) -> None:
        """
        Creates semantic text chunks from column metadata, embeds them,
        and saves them into a dataset-specific vector collection.
        
        Args:
            dataset_id (str): UUID referencing the active dataset.
            columns (List[Dict[str, Any]]): Struct of column characteristics to index.
        """
        collection_name = f"dataset_{dataset_id.replace('-', '_')}"
        logger.info(f"Indexing column metadata for dataset {dataset_id} in ChromaDB collection '{collection_name}'...")
        
        try:
            # Drop collection if it already exists to prevent duplication
            try:
                self.client.delete_collection(name=collection_name)
                logger.info(f"Cleared existing collection '{collection_name}' before re-indexing.")
            except Exception:
                pass

            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            
            documents = []
            metadatas = []
            ids = []
            
            for col in columns:
                col_name = col["name"]
                col_type = col.get("data_type", "Unknown")
                missing_pct = col.get("missing_pct", 0.0)
                missing_count = col.get("missing_count", 0)
                unique_count = col.get("unique_count", 0)
                mean_val = col.get("mean")
                min_val = col.get("min")
                max_val = col.get("max")
                
                # Construct a descriptive text paragraph representing the column variables and stats
                desc = (
                    f"Column name: '{col_name}'\n"
                    f"Data Type: {col_type}\n"
                    f"Missing / Null ratio: {missing_pct:.2f}% ({missing_count} missing rows)\n"
                    f"Cardinality / Unique values: {unique_count} distinct items\n"
                )
                if mean_val is not None:
                    desc += f"Statistical Mean average: {mean_val:.4f}\n"
                if min_val is not None:
                    desc += f"Minimum value boundary: {min_val}\n"
                if max_val is not None:
                    desc += f"Maximum value boundary: {max_val}\n"
                    
                documents.append(desc)
                metadatas.append({
                    "column_name": col_name,
                    "type": col_type,
                    "dataset_id": dataset_id
                })
                ids.append(f"{dataset_id}_{col_name}")
                
            if documents:
                logger.info(f"Adding {len(documents)} column vectors to collection '{collection_name}'...")
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info("ChromaDB vector indexing completed successfully.")
            else:
                logger.warning("No column documents prepared for indexing.")
                
        except Exception as e:
            logger.error(f"Failed to index dataset metadata in ChromaDB: {e}", exc_info=True)
            raise RuntimeError(f"ChromaDB indexing error: {e}")

    def search_relevant_columns(self, dataset_id: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB collection to retrieve columns semantically related to query.
        
        Args:
            dataset_id (str): UUID referencing the dataset.
            query (str): Natural language question.
            top_k (int): Maximum matched chunks.
            
        Returns:
            List[Dict[str, Any]]: Matched column metadata elements with similarity scores.
        """
        collection_name = f"dataset_{dataset_id.replace('-', '_')}"
        logger.info(f"Querying collection '{collection_name}' with semantic search query: '{query}'")
        
        try:
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            results = collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            hits = []
            if results and "documents" in results and results["documents"]:
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)
                ids = results["ids"][0]
                
                for doc, meta, dist, id_ in zip(docs, metas, distances, ids):
                    hits.append({
                        "id": id_,
                        "document": doc,
                        "metadata": meta,
                        "distance": float(dist)
                    })
            return hits
            
        except Exception as e:
            logger.error(f"Error querying ChromaDB collection '{collection_name}': {e}", exc_info=True)
            return []

    def delete_dataset_collection(self, dataset_id: str) -> None:
        """
        Deletes vector collection.
        
        Args:
            dataset_id (str): UUID of the dataset collection to purge.
        """
        collection_name = f"dataset_{dataset_id.replace('-', '_')}"
        logger.info(f"Deleting ChromaDB collection '{collection_name}'...")
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Collection '{collection_name}' deleted successfully.")
        except Exception as e:
            logger.warning(f"Failed to delete ChromaDB collection '{collection_name}' (it may not exist): {e}")

