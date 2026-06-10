"""
Hybrid RAG Engine.
Orchestrates routing configurations between database statistics query (JSON/SQLite tables)
and semantic similarity matches (ChromaDB Vector embeddings).
Integrates context structures into LangChain prompts and requests inference from LLMClient.
"""

import re
import json
import logging
from typing import Dict, Any, List
from sqlalchemy.orm import Session

from langchain_core.messages import HumanMessage as LcHumanMessage, AIMessage as LcAIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from backend.services.llm_client import LLMClient
from backend.services.vector_store import VectorStoreService
from backend.services.profiler_parser import ProfilerJSONParser
from db.models import Dataset, ChatHistory

logger = logging.getLogger("backend_rag_engine")

class RAGEngine:
    def __init__(self):
        """Initializes client configurations for RAG modules."""
        self.llm_client = LLMClient()
        self.vector_store = VectorStoreService()

    def determine_route(self, query: str) -> str:
        """
        Assesses if a query has statistical query intents vs semantic discovery intents.
        Routes to STRUCTURED_METRICS if factual keywords are found, else SEMANTIC_SEARCH.
        
        Args:
            query (str): User natural language prompt.
            
        Returns:
            str: Target route ("STRUCTURED_METRICS" or "SEMANTIC_SEARCH").
        """
        logger.info(f"Routing query: '{query}'")
        query_lower = query.lower()
        
        # Keywords suggesting exact statistical metrics, shapes, categories, or types
        factual_pattern = (
            r"\b(mean|average|min|max|minimum|maximum|null|missing|nan|blank|empty|"
            r"duplicate|distinct|unique|type|datatype|cardinality|correlation|std|"
            r"variance|rows|columns|shape|count|sum|avg|percentage|pct)\b"
        )
        
        if re.search(factual_pattern, query_lower):
            logger.info("Routed to: STRUCTURED_METRICS")
            return "STRUCTURED_METRICS"
        else:
            logger.info("Routed to: SEMANTIC_SEARCH")
            return "SEMANTIC_SEARCH"

    def retrieve_context(
        self, 
        dataset_id: str, 
        query: str, 
        route: str, 
        db: Session
    ) -> Dict[str, Any]:
        """
        Retrieves context variables from SQLite columns metadata or ChromaDB vectors.
        
        Args:
            dataset_id (str): UUID referencing the active dataset.
            query (str): User query.
            route (str): Resolved context target route.
            db (Session): Database session.
            
        Returns:
            Dict[str, Any]: Retreived context blocks and source references.
        """
        logger.info(f"Retrieving context for route: {route} on dataset {dataset_id}")
        
        # 1. Fetch dataset information from SQLite
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            logger.error(f"Dataset {dataset_id} not found in database during context retrieval.")
            return {"text": "No dataset metadata found on server.", "sources": []}
            
        context_texts = []
        sources = []

        # 2. Extract context according to routed pipeline
        if route == "STRUCTURED_METRICS":
            # Load ydata-profiling JSON summary statistics using Parser
            try:
                parser = ProfilerJSONParser(dataset.json_report_path)
                
                # Check if any columns are explicitly mentioned in the query text
                col_names = [col.name for col in dataset.columns]
                mentioned_cols = []
                for name in col_names:
                    pattern = r"\b" + re.escape(name.lower()) + r"\b"
                    if re.search(pattern, query.lower()):
                        mentioned_cols.append(name)
                        
                if mentioned_cols:
                    logger.info(f"Targeting columns: {mentioned_cols}")
                    # Include global dataset summary plus targeted columns stats
                    context_texts.append(parser.get_dataset_summary_text())
                    sources.append("dataset_summary")
                    for col in mentioned_cols:
                        context_texts.append(parser.get_column_summary_text(col))
                        sources.append(col)
                else:
                    logger.info("No specific columns mentioned. Fetching general statistics summary.")
                    context_texts.append(parser.get_dataset_summary_text())
                    sources.append("dataset_summary")
                    
                    # If correlation is mentioned, add correlation details
                    if "correlation" in query.lower():
                        context_texts.append(parser.get_correlation_summary_text(threshold=0.3))
                        sources.append("correlations")
            except Exception as e:
                logger.error(f"Failed to load JSON statistics: {e}", exc_info=True)
                context_texts.append("Failed to retrieve structured database metrics.")
                
        else:
            # Query ChromaDB collection for semantic similarity matches
            try:
                # Always prepends general dataset shape summary
                try:
                    parser = ProfilerJSONParser(dataset.json_report_path)
                    context_texts.append(parser.get_dataset_summary_text())
                    sources.append("dataset_summary")
                except Exception:
                    pass
                    
                hits = self.vector_store.search_relevant_columns(dataset_id, query, top_k=3)
                for hit in hits:
                    context_texts.append(hit["document"])
                    sources.append(hit["metadata"]["column_name"])
                    
                logger.info(f"Matched {len(hits)} semantic columns.")
            except Exception as e:
                logger.error(f"Failed to query semantic vector store: {e}", exc_info=True)
                context_texts.append("Failed to retrieve semantic vector metadata.")
        # 3. Load sample/full CSV rows for row-level Q&A support
        try:
            import os
            if dataset.file_path and os.path.exists(dataset.file_path):
                import pandas as pd
                df = pd.read_csv(dataset.file_path)
                row_count = len(df)
                if row_count <= 150:
                    # Include full dataset
                    csv_text = df.to_csv(index=False)
                    context_texts.append(f"Full Dataset CSV Rows (Total {row_count}):\n{csv_text}")
                else:
                    # Include first 15 rows preview
                    preview_text = df.head(15).to_csv(index=False)
                    context_texts.append(f"Dataset Rows Preview (First 15 of {row_count}):\n{preview_text}")
                    
                    # Filter matching rows dynamically based on query terms
                    query_words = [w.strip() for w in re.split(r'\W+', query.lower()) if len(w.strip()) > 2]
                    stopwords = {
                        "mean", "average", "min", "max", "rows", "columns", "count", "null", "missing", 
                        "nan", "blank", "empty", "what", "where", "who", "tell", "show", "list", "name", 
                        "whose", "that", "this", "employee", "dataset", "value", "values", "the", "and"
                    }
                    search_words = [word for word in query_words if word not in stopwords]
                    
                    if search_words:
                        mask = pd.Series(False, index=df.index)
                        for word in search_words:
                            for col in df.columns:
                                mask |= df[col].astype(str).str.lower().str.contains(word, na=False)
                        matching_rows = df[mask]
                        if not matching_rows.empty:
                            matched_csv = matching_rows.head(50).to_csv(index=False)
                            context_texts.append(f"Matching Dataset Rows for query terms {search_words}:\n{matched_csv}")
                
                sources.append("raw_csv_data")
        except Exception as csv_err:
            logger.warning(f"Failed to load raw CSV data into context: {csv_err}")

        context_text = "\n\n".join(context_texts)
        return {
            "text": context_text,
            "sources": list(set(sources))
        }

    def execute_query(self, dataset_id: str, query: str, db: Session, session_id: Any = None) -> Dict[str, Any]:
        """
        Coordinates full QA workflow: routing, retrieving context, compiling memory history,
        executing the LangChain prompt template, and extracting structured answer/citations.
        
        Args:
            dataset_id (str): Dataset id linked to the query.
            query (str): Question prompt.
            db (Session): Database session.
            session_id (Any, optional): Kept for backwards compatibility.
            
        Returns:
            Dict[str, Any]: Answer content and listing of source references.
        """
        logger.info("Executing Q&A workflow in RAGEngine...")
        
        # 1. Route query
        route = self.determine_route(query)
        
        # 2. Retrieve context data
        context = self.retrieve_context(dataset_id, query, route, db)
        
        # 3. Pull last 3 persistent QA records from SQLite for conversational memory support (last 3 interactions = 6 messages)
        db_history = []
        try:
            db_history = (
                db.query(ChatHistory)
                .filter(ChatHistory.dataset_id == dataset_id)
                .order_by(ChatHistory.timestamp.desc())
                .limit(3)
                .all()
            )
            # Reverse to chronological order (oldest first)
            db_history = db_history[::-1]
        except Exception as msg_err:
            logger.warning(f"Failed to query chat history from SQLite: {msg_err}")
            
        chat_history = []
        for r in db_history:
            chat_history.append(LcHumanMessage(content=r.question))
            chat_history.append(LcAIMessage(content=r.answer))

        # 4. Construct LangChain Prompt Template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a strict, analytical Data Profiling Q&A Assistant.\n"
                "Answer the user's question strictly using the provided dataset context details. "
                "Never make up details, assume, or hallucinate outside the given context information. "
                "If you cannot find the exact answer in the context, state: 'I do not have enough information to answer this.'\n\n"
                "Context details of the dataset:\n"
                "=================================\n"
                "{context}\n"
                "=================================\n\n"
                "You MUST return a JSON object with the following schema:\n"
                "{{\n"
                "  \"answer\": \"Your concise and analytical answer here.\",\n"
                "  \"citation\": \"The specific column names, table summaries, or correlation metrics used to find the answer (e.g. 'columns.Age', 'summary.duplicates', or 'correlations.pearson')\"\n"
                "}}\n"
                "Ensure the output is valid JSON and nothing else."
            )),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        # 5. Build and invoke Chain: prompt template -> ChatGroq LLM client
        chain = prompt_template | self.llm_client.llm
        
        try:
            logger.info("Invoking LangChain RAG pipeline chain...")
            response = chain.invoke({
                "context": context["text"],
                "chat_history": chat_history,
                "question": query
            })
            response_content = response.content.strip()
        except Exception as chain_err:
            logger.error(f"Error invoking LangChain RAG chain: {chain_err}", exc_info=True)
            response_content = json.dumps({
                "answer": f"Inference execution failure: {chain_err}",
                "citation": "system.error"
            })

        # 6. Parse structured response JSON output
        try:
            parsed_res = json.loads(response_content)
            if "answer" not in parsed_res:
                parsed_res["answer"] = response_content
            if "citation" not in parsed_res:
                parsed_res["citation"] = "unknown"
        except Exception as parse_err:
            logger.warning(f"Response from LLM was not valid JSON: {response_content}. Error: {parse_err}")
            parsed_res = {
                "answer": response_content,
                "citation": "unknown"
            }
            
        citation_sources = parsed_res.get("citation", "unknown")
        sources_list = [citation_sources] if isinstance(citation_sources, str) else citation_sources
        
        # Merge semantic sources from retriever as well to provide comprehensive citation list
        if context["sources"]:
            sources_list.extend(context["sources"])
        # Unique list
        sources_list = list(set(sources_list))

        return {
            "answer": parsed_res.get("answer", response_content),
            "sources": sources_list
        }

