"""
Groq LLM Service Client.
Integrates with Groq API using LangChain interfaces to execute inference queries
over Llama-3.3-70b-versatile cloud model.
Configures structured JSON schema formats directly.
"""

import logging
from typing import Optional, List, Union
from backend.config import settings

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

logger = logging.getLogger("backend_llm_client")

class LLMClient:
    _llm = None

    def __init__(self):
        """Initializes LangChain ChatGroq module bindings in structured JSON mode."""
        if LLMClient._llm is None:
            logger.info(f"Connecting to Groq API for model: {settings.GROQ_MODEL}")
            try:
                # Enforce JSON mode output native to Groq API via langchain-groq
                LLMClient._llm = ChatGroq(
                    model=settings.GROQ_MODEL,
                    temperature=0.0,
                    groq_api_key=settings.GROQ_API_KEY,
                    model_kwargs={"response_format": {"type": "json_object"}}
                )
            except Exception as e:
                logger.error(f"Failed to initialize ChatGroq client: {e}", exc_info=True)
                raise RuntimeError(f"ChatGroq client initialization failure: {e}")
        self.llm = LLMClient._llm

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Sends configured system instructions and contextual user queries to Groq.
        Forces structured JSON output.
        
        Args:
            prompt (str): Prepared user query merged with context facts.
            system_prompt (Optional[str]): Operational system constraints.
            
        Returns:
            str: Generated JSON text answer.
        """
        logger.info("Sending prompt payload to ChatGroq in JSON format mode...")
        
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Failed to generate response from ChatGroq: {e}", exc_info=True)
            # If Groq connection fails, return a safe fallback JSON string
            import json
            fallback = {
                "answer": f"Error: LLM model inference failed due to connection failure. Details: {e}",
                "citation": "system.error"
            }
            return json.dumps(fallback)
            
    def invoke_messages(self, messages: List[BaseMessage]) -> str:
        """
        Directly executes list of LangChain messages on the LLM client.
        
        Args:
            messages (List[BaseMessage]): Chronological messages context.
            
        Returns:
            str: JSON text output.
        """
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Failed to execute messages list: {e}", exc_info=True)
            import json
            fallback = {
                "answer": f"Error: LLM execution failed. Details: {e}",
                "citation": "system.error"
            }
            return json.dumps(fallback)

