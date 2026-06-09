"""
Ollama LLM Service Client.
Integrates with local Ollama deployment using LangChain interfaces to execute inference queries
over local instruction models (qwen2.5:7b-instruct-q4_K_M).
Configures structured JSON schema formats directly.
"""

import logging
from typing import Optional, List, Union
from backend.config import settings

# Import ChatOllama from langchain-ollama
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

logger = logging.getLogger("backend_llm_client")

class LLMClient:
    _llm = None

    def __init__(self):
        """Initializes LangChain ChatOllama module bindings in structured JSON mode."""
        if LLMClient._llm is None:
            logger.info(f"Connecting to ChatOllama at {settings.OLLAMA_BASE_URL} for model: {settings.OLLAMA_MODEL}")
            try:
                # Enforce JSON mode output native to Ollama API via langchain-ollama
                LLMClient._llm = ChatOllama(
                    base_url=settings.OLLAMA_BASE_URL,
                    model=settings.OLLAMA_MODEL,
                    temperature=0.0,
                    format="json"
                )
            except Exception as e:
                logger.error(f"Failed to initialize ChatOllama client: {e}", exc_info=True)
                raise RuntimeError(f"ChatOllama client initialization failure: {e}")
        self.llm = LLMClient._llm

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Sends configured system instructions and contextual user queries to Ollama.
        Forces structured JSON output.
        
        Args:
            prompt (str): Prepared user query merged with context facts.
            system_prompt (Optional[str]): Operational system constraints.
            
        Returns:
            str: Generated JSON text answer.
        """
        logger.info("Sending prompt payload to local ChatOllama in JSON format mode...")
        
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Failed to generate response from ChatOllama: {e}", exc_info=True)
            # If Ollama connection fails, return a safe fallback JSON string
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

