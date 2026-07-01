from typing import Iterator, List, Dict
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from loguru import logger

from src.config import settings
from src.generation.prompts import KRISHNA_SYSTEM_PROMPT, USER_REWRITE_PROMPT

class KrishnaGenerator:
    """
    Handles prompt construction, query rewriting, and response generation
    (streaming and non-streaming) using Ollama.
    """

    def __init__(self):
        logger.info(f"Initializing Ollama LLM (model: {settings.OLLAMA_MODEL}, url: {settings.OLLAMA_BASE_URL})...")
        self.llm = Ollama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=0.2  # Keep temperature low for high grounding
        )
        
        # Setup Prompts
        self.system_prompt = PromptTemplate.from_template(KRISHNA_SYSTEM_PROMPT)
        self.rewrite_prompt = PromptTemplate.from_template(USER_REWRITE_PROMPT)

    def rewrite_query(self, query: str, history: List[Dict[str, str]]) -> str:
        """
        Rewrites a conversational user query into a search-optimized keyword query.
        """
        if not history:
            return query
            
        logger.info(f"Rewriting query based on conversation history. Current query: '{query}'")
        
        # Format history for prompt
        history_str = ""
        for turn in history:
            role = "User" if turn["role"] == "user" else "Krishna"
            history_str += f"{role}: {turn['content']}\n"
            
        formatted_prompt = self.rewrite_prompt.format(history=history_str, query=query)
        rewritten = self.llm.invoke(formatted_prompt).strip()
        
        logger.info(f"Rewritten query: '{rewritten}'")
        return rewritten

    def _prepare_context(self, documents: List[Document]) -> str:
        """
        Formats retrieved documents into a clean context string for the prompt.
        """
        context_parts = []
        for idx, doc in enumerate(documents):
            source = doc.metadata.get("source", "Bhagavad Gita")
            page = doc.metadata.get("page", "Unknown")
            chapter = doc.metadata.get("chapter", "Unknown")
            verse = doc.metadata.get("verse", "Unknown")
            
            # Format header
            header = f"[Source: {source} | Page: {page} | Chapter: {chapter} | Verse: {verse}]"
            content = doc.page_content.strip()
            
            context_parts.append(f"{header}\n{content}\n")
            
        return "\n---\n".join(context_parts)

    def generate(self, query: str, retrieved_docs: List[Document]) -> str:
        """
        Generates a non-streaming response as Lord Krishna.
        """
        context_str = self._prepare_context(retrieved_docs)
        formatted_prompt = self.system_prompt.format(context=context_str, query=query)
        
        logger.info("Generating response from LLM...")
        response = self.llm.invoke(formatted_prompt)
        return response

    def generate_stream(self, query: str, retrieved_docs: List[Document]) -> Iterator[str]:
        """
        Generates a streaming response as Lord Krishna, yielding chunks of text.
        """
        context_str = self._prepare_context(retrieved_docs)
        formatted_prompt = self.system_prompt.format(context=context_str, query=query)
        
        logger.info("Starting response stream from LLM...")
        for chunk in self.llm.stream(formatted_prompt):
            yield chunk
