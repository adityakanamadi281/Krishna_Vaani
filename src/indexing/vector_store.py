import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from loguru import logger

from src.config import settings

class FAISSIndexManager:
    """
    Manages building, saving, loading, and querying the FAISS vector store.
    """
    
    def __init__(self):
        self.embeddings = FastEmbedEmbeddings(model_name=settings.EMBEDDING_MODEL)
        self.index_path = settings.FAISS_INDEX_PATH
        self.vector_store: Optional[FAISS] = None

    def build_index(self, chunks: List[Dict[str, Any]]) -> FAISS:
        """
        Builds a new FAISS index from text chunks and saves it locally.
        """
        logger.info(f"Building FAISS index with {len(chunks)} chunks...")
        
        # Convert dictionary chunks to LangChain Document objects
        documents = []
        for chunk in chunks:
            # Metadata keys must not contain None or dict/list values for some vectorstores,
            # but FAISS/LangChain supports standard flat types (strings, ints, floats).
            metadata = {k: v for k, v in chunk["metadata"].items() if v is not None}
            doc = Document(page_content=chunk["text"], metadata=metadata)
            documents.append(doc)
            
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Save index locally
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(self.index_path))
        logger.info(f"Successfully built and saved FAISS index to {self.index_path}")
        return self.vector_store

    def load_index(self) -> FAISS:
        """
        Loads the FAISS index from the local filesystem.
        """
        if not self.index_path.exists() or not (self.index_path / "index.faiss").exists():
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}. Please run ingestion first.")
            
        logger.info(f"Loading FAISS index from {self.index_path}...")
        self.vector_store = FAISS.load_local(
            folder_path=str(self.index_path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True  # Required for loading pickle-based local FAISS indexes
        )
        logger.info("FAISS index loaded successfully")
        return self.vector_store

    def get_vector_store(self) -> FAISS:
        """
        Returns the vector store, loading it if not already loaded.
        """
        if self.vector_store is None:
            try:
                self.load_index()
            except FileNotFoundError:
                logger.warning("FAISS index does not exist yet. It needs to be built.")
        return self.vector_store
        
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Returns a retriever object for similarity searching.
        """
        store = self.get_vector_store()
        if store is None:
            raise ValueError("Vector store is not initialized or built.")
        
        kwargs = search_kwargs or {"k": 10}
        return store.as_retriever(search_kwargs=kwargs)
