import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from loguru import logger

from src.config import settings
from src.indexing import FAISSIndexManager

class HybridRetrieverManager:
    """
    Manages hybrid search by combining dense FAISS search and sparse BM25 search
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, faiss_manager: FAISSIndexManager):
        self.faiss_manager = faiss_manager
        self.docs_path = settings.FAISS_INDEX_PATH / "documents.pkl"
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.ensemble_retriever: Optional[EnsembleRetriever] = None

    def initialize_retrievers(self, docs: Optional[List[Document]] = None, dense_weight: float = 0.5, sparse_weight: float = 0.5):
        """
        Initializes FAISS and BM25 retrievers, and sets up the EnsembleRetriever.
        If `docs` is not provided, it attempts to load them from local disk.
        """
        vector_store = self.faiss_manager.get_vector_store()
        if vector_store is None:
            raise ValueError("FAISS vector store is not built/loaded. Run ingestion first.")
            
        # Get dense retriever
        dense_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        
        # Load or use provided documents for BM25
        if docs is None:
            if not self.docs_path.exists():
                logger.warning(f"Pickled documents not found at {self.docs_path}. Attempting to reconstruct from FAISS index...")
                # Fallback: Extract from FAISS database directly
                docs = list(vector_store.docstore._dict.values())
            else:
                logger.info(f"Loading pickled documents from {self.docs_path}...")
                with open(self.docs_path, "rb") as f:
                    docs = pickle.load(f)
                    
        logger.info(f"Initializing BM25 retriever with {len(docs)} documents...")
        self.bm25_retriever = BM25Retriever.from_documents(docs)
        self.bm25_retriever.k = 10
        
        # Build ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, self.bm25_retriever],
            weights=[dense_weight, sparse_weight]
        )
        logger.info(f"Hybrid retrieval system initialized (weights: Dense={dense_weight}, Sparse={sparse_weight})")

    def save_documents(self, chunks: List[Dict[str, Any]]):
        """
        Helper method to serialize the list of Document objects.
        """
        documents = []
        for chunk in chunks:
            metadata = {k: v for k, v in chunk["metadata"].items() if v is not None}
            doc = Document(page_content=chunk["text"], metadata=metadata)
            documents.append(doc)
            
        self.docs_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving {len(documents)} documents to {self.docs_path} for BM25...")
        with open(self.docs_path, "wb") as f:
            pickle.dump(documents, f)

    def retrieve(self, query: str, filter_dict: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Retrieves matching documents using hybrid search.
        Filters can be optionally applied post-retrieval (or via the retriever if supported).
        """
        if self.ensemble_retriever is None:
            self.initialize_retrievers()
            
        logger.info(f"Executing hybrid search for query: '{query}'")
        retrieved_docs = self.ensemble_retriever.invoke(query)
        
        # Apply metadata filtering if specified
        if filter_dict:
            filtered_docs = []
            for doc in retrieved_docs:
                matches = True
                for k, v in filter_dict.items():
                    if doc.metadata.get(k) != v:
                        matches = False
                        break
                if matches:
                    filtered_docs.append(doc)
            logger.info(f"Metadata filter applied. Reduced candidate list from {len(retrieved_docs)} to {len(filtered_docs)} docs.")
            return filtered_docs
            
        return retrieved_docs
