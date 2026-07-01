import os
from typing import List
from langchain_core.documents import Document
from flashrank import RerankRequest, Ranker
from loguru import logger

from src.config import settings

class LocalReranker:
    """
    Reranks retrieved documents using FlashRank (a fast, CPU-friendly Cross-Encoder model).
    """

    def __init__(self, model_name: str = settings.RERANKER_MODEL):
        self.model_name = model_name
        self.cache_dir = str(settings.DATA_DIR / "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Initializing FlashRank Reranker (model: {self.model_name}, cache: {self.cache_dir})...")
        # Load the reranker model (cache_dir must be passed to control where the model is downloaded)
        self.reranker = Ranker(model_name=self.model_name, cache_dir=self.cache_dir)
        logger.info("FlashRank Reranker initialized successfully")

    def rerank(self, query: str, documents: List[Document], top_n: int = 4) -> List[Document]:
        """
        Reranks a list of LangChain Document objects and returns the top_n items.
        """
        if not documents:
            return []
            
        logger.info(f"Reranking {len(documents)} documents for query: '{query}'")
        
        # Prepare data structure for FlashRank
        passages = []
        for idx, doc in enumerate(documents):
            passages.append({
                "id": idx,
                "text": doc.page_content,
                "meta": doc.metadata
            })
            
        rank_request = RerankRequest(query=query, passages=passages)
        reranked_results = self.reranker.rerank(rank_request)
        
        # Reconstruct LangChain Document objects from sorted results
        reranked_docs = []
        for res in reranked_results[:top_n]:
            idx = res["id"]
            score = res.get("score", 0.0)
            
            # Retrieve original document and update metadata with rerank score
            orig_doc = documents[idx]
            metadata = orig_doc.metadata.copy()
            metadata["rerank_score"] = float(score)
            
            reranked_docs.append(
                Document(page_content=orig_doc.page_content, metadata=metadata)
            )
            
        logger.info(f"Reranking complete. Selected top {len(reranked_docs)} documents.")
        return reranked_docs
