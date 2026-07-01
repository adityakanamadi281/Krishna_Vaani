from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

class SemanticChunker:
    """
    Groups and splits document chunks intelligently.
    Verse chunks are kept whole if they are within size limits to preserve commentary context.
    Intro and transition pages are split using a sliding window.
    """
    
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_documents(self, parsed_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"Chunking {len(parsed_docs)} documents...")
        final_chunks = []
        
        for doc in parsed_docs:
            text = doc["text"]
            metadata = doc["metadata"]
            
            # If it's a verse and is not excessively long, keep it intact
            if metadata.get("type") == "verse" and len(text) <= self.chunk_size * 1.5:
                final_chunks.append({
                    "text": text,
                    "metadata": metadata
                })
            else:
                # Split large chunks (intro, chapter_intro, or oversized verse chunks)
                split_texts = self.splitter.split_text(text)
                for idx, split_text in enumerate(split_texts):
                    # Copy metadata and add part info
                    meta = metadata.copy()
                    meta["chunk_id"] = idx
                    final_chunks.append({
                        "text": split_text,
                        "metadata": meta
                    })
                    
        logger.info(f"Generated {len(final_chunks)} final chunks for indexing")
        return final_chunks
