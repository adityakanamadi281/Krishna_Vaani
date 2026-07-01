import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Project Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    PDF_PATH: Path = DATA_DIR / "The Bhagavad Gita.pdf"
    FAISS_INDEX_PATH: Path = DATA_DIR / "faiss_index"

    # Embeddings & Retrieval
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "ms-marco-MiniLM-L-12-v2"
    
    # LLM Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3.5:0.8b"

    # API Settings
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000

    # Streamlit Settings
    STREAMLIT_PORT: int = 8501

settings = Settings()
