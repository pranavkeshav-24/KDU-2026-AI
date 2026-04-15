"""
Configuration module using pydantic-settings for environment variable management.
"""
import os

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load .env file from the same directory as this module
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    OPENROUTER_API_KEY: str = Field(default='', description='OpenRouter API key')

    # Paths
    CHROMA_PERSIST_DIR: str = Field(default='./chroma_db', description='ChromaDB persist directory')
    UPLOAD_DIR: str = Field(default='./uploads', description='PDF upload directory')

    # Model Configuration
    EMBEDDING_MODEL: str = Field(default='BAAI/bge-small-en-v1.5', description='Embedding model name')
    LLM_MODEL: str = Field(default='google/gemma-3-27b-it', description='Primary LLM model')
    LLM_FALLBACK_MODEL: str = Field(default='meta-llama/llama-4-scout', description='Fallback LLM model')
    RERANKER_MODEL: str = Field(default='cross-encoder/ms-marco-MiniLM-L-6-v2', description='Cross-encoder reranker model')

    # Retrieval Parameters
    TOP_K_RETRIEVE: int = Field(default=20, description='Number of candidates per retrieval method')
    TOP_K_RERANK: int = Field(default=5, description='Final chunks passed to LLM')
    RRF_K: int = Field(default=60, description='RRF smoothing constant')

    # LLM Parameters
    LLM_TEMPERATURE: float = Field(default=0.1, description='LLM sampling temperature')
    LLM_MAX_TOKENS: int = Field(default=1024, description='LLM max output tokens')
    LLM_MAX_RETRIES: int = Field(default=3, description='Retry count for rate-limited LLM calls')
    LLM_RETRY_BASE_SECONDS: float = Field(default=1.5, description='Base delay used for exponential backoff')

    # Semantic Chunker
    BREAKPOINT_THRESHOLD_TYPE: str = Field(default='percentile', description='SemanticChunker breakpoint type')
    BREAKPOINT_THRESHOLD_AMOUNT: float = Field(default=95, description='SemanticChunker breakpoint threshold')

    model_config = {
        'env_file': '.env',
        'env_file_encoding': 'utf-8',
        'extra': 'ignore',
    }


# Singleton settings instance
settings = Settings()
