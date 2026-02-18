"""Configuration management for the RAG system."""
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

class RAGConfig(BaseModel):
    """Configuration settings for the RAG system."""
    # HuggingFace settings
    huggingface_token: str = Field(default=os.getenv("HUGGINGFACE_API_TOKEN", ""))
    
    # Model settings
    embedding_model: str = Field(default=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    llm_model: str = Field(default=os.getenv("LLM_MODEL", "google/flan-t5-base"))
    
    # Vector database settings
    vector_db_path: str = Field(default=os.getenv("VECTOR_DB_PATH", "./vector_db"))
    
    # Application settings
    log_level: str = Field(default=os.getenv("LOG_LEVEL", "INFO"))
    
    # RAG pipeline settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    
    def validate_config(self) -> bool:
        """Validate that all required settings are present."""
        if not self.huggingface_token:
            raise ValueError("HuggingFace API token is required")
        return True

# Create and validate config
config = RAGConfig()