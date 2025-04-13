"""Document embedding functionality for the RAG system."""
import os
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from app.config import config

class DocumentEmbedder:
    """Handles document embedding using HuggingFace models."""
    
    def __init__(self, model_name: str = None):
        """Initialize the embedder with a specified model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name or config.embedding_model
        self._load_model()
        
    def _load_model(self):
        """Load the embedding model."""
        # Set HuggingFace token if available
        if config.huggingface_token:
            os.environ["HUGGINGFACE_TOKEN"] = config.huggingface_token
            
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"Successfully loaded model: {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")
    
    def embed_documents(self, documents: List[str]) -> torch.Tensor:
        """Embed a list of documents.
        
        Args:
            documents: List of text documents to embed
            
        Returns:
            Tensor of document embeddings
        """
        return self.model.encode(documents, convert_to_tensor=True)
    
    def embed_query(self, query: str) -> torch.Tensor:
        """Embed a single query string.
        
        Args:
            query: Query text to embed
            
        Returns:
            Tensor of query embedding
        """
        return self.model.encode(query, convert_to_tensor=True)