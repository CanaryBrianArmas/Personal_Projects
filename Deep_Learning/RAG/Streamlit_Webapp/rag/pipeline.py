"""RAG pipeline orchestration."""
from typing import List, Dict, Any
from rag.embeddings import DocumentEmbedder
from rag.retriever import DocumentRetriever
from rag.generator import TextGenerator
from app.config import config

class RAGPipeline:
    """Orchestrates the complete RAG pipeline."""
    
    def __init__(self):
        """Initialize the RAG pipeline components."""
        self.embedder = DocumentEmbedder()
        self.retriever = DocumentRetriever(embedder=self.embedder)
        self.generator = TextGenerator()
        
        # Try to load existing index
        self.retriever.load_index()
    
    def add_documents(self, documents: List[str], document_ids: List[str] = None) -> None:
        """Add documents to the retrieval system.
        
        Args:
            documents: List of document texts
            document_ids: Optional list of document identifiers
        """
        self.retriever.add_documents(documents, document_ids)
    
    def query(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """Process a query through the RAG pipeline.
        
        Args:
            query: The user's question
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary containing the response and retrieved contexts
        """
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query, top_k)
        
        # Generate answer
        if retrieved_docs:
            response = self.generator.generate(query, retrieved_docs)
        else:
            response = "I don't have enough information to answer that question."
        
        return {
            "query": query,
            "response": response,
            "contexts": retrieved_docs
        }
    
    def has_documents(self) -> bool:
        """Check if the system has indexed documents.
        
        Returns:
            Boolean indicating if documents are available
        """
        return self.retriever.index is not None and self.retriever.index.ntotal > 0