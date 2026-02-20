"""Document retrieval system for the RAG pipeline."""
import os
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.embeddings import DocumentEmbedder
from app.config import config

class DocumentRetriever:
    """Handles document retrieval using FAISS vector store."""
    
    def __init__(self, embedder: DocumentEmbedder = None):
        """Initialize the retriever.
        
        Args:
            embedder: DocumentEmbedder instance for embedding documents
        """
        self.embedder = embedder or DocumentEmbedder()
        self.documents = []
        self.index = None
        self.doc_chunks = []
        
        # Create vector DB directory if it doesn't exist
        os.makedirs(config.vector_db_path, exist_ok=True)
        self.index_path = os.path.join(config.vector_db_path, "faiss_index")
        self.docs_path = os.path.join(config.vector_db_path, "documents.pkl")
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
    
    def add_documents(self, documents: List[str], document_ids: List[str] = None) -> None:
        """Process and add documents to the retrieval system.
        
        Args:
            documents: List of document texts
            document_ids: Optional list of document identifiers
        """
        # Split documents into chunks
        doc_chunks = []
        chunk_metadata = []
        
        for i, doc in enumerate(documents):
            doc_id = document_ids[i] if document_ids and i < len(document_ids) else f"doc_{i}"
            chunks = self.text_splitter.split_text(doc)
            
            for j, chunk in enumerate(chunks):
                doc_chunks.append(chunk)
                chunk_metadata.append({
                    "doc_id": doc_id,
                    "chunk_id": j,
                    "doc_index": i
                })
        
        # Generate embeddings
        embeddings = self.embedder.embed_documents(doc_chunks)
        
        # Convert to numpy array for FAISS
        embeddings_np = embeddings.cpu().numpy()
        
        # Create or update FAISS index
        vector_dimension = embeddings_np.shape[1]
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(vector_dimension)
        
        # Add vectors to index
        self.index.add(embeddings_np.astype(np.float32))
        
        # Store document chunks and metadata
        self.doc_chunks.extend([(chunk, meta) for chunk, meta in zip(doc_chunks, chunk_metadata)])
        self.documents.extend(documents)
        
        # Save index and documents
        self._save_index()
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for a query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of relevant document chunks with metadata
        """
        k = top_k or config.top_k
        
        if not self.index or self.index.ntotal == 0:
            raise ValueError("No documents have been indexed yet")
        
        # Embed query
        query_embedding = self.embedder.embed_query(query)
        query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1).astype(np.float32)
        
        # Search index
        distances, indices = self.index.search(query_embedding_np, k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.doc_chunks) and idx >= 0:
                chunk_text, chunk_meta = self.doc_chunks[idx]
                results.append({
                    "text": chunk_text,
                    "metadata": chunk_meta,
                    "score": float(distances[0][i])
                })
        
        return results
    
    def _save_index(self) -> None:
        """Save the FAISS index and documents to disk."""
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            
        with open(self.docs_path, 'wb') as f:
            pickle.dump((self.documents, self.doc_chunks), f)
    
    def load_index(self) -> bool:
        """Load the FAISS index and documents from disk.
        
        Returns:
            Boolean indicating success
        """
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.docs_path):
                self.index = faiss.read_index(self.index_path)
                
                with open(self.docs_path, 'rb') as f:
                    self.documents, self.doc_chunks = pickle.load(f)
                
                return True
            return False
        except Exception as e:
            print(f"Error loading index: {e}")
            return False