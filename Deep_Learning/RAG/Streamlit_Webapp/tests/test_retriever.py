"""Unit tests for the DocumentRetriever class."""
import unittest
import os
import numpy as np
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import torch
import faiss
from rag.retriever import DocumentRetriever
from rag.embeddings import DocumentEmbedder

class TestDocumentRetriever(unittest.TestCase):
    """Test cases for DocumentRetriever class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for the vector DB
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock the config
        self.config_patcher = patch('rag.retriever.config')
        self.mock_config = self.config_patcher.start()
        self.mock_config.vector_db_path = self.temp_dir
        self.mock_config.chunk_size = 100
        self.mock_config.chunk_overlap = 20
        self.mock_config.top_k = 3
        
        # Mock the embedder
        self.embedder_patcher = patch('rag.retriever.DocumentEmbedder')
        self.mock_embedder_class = self.embedder_patcher.start()
        self.mock_embedder = MagicMock()
        self.mock_embedder_class.return_value = self.mock_embedder
        
        # Configure mock embedding behavior
        def mock_embed_documents(texts):
            # Create mock embeddings with dimension 384
            mock_embeddings = torch.rand(len(texts), 384)
            return mock_embeddings
            
        def mock_embed_query(query):
            # Create a single mock embedding
            return torch.rand(1, 384)
            
        self.mock_embedder.embed_documents.side_effect = mock_embed_documents
        self.mock_embedder.embed_query.side_effect = mock_embed_query
        
        # Create the retriever
        self.retriever = DocumentRetriever()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop the patchers
        self.config_patcher.stop()
        self.embedder_patcher.stop()
        
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test proper initialization of DocumentRetriever."""
        self.assertIsNone(self.retriever.index)
        self.assertEqual(len(self.retriever.documents), 0)
        self.assertEqual(len(self.retriever.doc_chunks), 0)
    
    def test_add_documents(self):
        """Test adding documents to the retriever."""
        # Test documents
        documents = [
            "This is the first test document.",
            "This is the second test document with more content."
        ]
        document_ids = ["doc1", "doc2"]
        
        # Add documents
        self.retriever.add_documents(documents, document_ids)
        
        # Check that documents were stored
        self.assertEqual(len(self.retriever.documents), 2)
        
        # Check that an index was created
        self.assertIsNotNone(self.retriever.index)
        self.assertIsInstance(self.retriever.index, faiss.IndexFlatL2)
        
        # Check that the embedder was called
        self.mock_embedder.embed_documents.assert_called()
    
    def test_retrieve(self):
        """Test document retrieval."""
        # First add some documents
        documents = [
            "This is the first test document.",
            "This is the second test document with more content."
        ]
        document_ids = ["doc1", "doc2"]
        self.retriever.add_documents(documents, document_ids)
        
        # Mock the FAISS search method
        original_search = self.retriever.index.search
        def mock_search(x, k):
            # Return mock distances and indices
            distances = np.array([[0.1, 0.2, 0.3]])
            indices = np.array([[0, 1, -1]])  # -1 is an invalid index
            return distances, indices
            
        self.retriever.index.search = mock_search
        
        # Test retrieval
        results = self.retriever.retrieve("Test query")
        
        # Restore original search method
        self.retriever.index.search = original_search
        
        # Check results
        self.assertEqual(len(results), 2)  # Only 2 valid indices
        self.assertEqual(results[0]["metadata"]["doc_id"], "doc1")
        self.assertEqual(results[1]["metadata"]["doc_id"], "doc2")
        
        # Check scores
        self.assertEqual(results[0]["score"], 0.1)
        self.assertEqual(results[1]["score"], 0.2)
    
    def test_save_and_load_index(self):
        """Test saving and loading the index."""
        # Add documents
        documents = ["This is a test document."]
        self.retriever.add_documents(documents)
        
        # Create a new retriever
        new_retriever = DocumentRetriever()
        
        # Load the index
        success = new_retriever.load_index()
        
        # Check that the index was loaded
        self.assertTrue(success)
        self.assertIsNotNone(new_retriever.index)
        self.assertEqual(len(new_retriever.documents), 1)
    
    def test_retrieve_no_index(self):
        """Test retrieval with no index."""
        # Should raise a ValueError
        with self.assertRaises(ValueError):
            self.retriever.retrieve("Test query")

if __name__ == '__main__':
    unittest.main()