"""Unit tests for the DocumentEmbedder class."""
import unittest
import torch
import os
from unittest.mock import patch, MagicMock
from rag.embeddings import DocumentEmbedder

class TestDocumentEmbedder(unittest.TestCase):
    """Test cases for DocumentEmbedder class."""
    
    @patch('rag.embeddings.SentenceTransformer')
    def setUp(self, mock_sentence_transformer):
        """Set up test fixtures."""
        # Configure the mock SentenceTransformer
        self.mock_model = MagicMock()
        mock_sentence_transformer.return_value = self.mock_model
        
        # Configure the mock encoding behavior
        def mock_encode(texts, convert_to_tensor=False):
            # Create mock embeddings with dimension 384 (common for sentence-transformers)
            if isinstance(texts, list):
                # For document encoding, return a tensor with shape [n_docs, embedding_dim]
                mock_embeddings = torch.rand(len(texts), 384)
            else:
                # For single query encoding, return a tensor with shape [1, embedding_dim]
                mock_embeddings = torch.rand(1, 384)
                
            return mock_embeddings
            
        self.mock_model.encode.side_effect = mock_encode
        
        # Initialize the embedder with a test model name
        self.embedder = DocumentEmbedder(model_name="test-model")
    
    def test_initialization(self):
        """Test proper initialization of DocumentEmbedder."""
        self.assertEqual(self.embedder.model_name, "test-model")
    
    def test_embed_documents(self):
        """Test document embedding functionality."""
        documents = ["This is a test document.", "Another test document."]
        embeddings = self.embedder.embed_documents(documents)
        
        # Check that the mock encode was called with the documents
        self.mock_model.encode.assert_called_with(documents, convert_to_tensor=True)
        
        # Check that embeddings have the correct shape
        self.assertIsInstance(embeddings, torch.Tensor)
        self.assertEqual(embeddings.shape[0], len(documents))
    
    def test_embed_query(self):
        """Test query embedding functionality."""
        query = "Test query"
        embedding = self.embedder.embed_query(query)
        
        # Check that the mock encode was called with the query
        self.mock_model.encode.assert_called_with(query, convert_to_tensor=True)
        
        # Check that embedding is a tensor
        self.assertIsInstance(embedding, torch.Tensor)
    
    @patch('rag.embeddings.SentenceTransformer')
    def test_model_load_error(self, mock_sentence_transformer):
        """Test handling of model loading errors."""
        # Configure the mock to raise an exception
        mock_sentence_transformer.side_effect = Exception("Test exception")
        
        # Check that the model loading error is handled correctly
        with self.assertRaises(RuntimeError):
            DocumentEmbedder(model_name="error-model")
    
    def test_environment_variable_setting(self):
        """Test that HuggingFace token is set correctly."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('rag.embeddings.config') as mock_config:
                # Set the mock config to have a token
                mock_config.huggingface_token = "test-token"
                
                # Create a new embedder which should set the environment variable
                with patch('rag.embeddings.SentenceTransformer'):
                    DocumentEmbedder()
                    
                    # Check if the environment variable was set
                    self.assertEqual(os.environ.get("HUGGINGFACE_TOKEN"), "test-token")

if __name__ == '__main__':
    unittest.main()