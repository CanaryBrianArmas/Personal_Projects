"""Unit tests for the TextGenerator class."""
import unittest
from unittest.mock import patch, MagicMock
import torch
from rag.generator import TextGenerator

class TestTextGenerator(unittest.TestCase):
    """Test cases for TextGenerator class."""
    
    @patch('rag.generator.AutoTokenizer')
    @patch('rag.generator.AutoModelForSeq2SeqLM')
    def setUp(self, mock_model_class, mock_tokenizer_class):
        """Set up test fixtures."""
        # Configure the mocks
        self.mock_tokenizer = MagicMock()
        self.mock_model = MagicMock()
        
        mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer
        mock_model_class.from_pretrained.return_value = self.mock_model
        
        # Configure tokenizer behavior
        self.mock_tokenizer.return_value = {"input_ids": torch.tensor([[0, 1, 2, 3]])}
        self.mock_tokenizer.decode.return_value = "This is a generated response."
        
        # Configure model behavior
        self.mock_model.generate.return_value = torch.tensor([[10, 11, 12, 13]])
        
        # Initialize the generator
        self.generator = TextGenerator(model_name="test-model")
    
    def test_initialization(self):
        """Test proper initialization of TextGenerator."""
        self.assertEqual(self.generator.model_name, "test-model")
        self.assertIsNotNone(self.generator.tokenizer)
        self.assertIsNotNone(self.generator.model)
    
    def test_prepare_context(self):
        """Test context preparation from retrieved documents."""
        context_docs = [
            {"text": "This is document 1.", "metadata": {"doc_id": "doc1"}},
            {"text": "This is document 2.", "metadata": {"doc_id": "doc2"}}
        ]
        
        context = self.generator._prepare_context(context_docs)
        
        # Check that context contains both documents
        self.assertIn("Document 1:", context)
        self.assertIn("This is document 1.", context)
        self.assertIn("Document 2:", context)
        self.assertIn("This is document 2.", context)
    
    def test_generate(self):
        """Test text generation functionality."""
        # Test query and context
        query = "What is the capital of France?"
        context_docs = [
            {"text": "Paris is the capital of France.", "metadata": {"doc_id": "doc1"}},
            {"text": "France is in Europe.", "metadata": {"doc_id": "doc2"}}
        ]
        
        # Generate response
        response = self.generator.generate(query, context_docs)
        
        # Check that tokenizer and model were called
        self.mock_tokenizer.assert_called()
        self.mock_model.generate.assert_called()
        
        # Check that the response is correct
        self.assertEqual(response, "This is a generated response.")
    
    @patch('rag.generator.AutoTokenizer')
    @patch('rag.generator.AutoModelForSeq2SeqLM')
    def test_model_load_error(self, mock_model_class, mock_tokenizer_class):
        """Test handling of model loading errors."""
        # Configure the mock to raise an exception
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Test exception")
        
        # Check that the model loading error is handled correctly
        with self.assertRaises(RuntimeError):
            TextGenerator(model_name="error-model")

if __name__ == '__main__':
    unittest.main()