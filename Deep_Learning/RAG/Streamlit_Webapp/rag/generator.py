"""Text generation functionality for the RAG system."""
from typing import List, Dict, Any
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from app.config import config

class TextGenerator:
    """Handles text generation using HuggingFace models."""
    
    def __init__(self, model_name: str = None):
        """Initialize the generator with a specified model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name or config.llm_model
        self._load_model()
        
    def _load_model(self):
        """Load the text generation model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            print(f"Successfully loaded model: {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load text generation model: {e}")
    
    def _prepare_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved documents.
        
        Args:
            context_docs: List of retrieved document chunks
            
        Returns:
            Formatted context string
        """
        context_text = ""
        
        for i, doc in enumerate(context_docs):
            context_text += f"Document {i+1}:\n{doc['text']}\n\n"
            
        return context_text.strip()
    
    def generate(self, query: str, context_docs: List[Dict[str, Any]], 
                 max_length: int = 512) -> str:
        """Generate text based on query and context.
        
        Args:
            query: The user's question
            context_docs: List of retrieved document chunks
            max_length: Maximum length of generated text
            
        Returns:
            Generated text response
        """
        context = self._prepare_context(context_docs)
        
        prompt = f"""
Based on the following information, please answer this question:

Question: {query}

Information:
{context}

Answer:
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response