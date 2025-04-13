"""Helper utilities for the RAG application."""
import os
import re
import logging
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import streamlit as st
from app.config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_css():
    """Load custom CSS for the Streamlit app."""
    css_file = os.path.join(os.path.dirname(__file__), "..", "assets", "style.css")
    
    if os.path.exists(css_file):
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        logger.warning(f"CSS file not found at {css_file}")

def extract_text_from_file(file, file_path: str) -> Tuple[str, Optional[str]]:
    """Extract text content from various file formats.
    
    Args:
        file: The uploaded file object
        file_path: Path to the temporarily saved file
        
    Returns:
        Tuple of (extracted text, error message if any)
    """
    try:
        if file.name.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read(), None
                
        elif file.name.endswith('.pdf'):
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text, None
            except ImportError:
                return "", "PyPDF2 library not installed. Install with: pip install PyPDF2"
                
        elif file.name.endswith('.docx'):
            try:
                import docx
                doc = docx.Document(file_path)
                text = "\n".join([para.text for para in doc.paragraphs])
                return text, None
            except ImportError:
                return "", "python-docx library not installed. Install with: pip install python-docx"
                
        else:
            return "", f"Unsupported file format: {file.name}"
    
    except Exception as e:
        logger.error(f"Error extracting text from {file.name}: {str(e)}")
        return "", f"Error processing file: {str(e)}"

def clean_text(text: str) -> str:
    """Clean and normalize text.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s.,;:!?\'\"()-]', '', text)
    
    return text.strip()

def format_timestamp(timestamp=None):
    """Format timestamp for display.
    
    Args:
        timestamp: Optional timestamp to format (uses current time if None)
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length-3] + "..."

def apply_custom_theme():
    """Apply custom theme to Streamlit app."""
    # Define custom theme
    st.markdown("""
        <style>
        :root {
            --primary-color: #1E88E5;
            --background-color: #F8F9FA;
            --secondary-background-color: #FFFFFF;
            --text-color: #424242;
            --font: 'Source Sans Pro', sans-serif;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Load Google font
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap');
        html, body, [class*="st-"] {
            font-family: 'Source Sans Pro', sans-serif;
        }
        </style>
    """, unsafe_allow_html=True)

def get_file_icon(file_extension: str) -> str:
    """Get an icon for a file based on its extension.
    
    Args:
        file_extension: File extension without the dot
        
    Returns:
        Emoji icon for the file type
    """
    icon_map = {
        'pdf': '📄',
        'doc': '📝',
        'docx': '📝',
        'txt': '📄',
        'csv': '📊',
        'xls': '📊',
        'xlsx': '📊',
        'ppt': '📊',
        'pptx': '📊',
        'jpg': '🖼️',
        'jpeg': '🖼️',
        'png': '🖼️',
    }
    
    return icon_map.get(file_extension.lower(), '📁')