"""Sidebar component for the Streamlit RAG application."""
import os
import streamlit as st
import tempfile

def render_sidebar():
    """Render the sidebar with document upload functionality."""
    st.sidebar.header("Document Management")
    
    # File uploader
    uploaded_files = st.sidebar.file_uploader(
        "Upload documents",
        accept_multiple_files=True,
        type=["txt", "pdf", "docx"]
    )
    
    if uploaded_files:
        if st.sidebar.button("Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                process_documents(uploaded_files)
    
    # Document status
    st.sidebar.subheader("System Status")
    
    if hasattr(st.session_state.rag_pipeline.retriever, 'index') and \
       st.session_state.rag_pipeline.retriever.index is not None:
        doc_count = len(st.session_state.rag_pipeline.retriever.documents)
        chunk_count = len(st.session_state.rag_pipeline.retriever.doc_chunks)
        st.sidebar.success(f"✅ {doc_count} documents processed ({chunk_count} chunks)")
    else:
        st.sidebar.warning("No documents loaded")
    
    # Settings
    st.sidebar.subheader("Settings")
    top_k = st.sidebar.slider(
        "Number of contexts to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="The number of document chunks to retrieve for each query"
    )
    
    # Reset system
    if st.sidebar.button("Reset System", type="secondary"):
        if st.sidebar.checkbox("Confirm reset"):
            reset_system()
            st.sidebar.success("System reset successfully!")

def process_documents(uploaded_files):
    """Process uploaded documents and add them to the RAG pipeline.
    
    Args:
        uploaded_files: List of uploaded file objects
    """
    documents = []
    document_ids = []
    
    for file in uploaded_files:
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.name)
        
        # Save the file temporarily
        with open(temp_path, "wb") as f:
            f.write(file.getvalue())
        
        # Extract text based on file type
        if file.name.endswith('.txt'):
            with open(temp_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif file.name.endswith('.pdf'):
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(temp_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            except ImportError:
                st.error("Please install PyPDF2 to process PDF files")
                continue
        elif file.name.endswith('.docx'):
            try:
                import docx
                doc = docx.Document(temp_path)
                text = "\n".join([para.text for para in doc.paragraphs])
            except ImportError:
                st.error("Please install python-docx to process DOCX files")
                continue
        else:
            st.error(f"Unsupported file format: {file.name}")
            continue
        
        documents.append(text)
        document_ids.append(file.name)
    
    # Add documents to the RAG pipeline
    if documents:
        st.session_state.rag_pipeline.add_documents(documents, document_ids)
        st.success(f"Successfully processed {len(documents)} documents")

def reset_system():
    """Reset the RAG system and clear session state."""
    # Create a new RAG pipeline
    st.session_state.rag_pipeline = None
    st.session_state.query_history = []
    st.session_state.current_results = None
    
    # Reinitialize the pipeline
    from rag.pipeline import RAGPipeline
    st.session_state.rag_pipeline = RAGPipeline()