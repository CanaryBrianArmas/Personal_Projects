"""Main Streamlit application for the RAG system."""
import os
import streamlit as st
from rag.pipeline import RAGPipeline
from app.components.sidebar import render_sidebar
from app.components.results import render_results
from app.config import config
from app.utils.helpers import load_css, apply_custom_theme

# Page configuration
st.set_page_config(
    page_title="RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS and theme
load_css()
apply_custom_theme()


# Initialize session state
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline()

if "query_history" not in st.session_state:
    st.session_state.query_history = []

if "current_results" not in st.session_state:
    st.session_state.current_results = None


# Display app logo and title
logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.svg")
if os.path.exists(logo_path):
    with open(logo_path, "r") as f:
        logo_svg = f.read()
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown(f"""
            <div style="width: 100px; height: 100px;">
                {logo_svg}
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown('<h1 class="main-header">Retrieval-Augmented Generation System</h1>', unsafe_allow_html=True)
else:
    st.title("🔍 Retrieval-Augmented Generation System")

st.markdown("Ask questions based on your documents using advanced AI.")


# Render sidebar with file upload functionality
render_sidebar()


# Main query area
st.subheader("Ask a Question")
query = st.text_input("Enter your question:", key="query_input")
col1, col2 = st.columns([1, 9])
with col1:
    submit_button = st.button("Submit", type="primary", key="submit_button")


# Process query when submitted
if submit_button and query:
    with st.spinner("Processing your question..."):
        # Check if documents are available
        if not st.session_state.rag_pipeline.has_documents():
            st.error("Please upload documents first!")
        else:
            # Process query through RAG pipeline
            results = st.session_state.rag_pipeline.query(query)
            
            # Store results in session state
            st.session_state.current_results = results
            
            # Add to query history
            st.session_state.query_history.append({
                "query": query,
                "response": results["response"],
                "timestamp": st.session_state.get("timestamp", None)
            })
            
            # Clear query input
            st.session_state.query_input = ""


# Display results
if st.session_state.current_results:
    render_results(st.session_state.current_results)


# Display query history
if st.session_state.query_history:
    st.subheader("Query History")
    for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
        with st.expander(f"Q: {item['query']}", expanded=i==0):
            st.markdown('<div class="answer-container">', unsafe_allow_html=True)
            st.markdown(f"**Response:** {item['response']}")
            st.markdown('</div>', unsafe_allow_html=True)
            if item.get("timestamp"):
                st.caption(f"Asked at: {item['timestamp']}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and HuggingFace models")