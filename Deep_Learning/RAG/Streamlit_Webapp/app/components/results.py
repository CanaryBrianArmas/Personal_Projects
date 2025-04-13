"""Results display component for the Streamlit RAG application."""
import streamlit as st

def render_results(results):
    """Render the query results in the Streamlit interface.
    
    Args:
        results: Dictionary containing the query, response, and contexts
    """
    # Display response
    st.subheader("Answer")
    st.markdown(f"**{results['response']}**")
    
    # Display source contexts
    st.subheader("Sources")
    
    for i, context in enumerate(results["contexts"]):
        with st.expander(f"Source {i+1}: {context['metadata']['doc_id']}", expanded=i==0):
            st.markdown(context["text"])
            st.caption(f"Relevance score: {context['score']:.4f}")
            
            # Display document metadata
            st.info(f"""
                **Document**: {context['metadata']['doc_id']}  
                **Chunk**: {context['metadata']['chunk_id']}
            """)
    
    # Feedback buttons (not functional, just for UI purposes)
    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        st.button("👍 Helpful")
    with col2:
        st.button("👎 Not helpful")