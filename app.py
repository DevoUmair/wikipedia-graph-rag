# streamlit_app.py
import streamlit as st
import sys
import os

# Add the current directory to path to import your modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_pipeline.data_loader import load_wikipedia_documents
from data_pipeline.llm_processor import extract_graph_documents
from data_pipeline.graph_client import get_neo4j_client, insert_graph_data, graph_is_not_empty
from rag_system.chain import get_rag_chain

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False

def load_data():
    """Load data into Neo4j if not already loaded"""
    if not st.session_state.data_loaded:
        with st.spinner("Checking database..."):
            if graph_is_not_empty():
                st.success("✓ Database already contains data")
                st.session_state.data_loaded = True
                return True
            else:
                st.info("Database is empty. Loading data about Elizabeth I...")
                
                with st.spinner("Loading Wikipedia documents..."):
                    documents = load_wikipedia_documents("Elizabeth I")
                
                if not documents:
                    st.error("Failed to load documents")
                    return False
                
                with st.spinner("Extracting graph data..."):
                    graph_docs = extract_graph_documents(documents)
                
                with st.spinner("Inserting into Neo4j..."):
                    graph = get_neo4j_client()
                    insert_graph_data(graph, graph_docs)
                
                st.session_state.data_loaded = True
                st.success("✓ Data loaded successfully!")
                return True
    return True

def initialize_rag_chain():
    """Initialize the RAG chain"""
    if st.session_state.chain is None:
        with st.spinner("Initializing RAG system..."):
            st.session_state.chain = get_rag_chain()
        st.success("✓ RAG system ready!")

def display_chat_message(role, content):
    """Display a chat message"""
    with st.chat_message(role):
        st.markdown(content)

def main():
    st.set_page_config(
        page_title="RAG Knowledge Graph Chat",
        page_icon="🧠",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🧠 RAG Knowledge Graph Chat")
    st.markdown("Chat with your knowledge graph about Elizabeth I and related historical figures")
    
    # Sidebar for information and controls
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This RAG system combines:
        - **Structured data** from Neo4j knowledge graph
        - **Unstructured data** from Wikipedia documents
        - **LLM-powered** question answering
        """)
        
        st.header("Controls")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("Reload Data"):
            st.session_state.data_loaded = False
            st.session_state.chain = None
            st.rerun()
    
    # Main chat area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Chat Interface")
        
        # Load data and initialize chain
        if load_data():
            initialize_rag_chain()
            
            # Display chat messages
            for message in st.session_state.messages:
                display_chat_message(message["role"], message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask a question about Elizabeth I..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                display_chat_message("user", prompt)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            # Prepare chat history for the chain
                            chat_history = []
                            for i in range(0, len(st.session_state.messages) - 1, 2):
                                if i + 1 < len(st.session_state.messages):
                                    user_msg = st.session_state.messages[i]
                                    assistant_msg = st.session_state.messages[i + 1]
                                    if user_msg["role"] == "user" and assistant_msg["role"] == "assistant":
                                        chat_history.append((user_msg["content"], assistant_msg["content"]))
                            
                            chain_input = {"question": prompt}
                            if chat_history:
                                chain_input["chat_history"] = chat_history
                            
                            response = st.session_state.chain.invoke(chain_input)
                            
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            
                        except Exception as e:
                            error_msg = f"Sorry, I encountered an error: {str(e)}"
                            st.markdown(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    with col2:
        st.subheader("Sample Questions")
        st.markdown("""
        Try asking:
        - Who was Elizabeth I?
        - Which house did she belong to?
        - Who were her parents?
        - When was she born?
        - Who succeeded her?
        - What was the Elizabethan era?
        - Tell me about her relationship with Mary Queen of Scots
        - What was the Spanish Armada?
        """)
        
        st.subheader("Quick Actions")
        if st.button("Who was Elizabeth I?"):
            st.session_state.messages.append({"role": "user", "content": "Who was Elizabeth I?"})
            st.rerun()
        
        if st.button("Tell me about her family"):
            st.session_state.messages.append({"role": "user", "content": "Tell me about Elizabeth I's family"})
            st.rerun()
        
        if st.button("What was her reign known for?"):
            st.session_state.messages.append({"role": "user", "content": "What was Elizabeth I's reign known for?"})
            st.rerun()

if __name__ == "__main__":
    main()