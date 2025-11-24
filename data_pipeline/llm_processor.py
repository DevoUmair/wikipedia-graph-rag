import sys
import os
from langchain_groq import ChatGroq
from langchain_experimental.graph_transformers import LLMGraphTransformer
from config import GROQ_API_KEY


def extract_graph_documents(documents):
    """Initialize Groq LLM, transform text into graph documents."""
    
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile"
    )

    transformer = LLMGraphTransformer(llm)

    graph_docs = transformer.convert_to_graph_documents(documents)
    return graph_docs
