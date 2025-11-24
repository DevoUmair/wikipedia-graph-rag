from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import TokenTextSplitter


def load_wikipedia_documents(query: str):
    """Load and split data from Wikipedia."""
    
    raw_docs = WikipediaLoader(query=query).load()

    splitter = TokenTextSplitter(
        chunk_size=512,
        chunk_overlap=24
    )

    documents = splitter.split_documents(raw_docs[:3])
    return documents
