from typing import List, Tuple
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser

from .retriever import RAGRetriever

def get_rag_chain():
    """Initialize and return the RAG chain"""
    retriever = RAGRetriever()
    
    # Question condensation
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
    in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    
    def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content=human))
            buffer.append(AIMessage(content=ai))
        return buffer
    
    _search_query = RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            RunnablePassthrough.assign(
                chat_history=lambda x: _format_chat_history(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | retriever.llm
            | StrOutputParser(),
        ),
        RunnableLambda(lambda x: x["question"]),
    )
    
    # Final answer generation
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    Use natural language and be concise.
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Main chain
    chain = (
        RunnableParallel(
            {
                "context": _search_query | retriever.combined_retriever,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | retriever.llm
        | StrOutputParser()
    )
    
    return chain

# For direct usage without chat history
def get_simple_chain():
    """Get a simple chain without chat history support"""
    retriever = RAGRetriever()
    
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    Use natural language and be concise.
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        RunnableParallel(
            {
                "context": lambda x: retriever.combined_retriever(x["question"]),
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | retriever.llm
        | StrOutputParser()
    )
    
    return chain