from data_pipeline.data_loader import load_wikipedia_documents
from data_pipeline.llm_processor import extract_graph_documents
from data_pipeline.graph_client import get_neo4j_client, insert_graph_data, graph_is_not_empty
from rag_system.chain import get_rag_chain

def data_pipeline():
    """Run the data loading pipeline"""
    if graph_is_not_empty():
        print("Neo4j already contains graph data. Skipping pipeline.\n")
        return True
    else:
        print("Graph is empty. Running full pipeline...\n")

        documents = load_wikipedia_documents("Elizabeth I")
        if not documents:
            print("No documents were loaded. Exiting.")
            return False
        
        graph_docs = extract_graph_documents(documents)
        graph = get_neo4j_client()
        insert_graph_data(graph, graph_docs)
        print("\nPipeline completed successfully!\n")
        return True

def chat_interface():
    """Interactive chat interface with the RAG system"""
    chain = get_rag_chain()
    chat_history = []
    
    print("\n" + "="*50)
    print("RAG Knowledge Graph Chat Interface")
    print("Type 'quit' to exit")
    print("="*50)
    
    while True:
        question = input("\nQuestion: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
            
        if not question:
            continue
            
        try:
            chain_input = {"question": question}
            if chat_history:
                chain_input["chat_history"] = chat_history
            
            response = chain.invoke(chain_input)

            print(f"\nAnswer: {response}")
            
            chat_history.append((question, response))
            
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main function to run data pipeline and start chat interface"""
    success = data_pipeline()
    
    if success:
        chat_interface()
    else:
        print("Data pipeline failed. Cannot start RAG system.")

if __name__ == "__main__":
    main()