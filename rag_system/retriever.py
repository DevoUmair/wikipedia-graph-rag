from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jVector

from data_pipeline.graph_client import get_neo4j_client
from config import GROQ_API_KEY
from .schemas import Entities

class RAGRetriever:
    def __init__(self):
        self.graph = get_neo4j_client()
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile"
        )

        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.vector_index = Neo4jVector.from_existing_graph(
            embedding=self.embedding,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        
        self.entity_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are extracting organization and person entities from the text."),
            ("human", "Use the given format to extract information from the following input: {question}"),
        ])
        self.entity_chain = self.entity_prompt | self.llm.with_structured_output(Entities)

    def structured_retriever(self, question: str) -> str:
        """Search for entity relationships directly"""
        result = ""
        entities = self.entity_chain.invoke({"question": question})
        print(f"Extracted entities: {entities.names}")
        
        for entity in entities.names:
            print(f"Searching for relationships of '{entity}'")
            
            try:
                response = self.graph.query(
                    """
                    MATCH (entity:__Entity__)
                    WHERE toLower(entity.id) CONTAINS toLower($entity_name)
                    WITH entity
                    LIMIT 1
                    OPTIONAL MATCH (entity)-[out_rel]->(target:__Entity__)
                    WHERE type(out_rel) <> 'MENTIONS'
                    OPTIONAL MATCH (entity)<-[in_rel]-(source:__Entity__)
                    WHERE type(in_rel) <> 'MENTIONS'
                    RETURN 
                        CASE 
                            WHEN out_rel IS NOT NULL THEN entity.id + ' - ' + type(out_rel) + ' -> ' + target.id
                            WHEN in_rel IS NOT NULL THEN source.id + ' - ' + type(in_rel) + ' -> ' + entity.id
                        END AS relationship
                    """,
                    {"entity_name": entity},
                )
                
                if response:
                    relationships = [rel['relationship'] for rel in response if rel['relationship']]
                    if relationships:
                        result += "\n".join(relationships) + "\n"
                    else:
                        print(f"No relationships found for '{entity}'")
                else:
                    print(f"Entity '{entity}' not found")
                    
            except Exception as e:
                print(f"Error searching for '{entity}': {e}")
                
        return result.strip()

    def combined_retriever(self, question: str) -> str:
        """Combine structured and unstructured retrieval"""
        print(f"Search query: {question}")
        structured_data = self.structured_retriever(question)
        print(f"Structured data:\n{structured_data}\n")
        unstructured_data = [el.page_content for el in self.vector_index.similarity_search(question)]
        
        final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ".join(unstructured_data)}"""
        print(f"Combined data:\n{final_data}\n")
        return final_data