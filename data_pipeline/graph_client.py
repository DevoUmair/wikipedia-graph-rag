from langchain_neo4j import Neo4jGraph
from config import (
    NEO4J_URI, 
    NEO4J_USERNAME, 
    NEO4J_PASSWORD
)
from neo4j import GraphDatabase



def graph_is_not_empty() -> bool:
    """Return True if Neo4j already contains ANY nodes."""
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    query = "MATCH (n) RETURN COUNT(n) AS count"

    with driver.session() as session:
        result = session.run(query).single()
        count = result["count"]

    driver.close()

    return count > 0


def get_neo4j_client():
    """Connect to Neo4j database."""
    
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )
    
    print("Connected to Neo4j.")
    return graph


def insert_graph_data(graph, graph_documents):
    """Insert extracted graph documents into Neo4j."""
    
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )

    print("Graph documents inserted successfully.")
