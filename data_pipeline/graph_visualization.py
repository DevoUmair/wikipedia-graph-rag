from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from neo4j import GraphDatabase
from yfiles_jupyter_graphs_for_neo4j import Neo4jGraphWidget as GraphWidget

default_cipher = "MATCH (s)-[r]->(t) RETURN s, r, t LIMIT 50"

def show_graph(cipher: str = default_cipher) -> GraphWidget:
    """
    Show a Neo4j graph in Jupyter or exportable to HTML.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    widget = GraphWidget(driver=driver)
    widget.show_cypher(cipher)
    
    return widget
