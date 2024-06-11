import os
import logging
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from py2neo import Graph, Node, Relationship
from langchain_google_genai import GoogleGenerativeAI


# Configure logging to capture debug-level messages
logging.basicConfig(level=logging.DEBUG)

# Set up Neo4j connection using environment variables (recommended)
uri = os.environ.get("NEO4J_URI")
username = os.environ.get("NEO4J_USERNAME")
password = os.environ.get("NEO4J_PASSWORD")
database = os.environ.get("NEO4J_DATABASE")

try:
    # Initialize connection to the Neo4j graph database
    graph = Graph(uri, auth=(username, password), name=database)
    logging.debug("Successfully connected to Neo4j database")
except Exception as e:
    logging.error(f"Failed to connect to Neo4j database: {e}")
    exit()

# Replace with your actual Google API key (if using Google Generative AI)
google_api_key = "YOUR_GOOGLE_API_KEY"  # Placeholder for Google API key (if applicable)

# Initialize LLM model (replace with your specific LLM and configuration)
llm = PLACEHOLDER_LLM(model="models/text-bison-001", google_api_key=google_api_key, temperature=0.1)  # Replace with your LLM details

# Initialize LLMGraphTransformer with your LLM
llm_transformer = LLMGraphTransformer(llm=llm)

# Define the text to be converted into graph documents
text = """
Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
She was, in 1906, the first woman to become a professor at the University of Paris.
"""

# Create a document from the text
documents = [Document(page_content=text)]

print("Text Document:")
print(documents)

try:
    # Convert text document into graph documents using LLMGraphTransformer
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
except KeyError as e:
    # Log any errors encountered during the conversion process
    logging.error(f"Error occurred while processing response: {e}")
    logging.debug(f"Response data: {documents}")
    exit()

# Print the response data for debugging
print("Graph Documents:")
print(graph_documents)

# If graph_documents were successfully created, print details of the nodes and relationships
if graph_documents:
    for doc in graph_documents:
        print("Relationships:")
        for rel in doc.relationships:
            # Log the relationship type
            logging.debug(f"Relationship type: {rel.type}")
            # Log the start and end nodes of each relationship
            if hasattr(rel, 'source') and hasattr(rel, 'target'):
                logging.debug(f"Start node: {rel.source}")
                logging.debug(f"End node: {rel.target}")
            else:
                logging.debug("Missing source or target node information")

    # Store the nodes and relationships in Neo4j
    for doc in graph_documents:
        for node in doc.nodes:
            # Create or merge node in the graph
            node_obj = Node(node.type, id=node.id)
            graph.merge(node_obj, node.type, "id")
        for rel in doc.relationships:
            # Create or merge relationship in the graph
            source_node = graph.nodes.match(id=rel.source.id).first()
            target_node = graph.nodes.match(id=rel.target.id).first()
            if source_node and target_node:
                relationship = Relationship(source_node, rel.type, target_node)
                graph.merge(relationship)
    logging.debug("Graph stored in Neo4j successfully.")
