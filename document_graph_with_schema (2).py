import hashlib
import json
import re
import os
import logging
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph

load_dotenv()


def setup_logging():
    """Set up basic logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_pdf_text(pdf_docs):
    """Extract text from PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(stream=pdf)
        text += " ".join(page.extract_text() for page in pdf_reader.pages if page.extract_text() is not None)
    return text


def get_text_chunks(text):
    """Split text into chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=2000,
        chunk_overlap=300,
        length_function=len
    )
    return text_splitter.split_text(text)


def clean_text(text):
    """Clean text by removing non-ASCII characters and HTML tags."""
    text_no_ascii = re.sub(r'[^\x00-\x7F]+', ' ', text)
    cleaned_text = re.sub(r'<[^>]+>', ' ', text_no_ascii)
    return cleaned_text


def get_entity_extraction_chain():
    """Create an entity extraction chain using Google Generative AI."""
    llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.1)
    memory = ConversationBufferWindowMemory(k=2, input_key="text")
    chain = LLMChain(llm=llm, verbose=True, prompt=get_prompt(), memory=memory)
    return chain


def get_prompt():
    """Get prompt template for the LLM."""
    template = """
    You're a data scientist at a company developing a graph database. Your job is to extract data and convert it into a graph database schema. 

    **Task:**  
    Given the data, create nodes and relationships according to the schema.
    **Guidelines:**  
    - If there's no data for a property, set it to null.
    - Don't fabricate data or add extra information.
    - Include only nodes and relationships from the schema.
    - If no relationships are in the schema, only add nodes.
    - Update the previous output based on new context if necessary, otherwise extract new nodes and edges with new unique IDs.

    **Example:**  
    Schema: {Example}  
    Text: Alice is 25 years old and Bob is her roommate.  
    Output: {output}

    **Given Schema:** {schema}  
    Description: {desc}

    **Guidelines for Previous Output and Context:**  
    - Update the previous output based on new context if necessary. For example, if there are new relationships or properties for existing nodes in the new context, capture that and adjust the existing information accordingly. Otherwise, extract new nodes and edges with new incremental IDs.
    **Previous Output and Context:** 
    {history}

    **More Context:**  
    {text}

    **Note:**  
    Please ensure the output format is strictly as below in plain string which can be parsed in Python data structures. I repeat, please ensure the output is well structured like a Python dictionary but as plain text:
    {output}
    """
    prompt = PromptTemplate(template=template, input_variables=["schema", "text", "Example", "desc", "output", "history"])
    return prompt


def read_json_schema(file_path):
    """Read JSON schema from a file."""
    with open(file_path, 'r') as file:
        return json.load(file)


def get_prompt_definitions(schema_file_path, description=None):
    """Get prompt definitions including example schema, output, and actual schema."""
    example_schema = """{"Nodes":[{"type":"Person","properties":{"id":"integer","age":"integer","name":"string"}}],"Edges":[{"source":"Person","target":"Person","source_id":"integer","target_id":"integer","relationship":"roommate"}]}"""
    example_output = """{"Nodes": [{"type":"Person","properties":{"id":1,"age":25,"name":"Alice"}},{"type":"Person","properties":{"id":2,"age":null,"name":"Bob"}}],"Edges": [{"source":"Person","target":"Person","source_id":1,"target_id":2,"relationship":"roommate"}]}"""
    schema = read_json_schema(schema_file_path)
    desc = description if description else "There is no description provided. Please use the schema to extract the information."
    return example_schema, example_output, schema, desc


def extract_data(doc_path, schema_path, desc=None):
    """Extract data from the document based on the schema."""
    text = get_pdf_text([doc_path])
    chunks = get_text_chunks(text)
    chain = get_entity_extraction_chain()
    example_schema, example_output, schema, desc = get_prompt_definitions(schema_path, desc)

    nodes = []
    relations = []
    for chunk in chunks:
        try:
            result = chain.run(text=chunk, schema=schema, Example=example_schema, output=example_output, desc=desc)
            if result:
                result = eval(result.replace("\n", ""))
                nodes.extend(result.get('Nodes', []))
                relations.extend(result.get('Edges', []))
        except Exception as e:
            logging.info(f"Result parsing error or incomplete response generated: {e}")

    logging.info("Nodes: %s", nodes)
    logging.info("Relations: %s", relations)

    return nodes, relations


def create_node_cypher(result, doc_path, doc_id):
    """Create Cypher queries for nodes."""
    doc_url_hash_val = hashlib.md5(doc_path.encode("utf-8")).hexdigest()
    node_queries = [f"MERGE (n:Document {{ id: {doc_id}, url: '{doc_path}', url_hash: '{doc_url_hash_val}' }})"]

    for item in result:
        label = item['type']
        properties = item['properties']
        props = []
        for prop_key, prop_val in properties.items():
            if prop_key != 'id':
                if isinstance(prop_val, str):
                    prop_val = prop_val.replace("'", "")  # Remove single quotes for string values
                props.append(f"{prop_key}: '{prop_val}'")
        props_string = ", ".join(props)
        query = f"MERGE (n:{label} {{ id: {properties['id']}, {props_string} }})"
        node_queries.append(query)
        query = f"MATCH (n:Document{{id:{doc_id}}}), (m:{label} {{ id: {properties['id']}, {props_string} }})" \
                f"MERGE (n)<-[:HAS_DOCUMENT]-(m);"
        node_queries.append(query)

    return node_queries


def create_edge_cypher(result):
    """Create Cypher queries for edges."""
    edge_queries = []
    for item in result:
        try:
            source = item['source']
            target = item['target']
            source_id = item['source_id']
            target_id = item['target_id']
            relationship = item['relationship']

            query = f"MATCH (n:{source} {{id: {source_id}}}), (m:{target} {{id: {target_id}}}) " \
                    f"MERGE (n)-[:{relationship}]->(m);"
            edge_queries.append(query)
        except KeyError as e:
            logging.error(f"Missing key in edge data: {e}")
            continue

    return edge_queries


def neo4j_conn(URL, username, password, database):
    """Establish connection to the Neo4j database."""
    try:
        graph = Neo4jGraph(
            url=URL,
            username=username,
            password=password,
            database=database)
        logging.info('Neo4j connection successful')
        return graph
    except Exception as e:
        logging.error('Neo4j connection failed: %s', e)
        return None


def write_graph(URL, username, password, database, doc_path, doc_id, schema_path, desc=None):
    """Write extracted data into the Neo4j database."""
    graph = neo4j_conn(URL, username, password, database)
    if graph:
        node_data, edge_data = extract_data(doc_path, schema_path, desc)
        node_queries = create_node_cypher(node_data, doc_path, doc_id)
        edge_queries = create_edge_cypher(edge_data)

        try:
            node_count = 0
            for query in node_queries:
                try:
                    graph.query(query)
                    node_count += 1
                    logging.info(f"Successfully created node: {query}")
                except Exception as e:
                    logging.info(f"Cannot create node with query: {query} : {e}")

            edge_count = 0
            for query in edge_queries:
                try:
                    graph.query(query)
                    edge_count += 1
                    logging.info(f"Successfully created relationship: {query}")
                except Exception as e:
                    logging.info(f"Cannot create relationship with query: {query} : {e}")

            logging.info("Graph created successfully")

            response = {
                "status": "success",
                "message": f"Successfully created graph for document: {os.path.basename(doc_path)}",
                "data": {
                    "text": None
                },
                "meta": {
                    "nodeCount": node_count,
                    "edgeCount": edge_count,
                    "total": None,
                    "type": None,
                    "query": None
                },
                "error": {
                    "code": None,
                    "message": None,
                    "details": None
                }
            }
            return json.dumps(response)

        except Exception as e:
            logging.error(f"Error occurred while writing graph: {e}")
            return json.dumps({
                "status": "error",
                "message": "Request failed",
                "data": [],
                "meta": [],
                "error": {
                    "code": None,
                    "message": str(e),
                    "details": None
                }
            })


setup_logging()
