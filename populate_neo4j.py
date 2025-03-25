from docx import Document as DocxDocument
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv
import re
import os

load_dotenv()

NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USER, password=NEO4J_PASSWORD)


def read_docx(file_path):
    doc = DocxDocument(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text


def parse_best_practices(text):
    principles = []
    current_principle = None

    lines = text.strip().split("\n")  # Split by lines
    for line in lines:
        line = line.strip()

        principle_match = re.match(r"^Principle (\d+\.\d+): (.+)", line)
        practice_match = re.match(r"^Practice (\d+\.\d+): (.+)", line)

        if principle_match:
            if current_principle:
                principles.append(current_principle)
            current_principle = {
                "id": principle_match.group(1),
                "name": principle_match.group(2),
                "practices": [],
            }
        elif practice_match and current_principle:
            current_principle["practices"].append(
                {"id": practice_match.group(1), "description": practice_match.group(2)}
            )

    if current_principle:
        principles.append(current_principle)

    return principles


best_practices_text = read_docx(
    "./data/The CyberGov™ Framework – Optimizing Your Cybersecurity Posture v. 8.0 14 Dec 2023.docx"
)
principles = parse_best_practices(best_practices_text)
# workaround for an anomaly in the data
# principles[2]['practices'][1]['id'] = '3.1.1'

# List of schema queries
queries = [
    """
    CREATE CONSTRAINT unique_principle_id IF NOT EXISTS 
    FOR (p:Principle) REQUIRE p.id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT unique_practice_id IF NOT EXISTS 
    FOR (pr:Practice) REQUIRE pr.id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT unique_keyindicator_details IF NOT EXISTS 
    FOR (ki:KeyIndicator) REQUIRE ki.details IS UNIQUE
    """,
]

for query in queries:
    graph.query(query)

# Inserting principles and practices into Neo4j
for principle in principles:
    # Ensure Principle node is created or matched
    graph.query(
        """
        MERGE (p:Principle {id: $principle_id})
        ON CREATE SET p.name = $principle_name
        """,
        params={"principle_id": principle["id"], "principle_name": principle["name"]},
    )

    for practice in principle["practices"]:
        graph.query(
            """
            MATCH (p:Principle {id: $principle_id})  // Ensure Principle exists
            MERGE (pr:Practice {id: $practice_id})  // Ensure unique Practice by ID
            ON CREATE SET pr.description = $practice_desc  // Set description only on creation
            MERGE (p)-[:HAS_PRACTICE]->(pr)  // Create relationship
            """,
            params={
                "principle_id": principle["id"],
                "practice_id": practice["id"],
                "practice_desc": practice["description"],
            },
        )

# key indicators mapped to their corresponding practice IDs (provided by Bob)
key_indicators = {
    "2.6": [
        "How can we guarantee that all subsidiaries fully implement cybersecurity communication channels?",
        "What barriers might delay the complete deployment of these communication frameworks, and how can they be mitigated?",
        "How do we foster greater trust among suppliers and third parties to encourage transparency in cybersecurity risk sharing?",
        "Could leveraging contractual obligations improve data-sharing practices with external partners?",
    ],
    "3.4": [
        "What mechanisms can be implemented to extend supply chain cybersecurity risk management to international vendors?",
        "What challenges might arise from merging cybersecurity risk with enterprise risk management, and how can they be resolved?",
        "How can board members be encouraged to perceive cybersecurity as a key component of corporate governance rather than a standalone function?",
    ],
    "4.5": [
        "How can we better model cyber risks to enhance response planning in unpredictable scenarios?",
        "What initiatives can be introduced to align staff and board perspectives on a unified incident response strategy?",
        "How can we ensure that real-world data collection is comprehensive and accessible across all business units?",
    ],
}

# Insert key indicators and establish relationships
for practice_id, questions in key_indicators.items():
    for question in questions:
        graph.query(
            """
            MATCH (pr:Practice {id: $practice_id})
            MERGE (ki:KeyIndicator {question: $question})
            MERGE (pr)-[:HAS_KEY_INDICATOR]->(ki)
            """,
            params={"practice_id": practice_id, "question": question},
        )

result = graph.query(
    """
    MATCH (n) RETURN count(n) AS node_count;
    """
)

if result:
    print(
        "Graph populated successfully! No. of nodes inserted:", result[0]["node_count"]
    )
