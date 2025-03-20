import streamlit as st
import docx
import os
from langchain_neo4j import Neo4jGraph
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LCDocument
from langchain.llms import OpenAI
from dotenv import load_dotenv
from pprint import pprint
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
import json
import re
from docx import Document as DocxDocument
from PyPDF2 import PdfReader

load_dotenv()


def read_docx(file_path):
    doc = DocxDocument(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])  # Extract all paragraphs
    return text


def read_pdf(file_path):
    """Reads and extracts text from a PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"  # Extract text from each page

    # Clean the text to remove special tokens that might cause issues
    text = text.replace("<|endoftext|>", "")
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


best_practices_file_path = "./data/The CyberGov™ Framework – Optimizing Your Cybersecurity Posture v. 8.0 14 Dec 2023.docx"
best_practices_text = read_docx(best_practices_file_path)
best_practices_doc = Document(
    best_practices_text, metadata={"source": best_practices_file_path}
)
principles = parse_best_practices(best_practices_text)


class ComplianceReport(BaseModel):
    status: str = Field(description="The compliance status: 'Pass' or 'Fail'")
    causality: str = Field(
        description="A to-the-point concise reason (Cause) for the compliance status (Effect)"
    )
    corrective_measures: str = Field(
        description="Suggested corrective measures if status is 'Fail', or empty if 'Pass'"
    )


# Initialize LangChain components
graph = Neo4jGraph(
    url="bolt://localhost:7687", username="neo4j", password="password"
)
embeddings = OpenAIEmbeddings(disallowed_special=())
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
)
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.1,
    model_kwargs={"response_format": {"type": "json_object"}},
)
parser = PydanticOutputParser(pydantic_object=ComplianceReport)
prompt_template = PromptTemplate(
    template="""
You are an expert compliance analyst tasked with evaluating the compliance status of the best practice based on the provided context. 
The context consists of relevant remarks from board members. Clearly state the status in your response as "Pass" or "Fail" at the top.
You will be provided with a key indicator and a practice statement. You need to evaluate the compliance status of the practice based on the key indicator.

### Best Practice:
{practice}

### Key Indicator:
{key_indicator}

### Context:
{context}

### Question:
Based on the context, does the organization comply with this best practice? Provide reasoning if it doesn't and corrective measures. Your description should be easy to comprehend. If you don't find any relevant information, you can state that as well.

Format your response as a JSON object with the following fields:
- status: "Pass" or "Fail"
- causality: A to-the-point concise reason (Cause) for the compliance status (Effect)
- corrective_measures: Suggested actions if failing, or empty string if passing

### Answer:
""",
            input_variables=["practice", "key_indicator", "context"],
        )
compliance_chain = LLMChain(
    llm=llm, prompt=prompt_template, output_key="compliance_report"
)

def check_compliance(practice_statement, key_indicator, vector_store, k=5):
    # Retrieve relevant documents
    retrieved_docs = vector_store.similarity_search_with_score(
        practice_statement, k=k
    )

    # Format retrieved documents into a structured context
    context = "\n\n".join(
        [
            f"Document {i+1} (score: {score}):\n{doc.page_content}"
            for i, (doc, score) in enumerate(retrieved_docs)
        ]
    )

    # Run the compliance chain
    result = compliance_chain.invoke(
        {
            "practice": practice_statement,
            "key_indicator": key_indicator,
            "context": context,
        }
    )

    # Parse the JSON string into a ComplianceReport object
    try:
        json_str = result["compliance_report"]
        parsed_json = json.loads(json_str)
        return ComplianceReport(**parsed_json)
    except Exception as e:
        return f"Failed to parse result into ComplianceReport model. Error: {e}\n"

# Streamlit App
def main():
    st.title("CyberGov Powered by Diogenes")
    
    uploaded_file = st.file_uploader("Upload a DOCX or PDF file", type=["pdf", "docx"])
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        if file_extension == "pdf":
            text = read_pdf(uploaded_file)
        elif file_extension == "docx":
            text = read_docx(uploaded_file)
        else:
            st.error("Unsupported file format.")
            return
        
        chunks = text_splitter.split_text(text)
        docs = [LCDocument(page_content=chunk) for chunk in chunks]
        vector_store_memo = FAISS.from_documents(docs, embeddings)

        for principle in principles:
            result = graph.query(
                """
                MATCH (p:Principle)-[:HAS_PRACTICE]->(pr:Practice)-[:HAS_KEY_INDICATOR]->(ki:KeyIndicator)
                WHERE p.id = $principle_id
                RETURN p, pr, ki;
                """,
                params={"principle_id": principle["id"]},
            )

            for record in result:
                practice = record["pr"]
                key_indicator = record["ki"]

                if key_indicator:
                    st.markdown(f"#### Practice {practice['id']}: {practice['description']}")
                    st.markdown(f"**Key Indicator:** {key_indicator['question']}")

                    report = check_compliance(
                        practice["description"],
                        key_indicator["question"],
                        vector_store_memo,
                    )
                    if report:
                        if report.status == "Pass":
                            st.markdown(f"**Status:** :green[{report.status}]")
                        else:
                            st.markdown(f"**Status:** :red[{report.status}]")
                        st.markdown(f"**Cause:** {report.causality}")
                        st.markdown(f"**Corrective measures:** {report.corrective_measures}")
                    
                    st.markdown("---")  # Horizontal line as separator

if __name__ == "__main__":
    main()
