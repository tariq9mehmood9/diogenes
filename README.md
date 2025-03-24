## Quick Start

Follow these steps to get the project up and running:

### 1. Populate the Neo4j Graph
1. Ensure you have Docker installed and running on your system.
2. Start a Neo4j container:
    ```bash
    docker run --name neo4j -p 7474:7474 -p 7687:7687 -d \
    -e NEO4J_AUTH=neo4j/test \
    -e NEO4JLABS_PLUGINS='["apoc"]' \
    neo4j:latest
    ```
3. Populate the graph database with the best_practices data. This can be done by running the following command:
    ```bash
    python3 populate_neo4j.py
    ```

### 2. Set Up OpenAI API
1. Create a `.env` file in the project root directory and add your OpenAI API key:
    ```
    OPENAI_API_KEY=your_openai_api_key_here
    ```

### 3. Run the Streamlit Application
1. Install the required Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Start the Streamlit application:
    ```bash
    streamlit run st.py
    ```

### Notes
- Ensure the Neo4j container is running before starting the Streamlit application.
- Update the Neo4j connection details in the application configuration if necessary.

You're all set!