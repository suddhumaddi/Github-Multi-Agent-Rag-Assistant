# 📚 Multi-Agent RAG: AI Publication Assistant

## Project Overview

The **AI Publication Assistant** is a multi-agent system designed to enhance the discoverability, clarity, and completeness of technical projects shared on platforms like GitHub. By integrating **Retrieval-Augmented Generation (RAG)** into an orchestrated workflow, the system analyzes a provided public repository URL and generates grounded, actionable suggestions for improving the project's title, summary, and `README.md` structure.

This system demonstrates mastery of **Module 2: Agent Orchestration and Tool Integration** by implementing a minimum of three specialized agents coordinated by **LangGraph**.

---

## 🛠️ System Architecture and Agent Roles

The system uses a sequential **LangGraph** flow to manage the collaboration and data handoff between three specialized agents.

### Agent Collaboration Flow

1.  **Repo Analyzer** clones the repo and generates the RAG knowledge base.
2.  **Metadata Recommender** extracts keywords from the content provided by Agent 1.
3.  **Content Improver** uses the RAG knowledge base (from Agent 1) and the keywords (from Agent 2) to generate the final, grounded output.

| Agent | Role & Responsibility | Key Output Handover |
| :--- | :--- | :--- |
| **1. RepoAnalyzerAgent** | **File Processing & RAG Initialization.** Clones the public target repo, parses key files (`README.md`, `main.py`), performs **Text Chunking**, and initializes the embedding model to create the Retriever Tool. | Passes **Retriever Object** and **Original Content**. |
| **2. MetadataRecommenderAgent** | **Keyword & Tag Extraction.** Performs lightweight NLP (NLTK) on the original content to identify core project topics, keywords, and category suggestions. | Passes **Metadata Dictionary** (Keywords, Tags, Categories). |
| **3. ContentImproverAgent** | **LLM Generation & Structured Output.** Consumes the Retriever (for grounding) and Metadata (for scope). Queries the OpenRouter LLM (`GPT-4o-mini`) using Pydantic schemas to generate the final, structured improvement suggestions. | Returns **Structured Suggestions** (Title, Summary, Edits). |

### Tool Integration (4 Distinct Tools)

The system integrates four different tools, exceeding the project minimum of three:

1.  **Repo Reader Tool:** Uses `gitpython` and LangChain's `TextLoader` for file system cloning and content parsing.
2.  **RAG Retriever Tool:** A custom tool built on **`HuggingFaceEmbeddings`** (`all-MiniLM-L6-v2`) and **`FAISS`** for vector creation and efficient semantic search.
3.  **Keyword Extractor Tool:** A custom NLP tool using **`nltk`** for fast frequency analysis and keyword identification.
4.  **LLM Generation API:** Connects to the **OpenRouter** API (targeting `GPT-4o-mini`) for complex reasoning and guaranteed structured output.

---

## ⚙️ Setup and Installation Instructions

This section fulfills the "Essential" criteria for basic installation and usability.

### Prerequisites

You must have the following software installed:

1.  **Python 3.10+**
2.  **Git Command-Line Tool** (Crucial for cloning repositories).

### 1. Project Initialization

1.  Clone this repository locally:
    ```bash
    git clone [YOUR REPO URL HERE]
    cd multi_agent_rag
    ```
2.  Install all required Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Download NLTK data (required for keyword extraction in Agent 2):
    ```bash
    python -c "import nltk; nltk.download('stopwords')"
    ```

### 2. Secure Configuration

To run the LLM agent, you must secure your API key.

1.  Create a file named **`.env`** in the root directory.
2.  Add your OpenRouter API key inside this file:

    ```text
    OPENROUTER_API_KEY="YOUR_API_KEY_HERE"
    ```
    *The system securely manages this environment variable using the `python-dotenv` library.*

---

## 🚀 Usage Instructions (Streamlit Web App)

The application runs locally as a Streamlit web interface, providing a smooth user experience.

1.  **Launch the App:** Open your terminal in the project root and run:
    ```bash
    python -m streamlit run app.py
    ```
2.  **Run as Administrator (Recommended):** For best results on Windows, run your terminal (PowerShell/CMD) as **Administrator** before executing the launch command to avoid **`[WinError 5] Access is denied`** errors during cloning.
3.  **Open Browser:** The application will open automatically in your browser (usually at `http://localhost:8501`).
4.  **Analyze a Repository:** Paste a **public GitHub URL** (e.g., `https://github.com/justin-hu/micro-repo-for-testing`) into the input box and click **"🚀 Start Analysis."**

---

## 📝 Technical RAG Configuration

This section addresses the need for specific details on text processing and configuration management (as per Module 1 feedback).

| Setting | Value | Description |
| :--- | :--- | :--- |
| **Retriever Type** | `FAISS` | Fast vector similarity search for low-latency retrieval. |
| **Embedding Model** | `HuggingFaceEmbeddings(all-MiniLM-L6-v2)` | Efficient, small, and locally run model for creating dense vector representations of the repository content. |
| **Text Chunk Size** | `1000` | Optimal size for preserving contextual blocks of code and documentation text. |
| **Text Chunk Overlap** | `200` | Provides necessary overlap to ensure semantic continuity across split chunks, crucial for RAG quality. |