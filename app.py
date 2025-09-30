import streamlit as st
import json
import os
import time
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# Import Agents
from agents.repo_analyzer import RepoAnalyzerAgent
from agents.metadata_recommender import MetadataRecommenderAgent
from agents.content_improver import ContentImproverAgent


# --- AGENT AND GRAPH LOGIC ---

# Define the state of our graph
class AgentState(TypedDict):
    """Represents the shared state passed between agents."""
    repo_url: str
    original_content: str
    chunks: list
    retriever: object
    metadata: dict
    improved_content: dict

# Define the agent nodes (functions)
def analyze_repo_node(state: AgentState):
    st.info("Agent 1: Analyzing repository structure and creating RAG index...")
    repo_url = state['repo_url']
    agent = RepoAnalyzerAgent(repo_url)
    
    try:
        chunks = agent.process_repo()
    except Exception as e:
        st.error(f"Cloning/Analysis Error: Ensure Git is installed and the repository is public. Error: {e}")
        raise ValueError("RepoAnalyzer failed.") 
        
    if not chunks:
        st.error("RepoAnalyzer failed to process any content. Stopping workflow.")
        raise ValueError("RepoAnalyzer failed.")

    retriever = agent.create_retriever(chunks)
    full_content = "\n\n".join([c.page_content for c in chunks])
    st.success("Analysis Complete. RAG index created.")
    
    return {
        "original_content": full_content,
        "chunks": chunks,
        "retriever": retriever,
    }

def recommend_metadata_node(state: AgentState):
    st.info("Agent 2: Extracting keywords and suggesting metadata...")
    original_content = state['original_content']
    agent = MetadataRecommenderAgent()
    metadata = agent.suggest_metadata(original_content)
    st.success("Metadata extraction complete.")
    
    return {
        "metadata": metadata
    }

def improve_content_node(state: AgentState):
    st.info("Agent 3: Generating new content and README edits using OpenRouter LLM...")
    original_content = state['original_content']
    metadata = state['metadata']
    retriever = state['retriever']
    
    with st.spinner('Waiting for OpenRouter LLM response... (This may take a moment)'):
        agent = ContentImproverAgent(retriever)
        improved_content = agent.generate_improved_content(original_content, metadata)
    
    st.success("Content generation complete.")
    
    return {
        "improved_content": improved_content
    }

def create_graph():
    """Defines and compiles the sequential LangGraph workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("analyze_repo", analyze_repo_node)
    workflow.add_node("recommend_metadata", recommend_metadata_node)
    workflow.add_node("improve_content", improve_content_node)

    workflow.add_edge(START, "analyze_repo")
    workflow.add_edge("analyze_repo", "recommend_metadata")
    workflow.add_edge("recommend_metadata", "improve_content")
    workflow.add_edge("improve_content", END)

    return workflow.compile()
# --- END AGENT AND GRAPH LOGIC ---


# --- STREAMLIT UI DEFINITION ---
def main():
    st.set_page_config(page_title="Gen-Authoring: Repo Improver", layout="wide")
    
    st.title("📚 Gen-Authoring: AI-Powered Repo Improvement")
    st.markdown("An automated multi-agent system to analyze GitHub repos and suggest content edits.")
    st.markdown("---")

    # Load environment variables
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        st.error("FATAL ERROR: OPENROUTER_API_KEY is not set in the .env file. Please check your environment setup.")
        return

    # Sidebar for config overview
    st.sidebar.title("System Configuration")
    st.sidebar.markdown(f"**LLM Provider:** OpenRouter (GPT-3.5-Turbo)")
    st.sidebar.markdown(f"**Embedding Model:** HuggingFace MiniLM")
    st.sidebar.markdown(f"**Orchestration:** LangGraph (3 Agents)")
    
    # ------------------
    # 1. Input Section
    # ------------------
    st.header("1. Project Setup")
    
    repo_url = st.text_input(
        "GitHub Repository URL", 
        value="https://github.com/langchain-ai/langgraph",
        placeholder="https://github.com/username/repository-name",
        help="Paste a public GitHub URL here. Ensure Git is installed on your machine to clone the repo."
    )
    
    # Button to start the analysis
    if st.button("🚀 Start Analysis", type="primary"):
        if not repo_url or "github.com" not in repo_url:
            st.error("Please enter a valid GitHub repository URL.")
            return

        # Initialize and run the graph
        st.subheader("2. Agent Workflow Execution")
        
        # Ensure 'data' directory exists for repo cloning
        if not os.path.exists("./data"):
            os.makedirs("./data")
            
        app = create_graph()
        inputs = {"repo_url": repo_url}
        
        start_time = time.time()
        
        # Use a Streamlit container for status messages
        status_container = st.container()
        
        try:
            # Run the entire pipeline
            final_state = app.invoke(inputs)

            status_container.success("All Agents Completed Successfully!")
            st.markdown("---")
            st.subheader("3. Final Suggestions")
            
            # Extract results
            results = final_state.get('improved_content', {})
            metadata = final_state.get('metadata', {})

            # Display Metadata
            with st.expander("Suggested Metadata & Tags"):
                st.json(metadata)

            # Display LLM Improvements
            st.markdown("### 🌟 Repository Core Content")
            st.markdown(f"**New Title Suggestion:**")
            st.code(results.get('new_title', 'N/A'), language='markdown')
            
            st.markdown(f"**Short Summary (Repo Description):**")
            st.markdown(f"> *{results.get('short_summary', 'N/A')}*")

            st.markdown("### 📝 Actionable README Edits")
            edits = results.get('readme_edits', [])
            
            if isinstance(edits, list) and edits:
                st.success("Improvements suggested:")
                for i, edit in enumerate(edits):
                    st.markdown(f"**{i+1}.** {edit}")
            else:
                st.warning(f"Could not retrieve actionable edit list. Error in LLM output: {edits}")

            st.caption(f"Total execution time: {time.time() - start_time:.2f} seconds.")

        except ValueError as e:
            # Error was already shown in the specific node's code
            status_container.error("Workflow failed. Please resolve the issue (e.g., install Git or check URL).")
        except Exception as e:
            status_container.error(f"An unexpected critical error occurred: {e}")
            st.warning("Please check your terminal logs for more details.")


if __name__ == "__main__":
    main()