import os
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, ValidationError 

# --- 1. Define the desired output structure using Pydantic ---
class ContentSuggestions(BaseModel):
    """Structured output model for repository content suggestions."""
    new_title: str = Field(description="A short, attention-grabbing, and descriptive title (max 10 words).")
    short_summary: str = Field(description="A compelling, one-paragraph summary for the repository description (max 80 words).")
    readme_edits: list[str] = Field(description="A list of 3-5 concrete, actionable suggestions for improving the README content or structure.")

class ContentImproverAgent:
    def __init__(self, retriever):
        # Switched to the highly reliable GPT-4o Mini model for guaranteed JSON output
        self.llm = ChatOpenAI(
            model="openai/gpt-4o-mini",  
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.0 
        )
        self.retriever = retriever

    def generate_improved_content(self, original_content: str, metadata: dict):
        """Generates structured improved content by enforcing Pydantic output."""
        
        retrieved_docs = self.retriever.invoke("summarize the repository and identify missing documentation sections")
        retrieved_context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
        
        prompt_template = PromptTemplate(
            template="""You are an expert GitHub repository analyst. Your task is to analyze the content and suggest improvements. The user requires the output to strictly adhere to the provided JSON schema.
            
            REPOSITORY METADATA & CONTEXT:
            {context}
            
            ORIGINAL CONTENT:
            {original_content}
            
            EXTRACTED METADATA:
            {metadata}
            
            Based on the information, generate the required structured response.
            """,
            input_variables=["context", "original_content", "metadata"]
        )

        full_prompt = prompt_template.invoke({
            "context": retrieved_context,
            "original_content": original_content,
            "metadata": json.dumps(metadata, indent=2)
        })

        try:
            # Enforce Pydantic Output using LangChain's built-in mechanism
            structured_llm = self.llm.with_structured_output(ContentSuggestions)
            
            response = structured_llm.invoke(full_prompt.text)
            
            # Convert the successful Pydantic object to a standard Python dictionary for display
            return response.dict()
        
        except ValidationError as e:
            # If the model fails validation, we catch the error and return failure details.
            print(f"\nFATAL VALIDATION ERROR: Model output did not match schema. {e}")
            return {"error": "LLM output validation failed.", "details": str(e)}
        except Exception as e:
             # Catch general API/Network errors
             print(f"\nAPI/Network Error: {e}")
             return {"error": "Failed to connect to OpenRouter or receive response.", "details": str(e)}