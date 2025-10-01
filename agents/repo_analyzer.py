import os
import shutil
import time 
import tempfile 
from git import Repo, GitCommandError 
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter 


class RepoAnalyzerAgent:
    # We no longer accept working_dir in __init__ as it is determined by tempfile
    def __init__(self, repo_url: str):
        self.repo_url = repo_url
        self.working_dir = "" # Temporary working directory path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

    def _clone_repo(self):
        """Clones the repository into a temporary working directory."""
        # Create a unique temporary directory (usually on the C: drive in a non-restricted area)
        self.working_dir = tempfile.mkdtemp(prefix="repo_clone_")
        
        print(f"DEBUG: Cloning to temporary directory: {self.working_dir}")
        print(f"DEBUG: Attempting to clone public repository: {self.repo_url}")
        
        try:
            # The Git clone operation
            Repo.clone_from(self.repo_url, self.working_dir)
            print("DEBUG: Cloning successful.")
        except GitCommandError as e:
            raise RuntimeError(f"Git Clone Failed. Check URL or Git PATH: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Cloning failed due to unknown error: {e}") from e


    def _load_and_split_files(self):
        """Loads and splits content from README and other key files using the temporary path."""
        docs = []
        file_paths = ["README.md", "main.py", "requirements.txt"]
        
        for file in file_paths:
            full_path = os.path.join(self.working_dir, file)
            if os.path.exists(full_path):
                print(f"DEBUG: Loading and splitting {file}...")
                try:
                    loader = TextLoader(full_path, encoding='utf-8')
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"Could not load {file}: {e}")
        
        chunks = self.text_splitter.split_documents(docs)
        print(f"DEBUG: Total content split into {len(chunks)} chunks.")
        return chunks

    def process_repo(self):
        """Main method to clone, read, and process the repository, ensuring cleanup."""
        temp_dir = ""
        try:
            self._clone_repo()
            temp_dir = self.working_dir # Store the temp path for cleanup
            time.sleep(1) 
            chunks = self._load_and_split_files()
            
            # --- Cleanup the temporary directory ---
            if os.path.exists(temp_dir):
                # Use shutil.rmtree to remove the directory and its contents recursively
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"DEBUG: Cleaned up temporary directory {temp_dir}")
            
            return chunks
        except Exception as e:
            print(f"\nFATAL REPO ANALYZER ERROR: {e}\n")
            # Try to clean up on failure before re-raising
            if os.path.exists(temp_dir):
                 shutil.rmtree(temp_dir, ignore_errors=True)
            raise e 


    def create_retriever(self, chunks):
        """Creates and returns a FAISS-based retriever from the document chunks."""
        # This part of the code initializes the embedding model and vector store.
        try:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
            
            vectorstore = FAISS.from_documents(chunks, embeddings_model)
            retriever = vectorstore.as_retriever()
            print("DEBUG: Retriever created successfully.")
            return retriever
        except Exception as e:
             # This will catch errors during the HuggingFace model download/load
             print(f"\nFATAL EMBEDDING/RETRIEVER ERROR: {e}\n")
             raise RuntimeError("Failed to initialize embeddings model or retriever. Check network access for model download.") from e