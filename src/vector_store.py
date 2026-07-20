import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from log_parser import parse_log_file

# Load environment variables from .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def get_embedding_function():
    """Returns Gemini embedding model wrapper."""
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )

def build_vector_store(log_file_path: str, persist_directory: str = "./data/chroma_db"):
    """
    Parses a log file, creates Document objects, and indexes them into ChromaDB.
    """
    parsed_logs = parse_log_file(log_file_path)
    
    documents = []
    for log in parsed_logs:
        doc = Document(
            page_content=log["content"],
            metadata={
                "line_number": log["line_number"],
                "level": log["level"],
                "source": log["source"]
            }
        )
        documents.append(doc)
        
    embeddings = get_embedding_function()
    
    # Create / update vector database
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"✅ Indexed {len(documents)} log entries into ChromaDB at '{persist_directory}'.")
    return vector_db

def query_relevant_logs(query: str, persist_directory: str = "./data/chroma_db", k: int = 3):
    """
    Searches ChromaDB for top-k log entries semantically relevant to the user query.
    """
    embeddings = get_embedding_function()
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    results = vector_db.similarity_search(query, k=k)
    return results

if __name__ == "__main__":
    # Test vector store indexing and query
    test_log = "logs/sample_auth.log"
    print("Indexing sample log...")
    db = build_vector_store(test_log)
    
    test_query = "Were there any unauthorized access or failed login attempts?"
    print(f"\nTesting Query: '{test_query}'")
    matched_docs = query_relevant_logs(test_query, k=2)
    
    for i, doc in enumerate(matched_docs, 1):
        print(f"\nMatch {i}:")
        print(f" Content: {doc.page_content}")
        print(f" Metadata: {doc.metadata}")
