import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vector_store import query_relevant_logs

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SYSTEM_PROMPT = """You are an expert Security Operations Center (SOC) analyst and System Administrator.
Analyze the provided system logs to accurately answer the user's inquiry.

Guidelines:
- Base your analysis strictly on the provided log context.
- Highlight key details like IP addresses, usernames, timestamps, or severity levels if available.
- If the logs don't contain enough information to answer, state that clearly instead of guessing.

Logs Context:
{context}
"""

def format_docs(docs):
    """Formats retrieved Document objects into a readable text block."""
    formatted = []
    for doc in docs:
        meta = doc.metadata
        line = f"[Line {meta.get('line_number', 'N/A')} | {meta.get('level', 'INFO')}] {doc.page_content}"
        formatted.append(line)
    return "\n".join(formatted)

def analyze_logs(user_query: str, k: int = 5):
    """
    Retrieves context from ChromaDB and generates an analysis using Gemini.
    """
    # 1. Retrieve matching logs from vector database
    retrieved_docs = query_relevant_logs(user_query, k=k)
    
    if not retrieved_docs:
        return "No relevant log entries found matching your query."
    
    formatted_context = format_docs(retrieved_docs)
    
    # 2. Initialize Gemini Model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
        timeout=30,  # Fails fast after 30 seconds instead of hanging
        max_retries=2
    )
    
    # 3. Build Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])
    
    # 4. Construct LangChain Chain
    chain = prompt | llm | StrOutputParser()
    
    # 5. Run Chain
    response = chain.invoke({
        "context": formatted_context,
        "question": user_query
    })
    
    return response, retrieved_docs

if __name__ == "__main__":
    query = "Summarize any suspicious or failed login attempts found in the logs."
    print(f"🔎 Query: '{query}'\n")
    print("Generating analysis...\n" + "-"*50)
    
    analysis, sources = analyze_logs(query, k=4)
    print(f"\n📋 **Gemini Analysis:**\n{analysis}\n")
    
    print("-" * 50)
    print("📌 **Sources Used:**")
    for doc in sources:
        print(f" - Line {doc.metadata.get('line_number')}: {doc.page_content}")
