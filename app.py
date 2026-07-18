import os
import sqlite3
import streamlit as st
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilyAnswer

# ==========================================
# 1. INITIALIZATION & API KEYS
# ==========================================
st.set_page_config(page_title="Modern Open-Source AI Agent", layout="wide")
st.title("🤖 Local Multi-Tool AI Agent (Groq + SQLite)")

from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY is not configured.")
    st.stop()

if not tavily_api_key:
    st.error("TAVILY_API_KEY is not configured.")
    st.stop()

DB_PATH = "company.db"
VECTOR_DB_DIR = "chroma_db"

# ==========================================
# 2. DEFINE THE AGENT TOOLS
# ==========================================

@tool
def query_uploaded_docs(query: str) -> str:
    """Useful when you need to answer questions based on the uploaded PDF documents. 
    Input must be a simple text query string."""
    if not os.path.exists(VECTOR_DB_DIR) or not os.listdir(VECTOR_DB_DIR):
        return "No PDFs have been uploaded or processed yet. Tell the user to upload a PDF in the sidebar."
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    retrieved_docs = retriever.invoke(query)
    if not retrieved_docs:
        return "No relevant information found in the uploaded PDFs."
        
    return "\n\n".join([f"[Source: {doc.metadata.get('source')}]: {doc.page_content}" for doc in retrieved_docs])

@tool
def internet_search(query: str) -> str:
    """Useful when you need to find up-to-date real-time information on the internet, current events, dates, sports scores, or news. 
    Input must be a single search string."""
    try:
        search = TavilyAnswer()
        return search.run(query)
    except Exception as e:
        return f"Internet search failed: {str(e)}"

@tool
def query_company_database(sql_query: str) -> str:
    """Useful for querying the internal company database. Input MUST be a valid SQLite SELECT statement.
    Do not attempt to modify data (NO INSERT, UPDATE, or DELETE)."""
    if not os.path.exists(DB_PATH):
        return f"Error: Database file '{DB_PATH}' not found in the local directory."
        
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        
        columns = [description[0] for description in cursor.description]
        conn.close()
        
        if not rows:
            return "Query executed successfully, but returned 0 results."
        
        result = [dict(zip(columns, row)) for row in rows]
        return str(result)[:4000]
    except Exception as e:
        return f"Database Error: {str(e)}"

@tool
def get_database_schema(dummy_arg: str = "") -> str:
    """Useful to discover available tables and columns in the company database before writing an SQL query."""
    if not os.path.exists(DB_PATH):
        return f"Error: Database file '{DB_PATH}' not found."
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        schema_info = []
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            col_details = [f"{col[1]} ({col[2]})" for col in columns]
            schema_info.append(f"Table Name: {table_name}\nColumns: {', '.join(col_details)}")
        
        conn.close()
        return "\n\n".join(schema_info)
    except Exception as e:
        return f"Failed to retrieve database schema: {str(e)}"

tools = [query_uploaded_docs, internet_search, query_company_database, get_database_schema]

# ==========================================
# 3. STREAMLIT SIDEBAR - PDF PROCESSING
# ==========================================
st.sidebar.header("📁 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or DOCX files",
    type=["pdf", "docx"],
    accept_multiple_files=True
)
if st.sidebar.button("Process Documents") and uploaded_files:
    with st.spinner("Processing and indexing PDFs locally..."):
        all_docs = []
        os.makedirs("temp_pdfs", exist_ok=True)
        
        for uploaded_file in uploaded_files:

            temp_path = os.path.join("temp_pdfs", uploaded_file.name)

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # PDF
            if uploaded_file.name.lower().endswith(".pdf"):
                loader = PyPDFLoader(temp_path)

            # DOCX
            elif uploaded_file.name.lower().endswith(".docx"):
                loader = Docx2txtLoader(temp_path)

            else:
                continue

            all_docs.extend(loader.load())

            os.remove(temp_path)
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=VECTOR_DB_DIR)
        st.sidebar.success(
            f"Indexed {len(uploaded_files)} document(s) successfully!"
        )
# ==========================================
# 4. AGENT & CHAT INTERFACE
# ==========================================
# NOTE: Ensure the model string exactly matches your Groq-supported model identifier
llm = ChatGroq(
    model="openai/gpt-oss-120b", 
    temperature=0,
    groq_api_key=groq_api_key
)

# Heavily modified prompt to eliminate conversational pushback regarding tool usage
system_prompt = """You are a direct, hyper-efficient AI Operations Assistant. The current year is 2026.

You are an AI assistant.
DO NOT ask the user for permission to use your tools. 
DO NOT explain what you cannot do or offer to look things up. 
If you lack factual knowledge, real-time context, or structural definitions to answer the request, you must IMMEDIATELY execute the appropriate tool without conversational preamble.
Answer only using information from the selected tool.
Do not make predictions or assumptions.
If the information is unavailable or the event has not occurred yet, say that you cannot determine the answer.
Never present guesses as facts.
You are a news assistant.
Only summarize information explicitly present in the retrieved search results.
Never invent headlines, dates, companies, or product names.
If the retrieved sources do not support a claim, state that the information could not be verified.
For every news item, mention the source.

Tool Rules:
1. 'internet_search': Use immediately for any query regarding today's date, real-time events, post-2023 information, or data you do not possess.
2. 'query_uploaded_pdfs': Use to check documentation uploaded by the user.
3. 'get_database_schema': Run this first before creating any SQL string.
4. 'query_company_database': Run standard SQLite SELECT queries."""

agent_executor = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask about company.db data, web information, or uploaded PDFs:"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("Executing modern tool loop..."):
            try:
                # Format history strictly as LangChain Message Objects
                formatted_messages = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        formatted_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        formatted_messages.append(AIMessage(content=msg["content"]))
                
                formatted_messages.append(HumanMessage(content=user_query))
                
                response = agent_executor.invoke({
                    "messages": formatted_messages
                })
                
                output_text = response["messages"][-1].content
                response_placeholder.markdown(output_text)
                
                st.session_state.messages.append({"role": "assistant", "content": output_text})
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")