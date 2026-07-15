import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# LangChain - Standard Imports
from langchain.agents import create_agent
from langchain_classic.tools.retriever import create_retriever_tool

# Groq
from langchain_groq import ChatGroq

# Tavily
from langchain_tavily import TavilySearch

# RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# --- 1. PRE-FLIGHT CHECK (Must happen before initializing LLMs) ---
required_keys = ["GROQ_API_KEY", "TAVILY_API_KEY"]
missing = [key for key in required_keys if not os.getenv(key)]

if missing:
    st.error(f"Missing environment variables:\n\n{', '.join(missing)}")
    st.stop()

# --- 2. STREAMLIT CONFIGURATION ---
st.set_page_config(
    page_title="AI Search Engine",
    page_icon="🔍",
    layout="wide"
)

# --- 3. CUSTOM CSS AND ANIMATIONS ---
st.markdown(
    """
    <style>
    /* Global Page Fade-in Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stApp {
        animation: fadeIn 0.8s ease-out;
    }

    /* Style the main title */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(45deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        animation: fadeIn 1s ease-out;
    }

    /* Beautiful Sidebar tweaks */
    section[data-testid="stSidebar"] {
        background-color: #0e1117;
        border-right: 1px solid #262730;
    }

    /* Styling Streamlit Buttons with transitions & hover effects */
    div.stButton > button {
        background: linear-gradient(135deg, #2a2a35 0%, #1e1e28 100%) !important;
        color: #ffffff !important;
        border: 1px solid #3e3e4f !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        width: 100% !important;
    }

    /* Hover State Animation */
    div.stButton > button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        border-color: #FF4B4B !important;
        box-shadow: 0 8px 15px rgba(255, 75, 75, 0.2) !important;
        color: #FF4B4B !important;
    }

    /* Active/Pressed State */
    div.stButton > button:active {
        transform: translateY(1px) scale(0.98) !important;
    }

    /* Distinct dangerous buttons style (Remove PDF / Clear) */
    div.st-key-remove-pdf button, div.st-key-clear-chat button {
        background: linear-gradient(135deg, #2b1c1c 0%, #1a1010 100%) !important;
        border: 1px solid #5a2c2c !important;
    }
    div.st-key-remove-pdf button:hover, div.st-key-clear-chat button:hover {
        border-color: #ff4b4b !important;
        color: #ff6b6b !important;
        box-shadow: 0 8px 15px rgba(255, 75, 75, 0.15) !important;
    }

    /* Custom File Uploader styling */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #3e3e4f;
        border-radius: 12px;
        padding: 10px;
        background-color: #161a24;
        transition: border-color 0.3s;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #FF4B4B;
    }
    
    /* Elegant Information Cards */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        animation: fadeIn 0.5s ease-out;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Render Custom Animated Title
st.markdown('<h1 class="main-title">🔍 AI Search Engine + RAG</h1>', unsafe_allow_html=True)
st.markdown('<p style="color: #8b949e; margin-bottom: 2rem;">Hybrid smart-routing AI agent using Llama 3.3 and Tavily.</p>', unsafe_allow_html=True)

st.sidebar.title("Upload your PDF files")

# --- 4. CACHED EMBEDDINGS MODEL ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embedding_model = load_embeddings()

# Initialize Chat Model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# Sidebar File Upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Knowledge Source (PDF)",
    type=["pdf"]
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- 5. FILE PROCESSING ---
if uploaded_file and st.session_state.vectorstore is None:
    with st.spinner("Analyzing and building local database..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            pdf_path = temp_file.name

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = splitter.split_documents(documents)

        st.session_state.vectorstore = FAISS.from_documents(
            docs,
            embedding_model
        )
        
        st.session_state.num_chunks = len(docs)
        st.session_state.file_name = uploaded_file.name

# Show Indexed Document Stats elegantly in the sidebar
if st.session_state.vectorstore is not None:
    st.sidebar.markdown(
        f"""
        <div class="metric-card">
            <h4 style="margin:0; color:#FF4B4B;">📄 Document Loaded</h4>
            <p style="margin:5px 0 0 0; font-size:14px; color:#c9d1d9;"><b>File:</b> {st.session_state.get('file_name')}</p>
            <p style="margin:2px 0 0 0; font-size:14px; color:#8b949e;"><b>Total Chunks:</b> {st.session_state.get('num_chunks')}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- 6. INITIALIZE TOOLS ---
tools = []

# Web Search Tool (Always available)
search_tool = TavilySearch(
    max_results=5,
    topic="general"
)
tools.append(search_tool)

# Add retriever tool only if PDF is indexed
if st.session_state.vectorstore is not None:
    retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )
    retriever_tool = create_retriever_tool(
        retriever,
        name="pdf_search",
        description="""
        Search inside uploaded PDFs. 
        Use this tool whenever the user asks questions about the uploaded document.
        """
    )
    tools.append(retriever_tool)

# --- 7. AI AGENT CONFIGURATION ---
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are an intelligent AI assistant.

You have access to two tools:

1. pdf_search
   - Use ONLY when the question is about the uploaded PDF.

2. tavily_search
   - Use for current events, internet facts, latest news, weather, technology, etc.

Rules:
- If the answer is in the uploaded PDF, use pdf_search.
- If the answer requires internet knowledge, use tavily_search.
- If no tool is required, answer directly.

Always give a clear, professional, and beautifully structured answer.
If a tool provides sources, list them cleanly at the end of your response.
"""
)

# --- 8. SIDEBAR ACTIONS (Styled using custom keys mapped to CSS rules) ---
st.sidebar.markdown("<br>", unsafe_allow_html=True)

if st.sidebar.button("🧹 Clear Chat History", key="clear-chat"):
    st.session_state.messages = []
    st.rerun()

if st.sidebar.button("🗑️ Remove Document", key="remove-pdf"):
    if "vectorstore" in st.session_state:
        st.session_state.vectorstore = None
    st.rerun()

# --- 9. CHAT INTERFACE ---
# Render past messages with smooth transition
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
user_question = st.chat_input("Ask me anything about the document or the web...")

if user_question:
    # Append & display User message
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate Assistant response with spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking and routing..."):
            response = agent.invoke(
                {
                    "messages": st.session_state.messages
                }
            )
            answer = response["messages"][-1].content
            st.markdown(answer)
            
            # Save response to memory
            st.session_state.messages.append({"role": "assistant", "content": answer})