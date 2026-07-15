# 🔍 AI Search Engine + Agentic RAG

An advanced, hybrid AI Search Engine that dynamically routes user queries using an autonomous AI Agent. Built with **Streamlit**, **LangChain**, **Groq (Llama 3.3)**, and **Tavily Search**, it intelligently decides when to query private documents (RAG) and when to fetch live results from the web.

---

## 🚀 Key Features

* **🤖 Agentic Routing:** Uses an autonomous Llama 3.3 agent as the decision-making "brain" to determine the optimal workflow path for any prompt.
* **📄 Local PDF RAG:** Upload and instantly index PDF documents. Semantic search is executed locally using **FAISS** vector storage and **HuggingFace** sentence-transformer embeddings.
* **🌐 Real-Time Web Search:** Integrates **Tavily Search API** to fetch up-to-the-minute web information when local documents don't have the answer.
* **💬 State-Aware Chat UI:** Features a modern, styled conversational interface with responsive animations, side-by-side action triggers, and chat memory history.
* **🎨 Animated Glassmorphic UI:** Premium CSS enhancements including glowing card statistics, smooth entrance transitions, and custom hover effects on action buttons.

---

## 🛠️ Architecture Flow


The application executes a routing decision immediately upon receiving a query:
1. **Direct Answer:** Responds purely via LLM reasoning if no external data is needed.
2. **Local Document Search (RAG):** Performs similarity search over FAISS chunks using HuggingFace Embeddings if the prompt concerns the uploaded PDF.
3. **Web Search:** Queries Tavily API for current events, tech news, or general live facts.

---

## 📦 Tech Stack

* **Frontend Framework:** [Streamlit](https://streamlit.io/)
* **Agentic Orchestration:** [LangChain](https://www.langchain.com/)
* **Inference Compute Engine:** [Groq Cloud API](https://groq.com/) (Llama-3.3-70b-versatile / Llama-3.1-8b-instant)
* **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss)
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
* **External Search API:** [Tavily AI](https://tavily.com/)

---

## 🔧 Installation & Setup

### 1. Clone the Repository
```bash
git clone <your_repo_url>
cd SearchEngine


# 2. Set Up a Virtual Environment

# Create environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate

# 3.Install Dependencies
pip install -r requirements.txt

# 4.Environment Variables
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

🏃 How to Run the App
streamlit run app.py