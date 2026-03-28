import streamlit as st
from rag import RAGEngine
import os
import shutil
from config import DATA_PATH, CHROMA_PATH
import ingest

# 1. Page Config
st.set_page_config(page_title="Private Local RAG Agent", layout="wide")

# --- Session Initialization ---
# This block runs only ONCE when the user first opens the app in a new session/tab.
if "initialized" not in st.session_state:
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    if os.path.exists(DATA_PATH):
        for f in os.listdir(DATA_PATH):
            os.remove(os.path.join(DATA_PATH, f))
    st.session_state.initialized = True
    # Force a cache clear for the engine as well
    st.cache_resource.clear()
# ------------------------------

# 2. Sidebar Configuration
st.sidebar.title("RAG Settings")
mode = st.sidebar.radio("Reasoning Mode", ["Auto-Route", "Force Deep Reasoning"])

# Clear Database Functionality
if st.sidebar.button("🗑️ Clear Database/Session"):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    if os.path.exists(DATA_PATH):
        for f in os.listdir(DATA_PATH):
            os.remove(os.path.join(DATA_PATH, f))
    st.sidebar.success("Database and local files cleared!")
    st.rerun()

st.sidebar.divider()
st.sidebar.info("This system uses Llama 3.2 for routing and DeepSeek-R1 for complex reasoning. All data stays local.")

# 3. Initialize RAG Engine
@st.cache_resource
def get_engine():
    # Only return engine if DB exists
    if not os.path.exists(CHROMA_PATH):
        return None
    return RAGEngine()

engine = get_engine()

# 4. Main UI
st.title("🛡️ Private-Local Hybrid RAG Agent")
st.markdown("---")

# File Uploader
uploaded_files = st.file_uploader("Upload PDFs for your session", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    
    # Save uploaded files
    for uploaded_file in uploaded_files:
        with open(os.path.join(DATA_PATH, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    if st.button("🚀 Process & Ingest Documents"):
        with st.spinner("Ingesting documents into local vector store..."):
            ingest.main()
            st.cache_resource.clear() # Reset engine cache after ingestion
            st.success("Ingestion complete! You can now ask questions.")
            st.rerun()

# Data Status
if not os.path.exists(CHROMA_PATH):
    st.warning("⚠️ Database is empty. Please upload PDFs and click 'Process' above.")
else:
    st.success("✅ Knowledge Base Active")

# 5. Query Input
query = st.text_input("Ask a question about your documents:", placeholder="e.g., What are the key takeaways from the report?")

if query:
    if engine is None:
        st.error("Please ingest documents first.")
    else:
        with st.spinner("Processing..."):
            try:
                result = engine.query(query, mode=mode)
                
                # Display Answer
                st.subheader("Answer")
                st.write(result["answer"])
                
                # Routing Info
                st.info(f"Model Routing Strategy: **{result['routing'].upper()}**")
                
                # Retrieved Context
                with st.expander("🔍 View Retrieved Context"):
                    st.markdown(result["context"])
                    
            except Exception as e:
                st.error(f"An error occurred during query execution: {e}")

# 6. Footer
st.sidebar.divider()
st.sidebar.caption("Hardware: RTX 3050 (4GB VRAM) / 24GB RAM Optimized")
