import os
import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import DATA_PATH, CHROMA_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, OLLAMA_BASE_URL

def clear_database():
    """Deletes the existing ChromaDB directory."""
    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
            print(f"Cleared existing database at {CHROMA_PATH}")
        except PermissionError:
            print(f"ERROR: Could not clear database at {CHROMA_PATH}.")
            print("The database is currently in use. Please STOP your Streamlit app or any other Python processes and try again.")
            exit(1)


def main():
    # Clear existing data for a fresh session
    clear_database()

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created {DATA_PATH} directory. Please add PDFs and run again.")
        return

    # 1. Load Documents
    print(f"Loading PDFs from {DATA_PATH}...")
    documents = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(DATA_PATH, file))
            documents.extend(loader.load())

    if not documents:
        print("No PDF files found in data directory.")
        return

    # Splitting the Docs
    print(f"Splitting {len(documents)} document pages...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Generated {len(chunks)} chunks.")

    # 3. Create Vector Database
    print(f"Creating vector database at {CHROMA_PATH} using {EMBEDDING_MODEL}...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
