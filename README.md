# Private-Local RAG Agent

A 100% local Retrieval-Augmented Generation (RAG) system. This agent is optimized for privacy , specifically tuned for hardware with limited VRAM (RTX 3050 4GB) and high system RAM (24GB).
(IF YOU HAVE LESS RAM THEN GO WITH OTHER COMPATIBLE MODELS AND JUST UPDATE THE 'config.py' FILE. )

## 🚀 Key Features

- **Privacy First:** 100% local processing. No data ever leaves your machine.
- **Model Routing:** 
  - **Llama 3.2 (3B):** Used for fast query routing and simple retrieval tasks.
  - **DeepSeek-R1 (7B):** Triggered automatically for "Why" questions, complex comparisons, or multi-step reasoning.
- **Ephemeral Sessions:** Automatically wipes the vector database and uploaded files at the start of every new browser session to ensure data hygiene.
- **Streamlit UI:** Easy-to-use Streamlit dashboard for multi-PDF uploads and real-time chatting.

## 🛠️ Tech Stack

- **Orchestration:** LangChain
- **Models (via Ollama):** 
  - Routing/Fast: `llama3.2:3b`
  - Reasoning: `deepseek-r1:7b`
  - Embeddings: `nomic-embed-text:latest`
  - (CAN CHANGE MODELS AS YOU WISH BY UPDATING THE 'config' FILE)
- **Vector Database:** ChromaDB (via `langchain-chroma`)
- **Frontend:** Streamlit

## 📦 Installation & Setup

### 1. Prerequisites
Ensure you have [Ollama](https://ollama.com/) installed and running.

### 2. Pull Required Models
Open your terminal and run:
```bash
ollama pull llama3.2:3b
ollama pull deepseek-r1:7b
ollama pull nomic-embed-text:latest
```

### 3. Install Python Dependencies 
- Install these inside a virtual environment to avoid global download.
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

## 📖 How to Use

1. **Upload:** Drag and drop one or multiple PDF files into the sidebar/main area.
2. **Process:** Click **"🚀 Process & Ingest Documents"**. This will chunk the text, generate embeddings locally, and build your search index.
3. **Chat:** Ask questions! 
   - **Auto-Route:** Let the system decide which model is best for your question.
   - **Force Deep Reasoning:** Manually trigger the DeepSeek-R1 engine for every query.
4. **Reset:** Use the **"🗑️ Clear Database/Session"** button or simply restart your browser session to wipe all local data.

## 📁 Project Structure
- `app.py`: Streamlit frontend and session management.
- `rag.py`: The core LangChain logic and hybrid routing engine.
- `ingest.py`: Document processing and vector store pipeline.
- `config.py`: Centralized model names and hardware settings.
- `data/`: Temporary storage for uploaded PDFs.
- `chroma_db/`: Local vector database storage.

## Contributions and PRs

Feel free to locate issues and send pull requests to improve this project.
