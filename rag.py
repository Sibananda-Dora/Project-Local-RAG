import re
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from config import CHROMA_PATH, EMBEDDING_MODEL, ROUTER_MODEL, REASONING_MODEL, OLLAMA_BASE_URL


class RAGEngine:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        self.vector_db = Chroma(
            persist_directory=CHROMA_PATH, embedding_function=self.embeddings)
        self.router_llm = OllamaLLM(
            model=ROUTER_MODEL, base_url=OLLAMA_BASE_URL)
        self.reasoning_llm = OllamaLLM(
            model=REASONING_MODEL, base_url=OLLAMA_BASE_URL)
        self.fast_llm = OllamaLLM(model=ROUTER_MODEL, base_url=OLLAMA_BASE_URL)

    def route_query(self, query: str) -> str:
        """
        Determines if a query needs reasoning or simple retrieval.
        Returns: 'reasoning' or 'fast'
        """
        prompt = ChatPromptTemplate.from_template(
            "Analyze the following user query. If it requires complex analysis, comparisons, "
            "definitions, scholarly names, or lists of characteristics, respond with ONLY the "
            "word 'reasoning'. Otherwise, respond with 'fast'.\n\nQuery: {query}\nResponse:"
        )
        chain = prompt | self.router_llm
        response = chain.invoke({"query": query}).strip().lower()

        # Simple heuristic if LLM is chatty
        if 'reasoning' in response:
            return 'reasoning'
        return 'fast'

    def get_context(self, query: str, k: int = 7):
        docs = self.vector_db.similarity_search(query, k=k)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        return context, docs

    def clean_think_tags(self, text: str) -> str:
        """Strips <think>...</think> tags and their content."""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def query(self, query: str, mode: str = "Smart Routing"): # Update default here
    # 1. Determine Mode
        if mode == "Smart Routing": # Update check here
            routing_decision = self.route_query(query)
        else:
            routing_decision = "reasoning"

        # 2. Retrieve Context
        context, docs = self.get_context(query)

        # 3. Prompting
        prompt_template = ChatPromptTemplate.from_template(
            """You are a Senior Researcher. Provide a rigorous answer based ONLY on the provided context.

    CRITICAL REQUIREMENTS:
    1. FORMULAS: If the context contains a mathematical equation (e.g., Attitudes = Beliefs x Values), you MUST include it using LaTeX.
    2. SCHOLARS: Mention any specific researchers cited (e.g., Allport, Krech, or Festinger).
    3. LISTS: Use numbered or bulleted lists for characteristics or components.
    4. STRUCTURE: Use bold headings for different sections of your answer.

    Context:
    {context}

    Question: {query}
    Answer:"""
        )

        if routing_decision == "reasoning":
            llm = self.reasoning_llm
            is_reasoning_model = True
        else:
            llm = self.fast_llm
            is_reasoning_model = False

        # 4. Generate Answer
        chain = prompt_template | llm
        raw_answer = chain.invoke({"context": context, "query": query})

        # 5. Post-process (if reasoning model)
        if is_reasoning_model:
            final_answer = self.clean_think_tags(raw_answer)
        else:
            final_answer = raw_answer

        return {
            "answer": final_answer,
            "routing": routing_decision,
            "context": context,
            "docs": docs
        }


if __name__ == "__main__":
    # Test
    # engine = RAGEngine()
    # print(engine.query("What is the main topic of these documents?"))
    pass
