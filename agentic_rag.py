from typing import List
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# =========================================================
# Configuration
# =========================================================
from dotenv import load_dotenv
load_dotenv()

VECTOR_INDEX_PATH = "vector_index/legal_compliance"
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o"


# =========================================================
# Data Models (Structured, Enterprise-Grade)
# =========================================================

class RetrievalTask(BaseModel):
    query: str
    reason: str


class RetrievalPlan(BaseModel):
    tasks: List[RetrievalTask]


# =========================================================
# Load Vector Store (Offline Artifact)
# =========================================================

def load_vectorstore():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    return FAISS.load_local(
        VECTOR_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


# =========================================================
# Planner Agent
# =========================================================

PLANNER_SYSTEM_PROMPT = """
You are a senior enterprise AI planner.

Your task:
- Decompose the user question into focused retrieval tasks
- Each task MUST contain:
  - query: a short search query
  - reason: why this retrieval is needed
- Do NOT answer the question
- Do NOT invent fields

You MUST return JSON in the following exact format:

{{
  "tasks": [
    {{
      "query": "string",
      "reason": "string"
    }}
  ]
}}

Return ONLY valid JSON.
"""



def build_planner():
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", PLANNER_SYSTEM_PROMPT),
        ("human", "{question}")
    ])

    return (
        prompt
        | llm
        | PydanticOutputParser(pydantic_object=RetrievalPlan)
    )


# =========================================================
# Retriever Agent
# =========================================================

def run_retriever(vectorstore, plan: RetrievalPlan, k: int = 5):
    all_docs = []

    for task in plan.tasks:
        docs = vectorstore.similarity_search(
            task.query,
            k=k
        )
        all_docs.extend(docs)

    return all_docs


# =========================================================
# Answerer Agent
# =========================================================

ANSWER_SYSTEM_PROMPT = """
You are an enterprise AI assistant.

Rules:
- Answer ONLY using the provided context
- Cite evidence using document metadata
- If the answer is not fully available, say so clearly
- Do NOT hallucinate or assume facts
"""

def build_answerer():
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0
    )


def generate_answer(llm, question: str, documents):
    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}] {doc.page_content}"
        for doc in documents
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_SYSTEM_PROMPT),
        ("human", "Question:\n{question}\n\nContext:\n{context}")
    ])

    formatted_prompt = prompt.format_prompt(
        question=question,
        context=context
    )

    return llm.invoke(formatted_prompt)



# =========================================================
# End-to-End Agentic RAG Pipeline
# =========================================================

def agentic_rag(question: str):
    print(" Loading vector store...")
    vectorstore = load_vectorstore()

    print(" Planning retrieval strategy...")
    planner = build_planner()
    plan = planner.invoke({"question": question})

    print(" Retrieval Plan:")
    for task in plan.tasks:
        print(f" - {task.query} | Reason: {task.reason}")

    print(" Retrieving evidence...")
    documents = run_retriever(vectorstore, plan)

    print(f" Retrieved {len(documents)} document chunks")

    print(" Generating grounded answer...")
    answer_llm = build_answerer()
    answer = generate_answer(answer_llm, question, documents)

    return answer


# =========================================================
# Main (Demo Entry Point)
# =========================================================

if __name__ == "__main__":
    query = (
        "What are the data retention requirements for customer PII, "
        "and what steps are required after a security incident?"
    )

    response = agentic_rag(query)

    print("\n================ FINAL ANSWER ================\n")
    print(response.content)
