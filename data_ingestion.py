from datetime import datetime
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# -----------------------------
# Configuration
# -----------------------------
from dotenv import load_dotenv
load_dotenv()

PDF_PATH = Path("Acme_Legal_and_Compliance_Policy_2024.pdf")
VECTOR_INDEX_PATH = Path("vector_index/legal_compliance")

EMBEDDING_MODEL = "text-embedding-3-large"

# -----------------------------
# Step 1: Load PDF
# -----------------------------

def load_pdf(pdf_path: Path):
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    return documents


# -----------------------------
# Step 2: Chunking Strategy
# -----------------------------

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)


# -----------------------------
# Step 3: Metadata Enrichment
# -----------------------------

def enrich_metadata(chunks):
    for chunk in chunks:
        chunk.metadata.update({
            "company": "Acme Financial Technologies Inc.",
            "doc_type": "legal_compliance",
            "document_name": "Legal & Compliance Policy Manual",
            "region": "global",
            "source": "Acme_Legal_and_Compliance_Policy_2024.pdf",
            "ingested_at": datetime.utcnow().isoformat()
        })
    return chunks


# -----------------------------
# Step 4: Vector Index Creation
# -----------------------------

def build_vector_index(chunks):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    VECTOR_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTOR_INDEX_PATH))

    return vectorstore


# -----------------------------
# Main Ingestion Pipeline
# -----------------------------

def run_ingestion():
    print(" ....Loading PDF...")
    docs = load_pdf(PDF_PATH)

    print(f" Loaded {len(docs)} pages")

    print("️...Chunking documents...")
    chunks = chunk_documents(docs)

    print(f" Created {len(chunks)} chunks")

    print(".....Enriching metadata...")
    chunks = enrich_metadata(chunks)

    print(" Building vector index...")
    build_vector_index(chunks)

    print(" Ingestion completed successfully")
    print(f" Vector index stored at: {VECTOR_INDEX_PATH}")


if __name__ == "__main__":
    run_ingestion()
