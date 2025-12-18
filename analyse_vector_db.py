"""
vector_index/legal_compliance/
├── index.faiss   ← vectors only (binary)
└── index.pkl     ← text + metadata (pickle)

"""
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vectorstore = FAISS.load_local(
    "vector_index/legal_compliance",
    embeddings,
    allow_dangerous_deserialization=True,

)

print("Number of vectors:", vectorstore.index.ntotal)
docstore = vectorstore.docstore._dict

print(f"Number of stored documents: {len(docstore)}")
first_key = next(iter(docstore))
doc = docstore[first_key]

print("ID:", first_key)
print("Metadata:", doc.metadata)
print("Content:\n", doc.page_content[:500])

for i, (doc_id, doc) in enumerate(docstore.items()):
    print("=" * 80)
    print(f"Chunk #{i}")
    print("ID:", doc_id)
    print("Metadata:", doc.metadata)
    print("Content Preview:", doc.page_content[:300])

    if i >= 3:
        break

import faiss

index = faiss.read_index("vector_index/legal_compliance/index.faiss")
print(index)
print("Number of vectors:", index.ntotal)



