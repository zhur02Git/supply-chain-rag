"""从ChromaDB检索最相关的chunk"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_PATH = "./chroma_db"
COLLECTION = "supply_chain"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)


def retrieve(query: str, k: int = None) -> list:
    k = k or int(os.getenv("TOP_K", 4))
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )
    results = store.similarity_search_with_score(query, k=k)
    output = []
    for doc, score in results:
        output.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "similarity": round(1 - score, 3),
        })
    return output