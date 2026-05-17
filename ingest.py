"""摄入文档：PDF → 分块 → 向量化 → 存入本地ChromaDB"""

import os, sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_PATH = "./chroma_db"
COLLECTION  = "supply_chain"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=int(os.getenv("CHUNK_SIZE", 512)),
    chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 100)),
    add_start_index=True,
)

def ingest(file_path: str):
    path = Path(file_path)
    print(f"📄 加载文件: {path.name}")

    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")

    docs    = loader.load()
    chunks  = splitter.split_documents(docs)
    print(f"✂️  切成 {len(chunks)} 个chunk")

    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )
    store.add_documents(chunks)
    print(f"✅ 完成！共存入 {len(chunks)} 个chunk")

if __name__ == "__main__":
    ingest(sys.argv[1])