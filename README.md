# 🔩 Supply Chain RAG System

A retrieval-augmented generation (RAG) system for manufacturing procurement teams to query supplier specifications, product datasheets, and compliance documents.

Built as part of my FDE interview preparation — demonstrating end-to-end RAG pipeline design with real supply chain context (TE Connectivity product knowledge).

## Demo

Ask questions like:
- "What products does TE Connectivity make?"
- "What industries does TE Connectivity serve?"
- "What are the connector specifications?"

## Architecture

Document (PDF/TXT)
↓ PyPDFLoader
↓ RecursiveCharacterTextSplitter (chunk_size=512, overlap=100)
↓ HuggingFace all-MiniLM-L6-v2 (384-dim embeddings, local)
↓ ChromaDB (persistent vector store)
↓ Similarity Search (Top-K retrieval)
↓ DeepSeek LLM (citation-aware generation)
→ Answer + Source attribution

## Tech Stack

| Component | Choice | Reason |
|---|---|---|
| Framework | LangChain | Industry standard for RAG pipelines |
| Embeddings | all-MiniLM-L6-v2 | Local, no API key, privacy-friendly |
| Vector DB | ChromaDB | Persistent local store, zero setup |
| LLM | DeepSeek | OpenAI-compatible, cost-effective |
| UI | Streamlit | Rapid prototyping for demo |

## Key Design Decisions

**Anti-hallucination prompt** — LLM is instructed to answer only from retrieved context and say "not found" rather than fabricate facts. Critical for enterprise document QA.

**Local embeddings** — Using `sentence-transformers/all-MiniLM-L6-v2` instead of OpenAI embeddings means data never leaves the machine. Important for manufacturing clients with IP sensitivity.

**Source attribution** — Every answer includes `[filename, page number]` citations so procurement teams can verify against original documents.

## Local Setup

```bash
git clone https://github.com/zhur02Git/supply-chain-rag.git
cd supply-chain-rag

python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
```

Create `.env`:
```env
OPENAI_API_KEY=your-deepseek-key
OPENAI_API_BASE=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
CHUNK_SIZE=512
CHUNK_OVERLAP=100
TOP_K=4
```

```bash
# Ingest documents
python ingest.py data/your_document.pdf

# Launch UI
streamlit run app.py
```

## What I Learned

- chunk_size directly impacts retrieval precision — smaller chunks improve accuracy for specific parameter queries, larger chunks preserve context for summary questions
- Normalizing embeddings (`normalize_embeddings=True`) is required for ChromaDB cosine similarity to work correctly
- Separating the ingest pipeline from the query pipeline allows independent scaling in production