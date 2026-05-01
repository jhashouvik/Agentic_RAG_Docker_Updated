# 🧠 PDF Q&A v2 — LangChain + LangGraph + FAISS + SQLite

A modern, fully Dockerized multi-agent PDF Q&A system using the 2024/2025 LangChain ecosystem.

---

## What's Modern About This Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **Agent orchestration** | LangGraph `StateGraph` | Microsoft-aligned pattern — same topology as AutoGen/Semantic Kernel agent graphs. Stateful node → node flow with typed shared state. |
| **PDF loading** | LangChain `PyPDFLoader` | Standard LangChain document loader with page metadata |
| **Chunking** | 4 LangChain strategies | RecursiveCharacterTextSplitter, TokenTextSplitter, CharacterTextSplitter, SemanticChunker |
| **Embeddings** | `langchain-openai` `OpenAIEmbeddings` | Wraps `text-embedding-3-small` / `3-large` / `ada-002` — switchable |
| **Vector store** | LangChain `FAISS` wrapper | Persistent (save/load) + cosine similarity with relevance scores |
| **LLM** | `langchain-openai` `ChatOpenAI` | GPT-4o or GPT-4o-mini via LangChain LCEL chains |
| **Memory** | SQLite (free, zero server) + `ConversationBufferWindowMemory` | SQLite scales to millions of rows, window memory gives LLM conversation context |
| **Self-reflection** | LangGraph `grade` node | LLM grades its own retrieved chunks for relevance before generating answer |
| **Prompts** | `ChatPromptTemplate` LCEL | Composable, typed, testable prompt templates |
| **UI** | Streamlit | Fast Python UI with file upload, comparison mode, live agent step display |

---

## LangGraph Agent Flow

```
User Question
      │
      ▼
┌─────────────┐
│  retrieve   │  FAISS cosine similarity → top-K chunks
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   grade     │  GPT-4o-mini judges each chunk: RELEVANT | IRRELEVANT
│             │  (self-reflection — filters noise before generation)
└──────┬──────┘
       │ filtered chunks
       ▼
┌─────────────┐
│  generate   │  GPT-4o RAG answer + conversation memory window
└──────┬──────┘
       │
      END → answer + metadata
```

---

## 4 Chunking Strategies — Comparison Guide

### 1. `recursive` (default — best all-round)
Uses `RecursiveCharacterTextSplitter`. Tries to split on `\n\n` → `\n` → `. ` → ` ` → `""`.
Produces natural paragraph-sized chunks that respect sentence boundaries.
**Use when:** You don't know the doc structure. Works well for 90% of PDFs.

### 2. `token`
Uses `TokenTextSplitter` with tiktoken `cl100k_base` (GPT-4o tokenizer).
Splits on exact token boundaries so you never exceed the LLM context window per chunk.
**Use when:** Docs with very dense technical content, code, or equations.

### 3. `character`
Uses `CharacterTextSplitter` splitting only on `\n\n`.
Very fast, minimal API calls. Produces uneven chunk sizes but respects section breaks.
**Use when:** Well-structured reports, academic papers, legal documents with clear headings.

### 4. `semantic`
Uses `SemanticChunker` (langchain-experimental). Calls the embedding API during chunking to detect meaning shifts and split where topics change.
Slowest (one embedding call per sentence split decision) but most coherent chunks.
**Use when:** Mixed-topic documents, interview transcripts, meeting notes.

---

## Memory Architecture

```
Every Q&A turn
      │
      ├── SQLite (memory.db)         ← permanent, scales to millions of rows
      │     tables: sessions, memory, comparisons
      │
      └── ConversationBufferWindowMemory  ← last 5 turns injected into LLM prompt
            (sliding window — LLM always gets recent context)
```

The SQLite DB is free, runs locally inside Docker, and persists across container restarts via a volume mount. No Redis, no external server needed.

---

## Comparison Mode

Enable **⚖️ Comparison Mode** in the sidebar to run any question through two chunking strategies simultaneously and see side-by-side answers. Results are saved to SQLite for later review.

---

## Quick Start

```bash
# 1. Unzip / clone
cd pdf-qa-v2

# 2. Configure
cp .env.example .env
# → Edit .env, add: OPENAI_API_KEY=sk-...

# 3. Run with Docker
docker compose up --build
# → Open http://localhost:8501

# 4. Or run locally (no Docker)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd streamlit_app
DATA_DIR=../data streamlit run app.py
```

---

## Project Structure

```
pdf-qa-v2/
├── agents/
│   ├── agent_ingestor.py      LangChain PyPDFLoader
│   ├── agent_chunker.py       4 LangChain chunking strategies
│   ├── agent_vectorstore.py   LangChain FAISS + OpenAIEmbeddings
│   ├── agent_memory.py        SQLite + ConversationBufferWindowMemory
│   └── agent_orchestrator.py  LangGraph StateGraph (retrieve→grade→generate)
├── streamlit_app/
│   ├── app.py                 Full UI — upload, Q&A, comparison mode
│   └── .streamlit/
│       └── config.toml        Dark theme
├── data/                      Persisted data (gitignored)
│   ├── faiss_store/           FAISS index per session
│   └── memory.db              SQLite conversation memory
├── tests/
│   └── test_all.py            Unit tests (no API calls needed)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required** |
| `EMBED_MODEL` | `text-embedding-3-small` | `text-embedding-3-small`, `text-embedding-3-large`, or `text-embedding-ada-002` |
| `FAISS_STORE_DIR` | `/data/faiss_store` | Where FAISS indexes are saved |
| `DATA_DIR` | `/data` | Root for SQLite DB and uploads |

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v --tb=short
```

Tests cover all agents without requiring an OpenAI API key (LLM calls are mocked).

---

## Cost Estimates

| Operation | Model | Approx cost |
|-----------|-------|-------------|
| Embed a 30-page PDF (recursive) | text-embedding-3-small | ~$0.0005 |
| Embed a 30-page PDF (semantic chunking) | text-embedding-3-small | ~$0.001 |
| Q&A turn with grading | gpt-4o-mini (grade) + gpt-4o (answer) | ~$0.02 |
| Q&A turn, all gpt-4o-mini | gpt-4o-mini | ~$0.002 |

---

## Next: Azure Deployment (Stage 2)

This is Stage 1 — local prototype. Stage 2 adds:
- Push image to **Azure Container Registry (ACR)**
- Deploy to **Azure Web App for Containers**
- **Azure DevOps CI/CD pipeline** (`azure-pipelines.yml`)
- **Azure Key Vault** for secrets
- **Azure Blob Storage** for PDF + FAISS persistence

---

## Troubleshooting

**`OPENAI_API_KEY not set`**
→ Check `.env` is in the same folder as `docker-compose.yml`. Run `docker compose config` to verify env vars are resolved.

**Port 8501 in use**
→ Change left side of ports mapping: `"8502:8501"`

**Semantic chunking is slow**
→ Expected — it calls the embedding API once per sentence to compute breakpoints. Use `recursive` for speed.

**FAISS index not persisting**
→ Check that `./data` volume is mounted correctly. Run `docker compose down && docker compose up` (not `docker compose restart`).

**`No text extracted from PDF`**
→ The PDF is image/scan-only. Use a text-based PDF or add `pytesseract` for OCR support.
