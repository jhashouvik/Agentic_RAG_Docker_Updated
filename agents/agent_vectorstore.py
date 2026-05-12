"""
Agent 3 — Hybrid Vector Store (FAISS + BM25)
Semantic search via FAISS OpenAI embeddings combined with BM25 keyword
search using Reciprocal Rank Fusion (RRF) for better retrieval accuracy.
"""
import os
import logging
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger("agent_vectorstore")

EMBED_MODEL     = os.getenv("EMBED_MODEL", "text-embedding-3-small")
FAISS_STORE_DIR = os.getenv("FAISS_STORE_DIR", "/data/faiss_store")


# ── Hybrid store wrapper ──────────────────────────────────────

class HybridStore:
    """
    Wraps FAISS (semantic) + BM25 (keyword) for hybrid retrieval.
    Pass this wherever a vectorstore is expected.
    """
    def __init__(self, faiss_store: FAISS, docs: list[Document]):
        self.faiss_store = faiss_store
        self.docs        = docs
        tokenized        = [d.page_content.lower().split() for d in docs]
        self.bm25        = BM25Okapi(tokenized)

    @property
    def index(self):
        return self.faiss_store.index


# ── Builder ───────────────────────────────────────────────────

def build_embeddings(model: str = EMBED_MODEL) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model)


def build_faiss_index(
    chunks: list[Document],
    embed_model: str = EMBED_MODEL,
    session_id: str = "default",
) -> HybridStore:
    """
    Embed all chunks, build FAISS index, and wrap with BM25 for hybrid search.
    Returns a HybridStore.
    """
    if not chunks:
        raise ValueError("No chunks provided to index.")

    embeddings  = build_embeddings(embed_model)
    logger.info(f"[VectorStore] Building FAISS index for {len(chunks)} chunks with {embed_model}…")
    faiss_store = FAISS.from_documents(chunks, embeddings)
    logger.info(f"[VectorStore] FAISS index built — {faiss_store.index.ntotal} vectors")
    return HybridStore(faiss_store, chunks)


# ── Hybrid search with Reciprocal Rank Fusion ─────────────────

def _rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion across multiple ranked lists of doc indices."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


def similarity_search_with_scores(
    store: HybridStore,
    query: str,
    top_k: int = 4,
) -> list[dict]:
    """
    Hybrid retrieval: FAISS semantic + BM25 keyword, fused with RRF.
    Falls back to pure FAISS if store is a plain FAISS object.
    """
    # ── Graceful fallback for plain FAISS objects ─────────────
    if isinstance(store, FAISS):
        return _faiss_only(store, query, top_k)

    fetch_k = min(top_k * 3, len(store.docs))  # retrieve wider pool for fusion

    # ── FAISS ranking ─────────────────────────────────────────
    faiss_hits = store.faiss_store.similarity_search_with_relevance_scores(query, k=fetch_k)
    # Map FAISS results to doc indices in store.docs by content match
    content_to_idx = {d.page_content: i for i, d in enumerate(store.docs)}
    faiss_ranking  = [
        content_to_idx[doc.page_content]
        for doc, _ in faiss_hits
        if doc.page_content in content_to_idx
    ]

    # ── BM25 ranking ──────────────────────────────────────────
    bm25_scores  = store.bm25.get_scores(query.lower().split())
    bm25_ranking = sorted(range(len(store.docs)), key=lambda i: bm25_scores[i], reverse=True)[:fetch_k]

    # ── Fuse with RRF ─────────────────────────────────────────
    fused = _rrf([faiss_ranking, bm25_ranking])
    top_indices = sorted(fused, key=lambda i: fused[i], reverse=True)[:top_k]

    # ── Build result hits ─────────────────────────────────────
    # Precompute FAISS score lookup for display
    faiss_score_map = {doc.page_content: score for doc, score in faiss_hits}

    hits = []
    for idx in top_indices:
        doc   = store.docs[idx]
        score = faiss_score_map.get(doc.page_content, bm25_scores[idx] / (max(bm25_scores) + 1e-9))
        hits.append({
            "text":        doc.page_content,
            "page":        doc.metadata.get("page", 0) + 1,
            "chunk_id":    doc.metadata.get("chunk_id", -1),
            "strategy":    doc.metadata.get("chunk_strategy", "unknown"),
            "score":       round(float(score), 4),
            "text_length": len(doc.page_content),
        })
    return hits


def _faiss_only(vectorstore: FAISS, query: str, top_k: int) -> list[dict]:
    """Pure FAISS search — used as fallback."""
    results = vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
    return [
        {
            "text":        doc.page_content,
            "page":        doc.metadata.get("page", 0) + 1,
            "chunk_id":    doc.metadata.get("chunk_id", -1),
            "strategy":    doc.metadata.get("chunk_strategy", "unknown"),
            "score":       round(score, 4),
            "text_length": len(doc.page_content),
        }
        for doc, score in results
    ]


# ── Persistence (saves underlying FAISS index) ────────────────

def save_faiss_index(store, session_id: str, store_dir: str = FAISS_STORE_DIR):
    faiss_store = store.faiss_store if isinstance(store, HybridStore) else store
    path = Path(store_dir) / session_id
    path.mkdir(parents=True, exist_ok=True)
    faiss_store.save_local(str(path))
    logger.info(f"[VectorStore] FAISS index saved to {path}")


def load_faiss_index(
    session_id: str,
    embed_model: str = EMBED_MODEL,
    store_dir: str = FAISS_STORE_DIR,
) -> FAISS | None:
    path = Path(store_dir) / session_id
    if not path.exists():
        return None
    embeddings = build_embeddings(embed_model)
    vs = FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
    logger.info(f"[VectorStore] FAISS index loaded from {path} — {vs.index.ntotal} vectors")
    return vs