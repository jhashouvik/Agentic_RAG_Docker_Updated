"""
Agent 3 — FAISS Vector Store (LangChain)
Uses LangChain's FAISS integration + OpenAIEmbeddings.
Supports save/load to disk for persistence across Streamlit sessions.
"""
import os
import logging
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger("agent_vectorstore")

# Default embedding model — change via EMBED_MODEL env var
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
FAISS_STORE_DIR = os.getenv("FAISS_STORE_DIR", "/data/faiss_store")


def build_embeddings(model: str = EMBED_MODEL) -> OpenAIEmbeddings:
    """Return LangChain OpenAIEmbeddings object."""
    return OpenAIEmbeddings(model=model)


def build_faiss_index(
    chunks: list[Document],
    embed_model: str = EMBED_MODEL,
    session_id: str = "default",
) -> FAISS:
    """
    Embed all chunks and build a FAISS index.
    Returns the LangChain FAISS vectorstore object.
    """
    if not chunks:
        raise ValueError("No chunks provided to index.")

    embeddings = build_embeddings(embed_model)
    logger.info(f"[VectorStore] Building FAISS index for {len(chunks)} chunks with {embed_model}…")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    logger.info(f"[VectorStore] FAISS index built — {vectorstore.index.ntotal} vectors")
    return vectorstore


def save_faiss_index(vectorstore: FAISS, session_id: str, store_dir: str = FAISS_STORE_DIR):
    """Persist FAISS index to disk."""
    path = Path(store_dir) / session_id
    path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(path))
    logger.info(f"[VectorStore] FAISS index saved to {path}")


def load_faiss_index(
    session_id: str,
    embed_model: str = EMBED_MODEL,
    store_dir: str = FAISS_STORE_DIR,
) -> FAISS | None:
    """Load a persisted FAISS index. Returns None if not found."""
    path = Path(store_dir) / session_id
    if not path.exists():
        return None
    embeddings = build_embeddings(embed_model)
    vectorstore = FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
    logger.info(f"[VectorStore] FAISS index loaded from {path} — {vectorstore.index.ntotal} vectors")
    return vectorstore


def similarity_search_with_scores(
    vectorstore: FAISS,
    query: str,
    top_k: int = 4,
) -> list[dict]:
    """
    Run cosine similarity search and return ranked results with scores.
    """
    results = vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
    hits = []
    for doc, score in results:
        hits.append({
            "text":       doc.page_content,
            "page":       doc.metadata.get("page", 0) + 1,
            "chunk_id":   doc.metadata.get("chunk_id", -1),
            "strategy":   doc.metadata.get("chunk_strategy", "unknown"),
            "score":      round(score, 4),
            "text_length": len(doc.page_content),
        })
    return hits
