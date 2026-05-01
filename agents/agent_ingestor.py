"""
Agent 1 — PDF Ingestor (LangChain)
Loads PDF using LangChain PyPDFLoader and returns raw LangChain Documents.
"""
import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger("agent_ingestor")


def load_pdf(pdf_path: str) -> dict:
    """
    Load a PDF using LangChain's PyPDFLoader.
    Returns raw Document list + metadata summary.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(str(path))
    pages: list[Document] = loader.load()

    if not pages:
        raise ValueError("No pages extracted. PDF may be image-only or corrupted.")

    full_text = "\n\n".join(p.page_content for p in pages)
    logger.info(f"[Ingestor] Loaded {len(pages)} pages, {len(full_text)} chars from '{path.name}'")

    return {
        "documents": pages,          # list[Document] with page metadata
        "full_text": full_text,
        "page_count": len(pages),
        "char_count": len(full_text),
        "doc_name": path.name,
    }
