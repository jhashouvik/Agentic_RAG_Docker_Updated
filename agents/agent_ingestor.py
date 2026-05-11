"""
Agent 1 — PDF Ingestor (LangChain + OCR fallback)
Primary:  PyPDFLoader  — fast, for text-layer PDFs
Fallback: Tesseract OCR — for scanned / mobile-captured PDFs
"""
import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger("agent_ingestor")


def _ocr_pdf(pdf_path: str) -> list[Document]:
    """
    OCR fallback using pdf2image + pytesseract.
    Converts each PDF page to an image then extracts text via Tesseract.
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError as e:
        raise ImportError(
            "OCR packages missing. Run: pip install pytesseract pdf2image"
        ) from e

    logger.info("[Ingestor] Falling back to OCR (Tesseract) for image-based PDF…")
    images = convert_from_path(pdf_path, dpi=150)
    pages = []
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img, lang="eng").strip()
        pages.append(Document(
            page_content=text,
            metadata={"page": i, "source": pdf_path, "ocr": True},
        ))
        logger.info(f"[Ingestor] OCR page {i + 1}/{len(images)} — {len(text)} chars extracted")
    return pages


def load_pdf(pdf_path: str) -> dict:
    """
    Load a PDF — tries PyPDFLoader first, falls back to OCR if no text found.
    Returns raw Document list + metadata summary.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Primary: fast text-layer extraction
    loader = PyPDFLoader(str(path))
    pages: list[Document] = loader.load()
    full_text = "\n\n".join(p.page_content for p in pages).strip()

    # Fallback: OCR for scanned / mobile-captured PDFs
    if not full_text:
        logger.warning("[Ingestor] No text found via PyPDFLoader — trying OCR fallback…")
        pages = _ocr_pdf(pdf_path)
        full_text = "\n\n".join(p.page_content for p in pages).strip()

    if not full_text:
        raise ValueError(
            "No text extracted even after OCR. "
            "Check the PDF is not corrupted and Tesseract is installed."
        )

    logger.info(
        f"[Ingestor] Loaded {len(pages)} pages, {len(full_text)} chars "
        f"from '{path.name}' "
        f"(ocr={'yes' if pages and pages[0].metadata.get('ocr') else 'no'})"
    )

    return {
        "documents":  pages,
        "full_text":  full_text,
        "page_count": len(pages),
        "char_count": len(full_text),
        "doc_name":   path.name,
    }