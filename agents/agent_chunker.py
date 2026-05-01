"""
Agent 2 — Multi-Strategy Chunker (LangChain)
Supports 4 chunking strategies so you can compare retrieval quality:

  1. recursive   — RecursiveCharacterTextSplitter  (best general-purpose)
  2. token       — TokenTextSplitter               (respects LLM token limits)
  3. character   — CharacterTextSplitter           (simple separator-based)
  4. semantic    — SemanticChunker                 (meaning-based, uses embeddings)

Each strategy returns a list of LangChain Documents with chunk metadata.
"""
import logging
from typing import Literal
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    CharacterTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger("agent_chunker")

ChunkStrategy = Literal["recursive", "token", "character", "semantic"]


# ─────────────────────────────────────────────────────────────
# Strategy implementations
# ─────────────────────────────────────────────────────────────

def _recursive_chunk(documents: list[Document], chunk_size: int, overlap: int) -> list[Document]:
    """
    RecursiveCharacterTextSplitter — tries to split on paragraph/sentence/word
    boundaries before falling back to character splits. Best general-purpose choice.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"[Chunker:recursive] {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks


def _token_chunk(documents: list[Document], chunk_size: int, overlap: int) -> list[Document]:
    """
    TokenTextSplitter — splits on tiktoken token boundaries.
    Best when you want to stay within exact LLM token limits.
    chunk_size here = number of tokens (not characters).
    """
    splitter = TokenTextSplitter(
        encoding_name="cl100k_base",   # GPT-4o tokenizer
        chunk_size=chunk_size // 4,    # convert chars → ~tokens (4 chars ≈ 1 token)
        chunk_overlap=overlap // 4,
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"[Chunker:token] {len(chunks)} chunks (~{chunk_size//4} tokens each)")
    return chunks


def _character_chunk(documents: list[Document], chunk_size: int, overlap: int) -> list[Document]:
    """
    CharacterTextSplitter — splits purely on a single separator (\n\n).
    Fastest, but can produce very uneven chunk sizes.
    """
    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"[Chunker:character] {len(chunks)} chunks")
    return chunks


def _semantic_chunk(documents: list[Document], embed_model: str) -> list[Document]:
    """
    SemanticChunker — uses embeddings to find natural breakpoints
    where topic/meaning shifts. Produces the most coherent chunks
    but requires an embedding API call during chunking.
    """
    embeddings = OpenAIEmbeddings(model=embed_model)
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",   # split where embedding distance > 95th percentile
        breakpoint_threshold_amount=95,
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"[Chunker:semantic] {len(chunks)} meaning-based chunks")
    return chunks


# ─────────────────────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────────────────────

def chunk_documents(
    documents: list[Document],
    strategy: ChunkStrategy = "recursive",
    chunk_size: int = 800,
    overlap: int = 150,
    embed_model: str = "text-embedding-3-small",
) -> list[Document]:
    """
    Route to the chosen chunking strategy.
    Adds chunk_id and strategy metadata to each Document.
    """
    if strategy == "recursive":
        chunks = _recursive_chunk(documents, chunk_size, overlap)
    elif strategy == "token":
        chunks = _token_chunk(documents, chunk_size, overlap)
    elif strategy == "character":
        chunks = _character_chunk(documents, chunk_size, overlap)
    elif strategy == "semantic":
        chunks = _semantic_chunk(documents, embed_model)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Choose recursive|token|character|semantic")

    # Tag every chunk with strategy + sequential id
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_strategy"] = strategy
        chunk.metadata["chunk_size_cfg"] = chunk_size
        chunk.metadata["text_length"] = len(chunk.page_content)

    return chunks


def get_strategy_description(strategy: ChunkStrategy) -> str:
    descriptions = {
        "recursive": (
            "**RecursiveCharacterTextSplitter** — Tries paragraph → sentence → word boundaries. "
            "Best all-round choice. Produces natural, human-readable chunks."
        ),
        "token": (
            "**TokenTextSplitter** — Splits on GPT-4o token boundaries (tiktoken cl100k_base). "
            "Guarantees you never exceed the LLM context window per chunk."
        ),
        "character": (
            "**CharacterTextSplitter** — Splits on double-newlines (\\n\\n). "
            "Fast and simple. Works best for well-structured documents with clear sections."
        ),
        "semantic": (
            "**SemanticChunker** — Uses `text-embedding-3-small` to detect meaning shifts. "
            "Calls the OpenAI API during chunking. Produces coherent topic-based chunks. "
            "Slowest but highest quality for complex docs."
        ),
    }
    return descriptions.get(strategy, "Unknown strategy")
