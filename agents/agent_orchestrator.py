"""
Agent 5 — LangGraph Orchestrator
Builds a stateful multi-agent graph using LangGraph (Microsoft-friendly pattern,
mirrors AutoGen / Semantic Kernel agent topology).

Graph nodes:
  retrieve   → cosine search FAISS for relevant chunks
  grade      → LLM grades retrieved chunks for relevance (self-reflection)
  generate   → RAG answer with memory-injected chat history
  summarize  → one-shot document summary (separate path)

State flows through a TypedDict so each node reads/writes a shared state object.
"""
import logging
import os
from typing import TypedDict, Annotated, Sequence
import operator

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END

logger = logging.getLogger("agent_orchestrator")


# ─────────────────────────────────────────────────────────────
# Shared Agent State
# ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    question:       str
    retrieved_docs: list[dict]
    graded_docs:    list[dict]
    answer:         str
    tokens_used:    int
    chat_history:   str           # sliding window from AgentMemory
    model:          str
    temperature:    float
    top_k:          int


# ─────────────────────────────────────────────────────────────
# Node 1 — Retrieve
# ─────────────────────────────────────────────────────────────

def node_retrieve(state: AgentState, vectorstore: FAISS) -> AgentState:
    """Embed the question and pull top-K chunks from FAISS."""
    from agents.agent_vectorstore import similarity_search_with_scores
    question = state["question"]
    hits = similarity_search_with_scores(vectorstore, question, top_k=state.get("top_k", 4))
    logger.info(f"[Retrieve] Got {len(hits)} chunks for: '{question[:60]}'")
    return {**state, "retrieved_docs": hits}


# ─────────────────────────────────────────────────────────────
# Node 2 — Grade (Self-Reflection)
# ─────────────────────────────────────────────────────────────

GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a relevance grader. Given a question and a document excerpt, "
     "respond with exactly one word: RELEVANT or IRRELEVANT.\n"
     "Do not explain. Just one word."),
    ("human", "Question: {question}\n\nDocument excerpt:\n{doc_text}"),
])


def node_grade(state: AgentState) -> AgentState:
    """
    LLM grades each retrieved chunk for relevance.
    Filters out chunks that are clearly off-topic.
    Self-reflection pattern — the agent checks its own retrieval quality.
    """
    llm   = ChatOpenAI(model=state.get("model", "gpt-4o-mini"), temperature=0)
    chain = GRADE_PROMPT | llm | StrOutputParser()

    graded = []
    for doc in state["retrieved_docs"]:
        verdict = chain.invoke({
            "question": state["question"],
            "doc_text": doc["text"][:600],
        }).strip().upper()
        if "RELEVANT" in verdict:
            graded.append(doc)

    logger.info(f"[Grade] {len(graded)}/{len(state['retrieved_docs'])} chunks passed relevance filter")
    # Fall back to all chunks if grader filtered everything
    if not graded:
        graded = state["retrieved_docs"]
        logger.warning("[Grade] All chunks filtered — using full retrieved set as fallback")

    return {**state, "graded_docs": graded}


# ─────────────────────────────────────────────────────────────
# Node 3 — Generate (RAG with memory)
# ─────────────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are a precise PDF analyst with access to document excerpts and conversation history.

RULES:
1. Answer ONLY from the provided context excerpts.
2. If the context is insufficient, say so clearly — do not hallucinate.
3. Reference page numbers: "As stated on Page 3…"
4. Use bullet points for multi-part answers.
5. Leverage conversation history for follow-up questions.

CONVERSATION HISTORY (last {window_k} turns):
{chat_history}
---
"""),
    ("human",
     """DOCUMENT EXCERPTS:
{context}

---
QUESTION: {question}

Answer based strictly on the excerpts above:"""),
])


def node_generate(state: AgentState) -> AgentState:
    """RAG generation with memory-injected conversation history."""
    llm = ChatOpenAI(
        model=state.get("model", "gpt-4o"),
        temperature=state.get("temperature", 0.2),
    )
    chain = RAG_PROMPT | llm | StrOutputParser()

    # Format context from graded docs
    context_parts = []
    for doc in state["graded_docs"]:
        context_parts.append(
            f"[Page {doc['page']} | Chunk #{doc['chunk_id']} | Score {doc['score']}]\n"
            f"{doc['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    answer = chain.invoke({
        "context":      context,
        "question":     state["question"],
        "chat_history": state.get("chat_history", "No previous conversation."),
        "window_k":     5,
    })

    # Approximate token count
    tokens = len(answer.split()) * 2 + len(context.split())
    logger.info(f"[Generate] Answer produced (~{tokens} tokens)")
    return {**state, "answer": answer, "tokens_used": tokens}


# ─────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────

def build_qa_graph(vectorstore: FAISS):
    """
    Build the LangGraph agent graph.
    Returns a compiled graph you can invoke with an AgentState dict.

    Flow: retrieve → grade → generate → END
    """
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("retrieve", lambda s: node_retrieve(s, vectorstore))
    workflow.add_node("grade",    node_grade)
    workflow.add_node("generate", node_generate)

    # Define edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_edge("grade",    "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# ─────────────────────────────────────────────────────────────
# Summarization (standalone — not in the main graph)
# ─────────────────────────────────────────────────────────────

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a document analyst. Produce a structured summary with sections:\n"
     "## Overview\n## Key Topics\n## Key Findings\n## Important Data (dates, numbers, names)\n\n"
     "Be concise and scannable. Use bullet points within sections."),
    ("human", "Summarize this document:\n\n{text}"),
])


def summarize_document(full_text: str, model: str = "gpt-4o", temperature: float = 0.2) -> str:
    llm   = ChatOpenAI(model=model, temperature=temperature)
    chain = SUMMARY_PROMPT | llm | StrOutputParser()
    try:
        return chain.invoke({"text": full_text[:12000]})
    except Exception as e:
        logger.error(f"[Summarize] Failed: {e}")
        return f"Summary unavailable: {e}"
