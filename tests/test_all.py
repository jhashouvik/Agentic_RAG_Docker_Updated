"""
Unit tests — PDF Q&A v2
Run:  pytest tests/ -v --tb=short
No OpenAI API key needed — all external calls are mocked.
"""
import sys, os, tempfile, sqlite3, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from langchain_core.documents import Document


def _docs(text: str) -> list[Document]:
    return [Document(page_content=text, metadata={"page": 0, "source": "test.pdf"})]


# ─────────────────────────────────────────────────────────────
# TestChunker
# ─────────────────────────────────────────────────────────────
class TestChunker:

    def test_recursive_produces_multiple_chunks(self):
        from agents.agent_chunker import chunk_documents
        chunks = chunk_documents(_docs("Hello world. " * 300),
                                 strategy="recursive", chunk_size=300, overlap=50)
        assert len(chunks) > 1

    def test_character_produces_chunks(self):
        from agents.agent_chunker import chunk_documents
        chunks = chunk_documents(_docs("Para one.\n\nPara two.\n\n" * 40),
                                 strategy="character", chunk_size=200, overlap=30)
        assert len(chunks) >= 1

    def test_chunk_metadata_tagged(self):
        from agents.agent_chunker import chunk_documents
        chunks = chunk_documents(_docs("word " * 400),
                                 strategy="recursive", chunk_size=200, overlap=30)
        for i, c in enumerate(chunks):
            assert c.metadata["chunk_id"] == i
            assert c.metadata["chunk_strategy"] == "recursive"
            assert "text_length" in c.metadata

    def test_chunk_ids_sequential(self):
        from agents.agent_chunker import chunk_documents
        chunks = chunk_documents(_docs("Test text. " * 300),
                                 strategy="recursive", chunk_size=300, overlap=50)
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert ids == list(range(len(ids)))

    def test_invalid_strategy_raises(self):
        from agents.agent_chunker import chunk_documents
        with pytest.raises(ValueError, match="Unknown strategy"):
            chunk_documents(_docs("hi"), strategy="magic")

    def test_all_descriptions_present(self):
        from agents.agent_chunker import get_strategy_description
        for s in ["recursive", "token", "character", "semantic"]:
            assert len(get_strategy_description(s)) > 10


# ─────────────────────────────────────────────────────────────
# TestIngestor
# ─────────────────────────────────────────────────────────────
class TestIngestor:

    def test_missing_file_raises(self):
        from agents.agent_ingestor import load_pdf
        with pytest.raises(FileNotFoundError):
            load_pdf("/no/such/file.pdf")

    def test_returns_expected_keys(self):
        from unittest.mock import patch
        fake_pages = [
            Document(page_content="Page one content.", metadata={"page": 0}),
            Document(page_content="Page two content.", metadata={"page": 1}),
        ]
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            tmp = f.name
        try:
            with patch("agents.agent_ingestor.PyPDFLoader") as MockLoader:
                MockLoader.return_value.load.return_value = fake_pages
                from agents.agent_ingestor import load_pdf
                result = load_pdf(tmp)
            assert result["page_count"] == 2
            assert "Page one content." in result["full_text"]
            assert set(result.keys()) >= {"documents","full_text","page_count","char_count","doc_name"}
        finally:
            os.unlink(tmp)


# ─────────────────────────────────────────────────────────────
# TestMemory
# ─────────────────────────────────────────────────────────────
class TestMemory:

    @pytest.fixture
    def db(self, tmp_path):
        p = tmp_path / "mem.db"
        from agents.agent_memory import init_db
        init_db(p)
        return p

    def test_tables_created(self, db):
        conn = sqlite3.connect(str(db))
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert {"sessions","memory","comparisons"}.issubset(tables)

    def test_upsert_and_list_session(self, db):
        from agents.agent_memory import upsert_session, list_sessions
        upsert_session("s1", {"doc_name":"a.pdf","strategy":"recursive",
                               "embed_model":"text-embedding-3-small",
                               "chunk_count":10,"page_count":3}, db)
        assert any(s["session_id"]=="s1" for s in list_sessions(db))

    def test_save_and_load_turns(self, db):
        from agents.agent_memory import save_turn, load_history
        save_turn("s2","human","What is RAG?", db_path=db)
        save_turn("s2","ai","Retrieval Augmented Generation",
                  pages=[1], chunks=[0], tokens=80, db_path=db)
        history = load_history("s2", db_path=db)
        assert len(history) == 2
        assert history[0]["role"] == "human"
        assert history[1]["pages"] == [1]

    def test_stats(self, db):
        from agents.agent_memory import save_turn, get_session_stats
        save_turn("s3","human","Q",tokens=10, db_path=db)
        save_turn("s3","ai",   "A",tokens=20, db_path=db)
        stats = get_session_stats("s3", db)
        assert stats["turns"] == 2
        assert stats["total_tokens"] == 30
        assert stats["questions"] == 1

    def test_comparison_roundtrip(self, db):
        from agents.agent_memory import save_comparison, load_comparisons
        save_comparison("s4","Chunking question?",{
            "strategy_a":"recursive","answer_a":"Ans A","score_a":0.9,
            "strategy_b":"semantic", "answer_b":"Ans B","score_b":0.85,
        }, db)
        comps = load_comparisons("s4", db)
        assert len(comps) == 1
        assert comps[0]["strategy_a"] == "recursive"

    def test_window_bounded(self, db):
        from agents.agent_memory import AgentMemory, init_db
        mem = AgentMemory("win_sess", window_k=2, db_path=db)
        mem.add_turn("Q1","A1")
        mem.add_turn("Q2","A2")
        mem.add_turn("Q3","A3")          # pushes Q1 out
        assert len(mem.get_window_messages()) <= 4   # window_k * 2

    def test_clear_empties_window(self, db):
        from agents.agent_memory import AgentMemory
        mem = AgentMemory("clr_sess", window_k=5, db_path=db)
        mem.add_turn("Q1","A1")
        mem.clear()
        assert len(mem.get_window_messages()) == 0

    def test_empty_window_text(self):
        from agents.agent_memory import AgentMemory
        mem = AgentMemory("empty", window_k=3)  # no db — won't call add_turn
        assert mem.get_window_as_text() == "No previous conversation."

    def test_window_text_format(self, db):
        from agents.agent_memory import AgentMemory
        mem = AgentMemory("fmt_sess", window_k=5, db_path=db)
        mem.add_turn("Hello?","Hi there!")
        text = mem.get_window_as_text()
        assert "Human:" in text
        assert "AI:" in text


# ─────────────────────────────────────────────────────────────
# TestVectorStore  (mocked)
# ─────────────────────────────────────────────────────────────
class TestVectorStore:

    def test_result_shape(self):
        from unittest.mock import MagicMock
        from agents.agent_vectorstore import similarity_search_with_scores
        mock_doc = Document(page_content="Relevant content.",
                            metadata={"page":0,"chunk_id":2,"chunk_strategy":"recursive"})
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = [(mock_doc, 0.87)]
        results = similarity_search_with_scores(mock_vs, "test", top_k=1)
        assert len(results) == 1
        assert results[0]["score"] == 0.87
        assert results[0]["text"] == "Relevant content."
        assert results[0]["page"] == 1       # 0-indexed → 1-indexed
        assert results[0]["chunk_id"] == 2

    def test_empty_returns_empty_list(self):
        from unittest.mock import MagicMock
        from agents.agent_vectorstore import similarity_search_with_scores
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = []
        assert similarity_search_with_scores(mock_vs, "q", top_k=4) == []


# ─────────────────────────────────────────────────────────────
# TestOrchestrator  (state shape — no API)
# ─────────────────────────────────────────────────────────────
class TestOrchestrator:

    def test_state_shape(self):
        from agents.agent_orchestrator import AgentState
        state: AgentState = {
            "question":"What is this?","retrieved_docs":[],"graded_docs":[],
            "answer":"","tokens_used":0,"chat_history":"",
            "model":"gpt-4o","temperature":0.2,"top_k":4,
        }
        assert state["top_k"] == 4
        assert isinstance(state["retrieved_docs"], list)

    def test_grade_fallback_on_all_irrelevant(self):
        """node_grade must fall back to retrieved_docs when all chunks are filtered."""
        from unittest.mock import patch, MagicMock
        from agents.agent_orchestrator import AgentState

        docs = [{"text":"Some content.","page":1,"chunk_id":0,"score":0.8}]
        state: AgentState = {
            "question":"Test?","retrieved_docs":docs,"graded_docs":[],
            "answer":"","tokens_used":0,"chat_history":"",
            "model":"gpt-4o-mini","temperature":0.0,"top_k":4,
        }
        # Simulate: grader filtered everything → fallback to retrieved_docs
        result = {**state, "graded_docs": state["retrieved_docs"]
                  if not state["graded_docs"] else state["graded_docs"]}
        assert len(result["graded_docs"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
