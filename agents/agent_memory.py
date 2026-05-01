"""
Agent 4 — Memory Layer  (modern LangChain + SQLite)

  SQLite (memory.db)       — permanent, zero-server, scales to millions of rows
  ChatMessageHistory       — in-process sliding window, last K turns → LLM prompt
"""
import sqlite3, json, logging
from datetime import datetime, timezone
from pathlib import Path
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger("agent_memory")
DEFAULT_DB_PATH = Path("/data/memory.db")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(db_path: Path = DEFAULT_DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY, doc_name TEXT, strategy TEXT,
            embed_model TEXT, chunk_count INTEGER, page_count INTEGER,
            created_at TEXT, updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
            pages TEXT, chunks TEXT, tokens INTEGER DEFAULT 0,
            strategy TEXT, timestamp TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );
        CREATE TABLE IF NOT EXISTS comparisons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL, question TEXT NOT NULL,
            strategy_a TEXT, answer_a TEXT, score_a REAL,
            strategy_b TEXT, answer_b TEXT, score_b REAL, timestamp TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_mem_session   ON memory(session_id);
        CREATE INDEX IF NOT EXISTS idx_mem_timestamp ON memory(timestamp);
    """)
    conn.commit(); conn.close()
    logger.info(f"[Memory] SQLite DB ready at {db_path}")


def upsert_session(session_id: str, meta: dict, db_path: Path = DEFAULT_DB_PATH):
    now = _now()
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        INSERT INTO sessions
          (session_id,doc_name,strategy,embed_model,chunk_count,page_count,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(session_id) DO UPDATE SET
          strategy=excluded.strategy, chunk_count=excluded.chunk_count, updated_at=excluded.updated_at
    """, (session_id, meta.get("doc_name",""), meta.get("strategy","recursive"),
          meta.get("embed_model","text-embedding-3-small"),
          meta.get("chunk_count",0), meta.get("page_count",0), now, now))
    conn.commit(); conn.close()


def list_sessions(db_path: Path = DEFAULT_DB_PATH) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_turn(session_id: str, role: str, content: str, *,
              pages: list = None, chunks: list = None, tokens: int = 0,
              strategy: str = "", db_path: Path = DEFAULT_DB_PATH):
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        INSERT INTO memory
          (session_id,role,content,pages,chunks,tokens,strategy,timestamp)
        VALUES (?,?,?,?,?,?,?,?)
    """, (session_id, role, content,
          json.dumps(pages or []), json.dumps(chunks or []),
          tokens, strategy, _now()))
    conn.commit(); conn.close()


def load_history(session_id: str, limit: int = 100,
                 db_path: Path = DEFAULT_DB_PATH) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM memory WHERE session_id=? ORDER BY timestamp ASC LIMIT ?",
        (session_id, limit)
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["pages"]  = json.loads(d.get("pages")  or "[]")
        d["chunks"] = json.loads(d.get("chunks") or "[]")
        result.append(d)
    return result


def get_session_stats(session_id: str, db_path: Path = DEFAULT_DB_PATH) -> dict:
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("""
        SELECT COUNT(*), SUM(tokens),
               SUM(CASE WHEN role='human' THEN 1 ELSE 0 END),
               MIN(timestamp), MAX(timestamp)
        FROM memory WHERE session_id=?
    """, (session_id,)).fetchone()
    conn.close()
    return {"turns": row[0] or 0, "total_tokens": row[1] or 0,
            "questions": row[2] or 0, "first_turn": row[3], "last_turn": row[4]}


def save_comparison(session_id: str, question: str, results: dict,
                    db_path: Path = DEFAULT_DB_PATH):
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        INSERT INTO comparisons
          (session_id,question,strategy_a,answer_a,score_a,
           strategy_b,answer_b,score_b,timestamp)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (session_id, question,
          results.get("strategy_a"), results.get("answer_a"), results.get("score_a", 0),
          results.get("strategy_b"), results.get("answer_b"), results.get("score_b", 0),
          _now()))
    conn.commit(); conn.close()


def load_comparisons(session_id: str, db_path: Path = DEFAULT_DB_PATH) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM comparisons WHERE session_id=? ORDER BY timestamp DESC",
        (session_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────
# AgentMemory — unified interface
# ─────────────────────────────────────────────────────────────
class AgentMemory:
    """
    Modern LangChain agent memory.

    In-process : ChatMessageHistory holds all turns.
                 get_window_messages() slices last window_k pairs for LLM context.
    Persistent : every turn written to SQLite immediately (zero-server, free).

    Scaling note:
      SQLite comfortably handles millions of rows.
      For multi-user production: swap SQLite for Postgres
      and ChatMessageHistory for RedisChatMessageHistory.
    """
    def __init__(self, session_id: str, window_k: int = 5,
                 db_path: Path = DEFAULT_DB_PATH):
        self.session_id = session_id
        self.window_k   = window_k
        self.db_path    = db_path          # explicit — no module-level default capture
        self._history   = ChatMessageHistory()

    def add_turn(self, question: str, answer: str, meta: dict = None):
        meta = meta or {}
        self._history.add_user_message(question)
        self._history.add_ai_message(answer)
        save_turn(self.session_id, "human", question,
                  strategy=meta.get("strategy", ""), db_path=self.db_path)
        save_turn(self.session_id, "ai", answer,
                  pages=meta.get("pages", []), chunks=meta.get("chunks", []),
                  tokens=meta.get("tokens", 0), strategy=meta.get("strategy", ""),
                  db_path=self.db_path)

    def get_window_messages(self) -> list:
        return self._history.messages[-(self.window_k * 2):]

    def get_window_as_text(self) -> str:
        msgs = self.get_window_messages()
        if not msgs:
            return "No previous conversation."
        lines = []
        for m in msgs:
            role = "Human" if isinstance(m, HumanMessage) else "AI"
            lines.append(f"{role}: {m.content}")
        return "\n".join(lines)

    def all_messages(self) -> list:
        return self._history.messages

    def clear(self):
        self._history.clear()
