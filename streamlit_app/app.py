"""
PDF Q&A Multi-Agent System — Streamlit App v2
Modern stack: LangChain + LangGraph + FAISS + SQLite memory
"""
import os, sys, uuid, tempfile, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from agents.agent_ingestor      import load_pdf
from agents.agent_chunker       import chunk_documents, get_strategy_description, ChunkStrategy
from agents.agent_vectorstore   import build_faiss_index, similarity_search_with_scores
from agents.agent_memory        import AgentMemory, init_db, upsert_session, get_session_stats, save_comparison, load_comparisons
from agents.agent_orchestrator  import build_qa_graph, summarize_document, AgentState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
init_db(DATA_DIR / "memory.db")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Q&A · LangGraph Agents",
    page_icon="🧠", layout="wide",
    initial_sidebar_state="expanded",
)

def _inject_css():
    dark = st.session_state.get("dark_mode", True)
    if dark:
        bg, sb_bg, sb_bd = "#0d1117", "#0d1117", "#21262d"
        h, p, sub        = "#e6edf3", "#c9d1d9", "#8b949e"
        card, bd         = "#161b27", "#21262d"
        chat_q, chat_a   = "#1c2333", "#0d2016"
        file_bg, file_bd = "#161b27", "#30363d"
        ca_bg, ca_bd     = "#0d1f2d", "#1f4068"
        cb_bg, cb_bd     = "#1a0d2d", "#4a1268"
    else:
        bg, sb_bg, sb_bd = "#f6f8fa", "#ffffff", "#d0d7de"
        h, p, sub        = "#1f2328", "#24292f", "#57606a"
        card, bd         = "#ffffff", "#d0d7de"
        chat_q, chat_a   = "#ddf4ff", "#dafbe1"
        file_bg, file_bd = "#f6f8fa", "#d0d7de"
        ca_bg, ca_bd     = "#ddf4ff", "#54aeff"
        cb_bg, cb_bd     = "#fbefff", "#d2a8ff"

    st.markdown(f"""<style>
[data-testid="stAppViewContainer"]{{background:{bg}}}
[data-testid="stSidebar"]{{background:{sb_bg};border-right:1px solid {sb_bd}}}
.block-container{{padding:1.5rem 2rem 2rem}}
h1,h2,h3{{color:{h}}}
p,li,label{{color:{p}}}
.hero{{font-size:1.9rem;font-weight:800;color:{h};margin-bottom:.2rem;padding-top:2.5rem;line-height:1.3}}
.hero-sub{{color:{sub};font-size:.92rem;margin-bottom:1rem}}
.pill{{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.7rem;font-weight:600;margin:2px}}
.pill-blue  {{background:rgba(88,166,255,.15);color:#58a6ff;border:1px solid rgba(88,166,255,.3)}}
.pill-green {{background:rgba(63,185,80,.15); color:#3fb950;border:1px solid rgba(63,185,80,.3)}}
.pill-amber {{background:rgba(255,166,0,.15); color:#e3b341;border:1px solid rgba(255,166,0,.3)}}
.pill-purple{{background:rgba(139,99,255,.15);color:#a78bfa;border:1px solid rgba(139,99,255,.3)}}
.pill-red   {{background:rgba(248,81,73,.15); color:#f85149;border:1px solid rgba(248,81,73,.3)}}
.sbox{{background:{card};border:1px solid {bd};border-radius:10px;padding:1rem;text-align:center;margin-bottom:.5rem}}
.sval{{font-size:1.55rem;font-weight:700;color:#58a6ff}}
.slbl{{font-size:.72rem;color:{sub};margin-top:2px}}
.chat-q{{background:{chat_q};border-left:3px solid #58a6ff;border-radius:0 10px 10px 10px;padding:.75rem 1rem;margin:.5rem 0;color:{h};font-size:.9rem}}
.chat-a{{background:{chat_a};border-left:3px solid #3fb950;border-radius:0 10px 10px 10px;padding:.75rem 1rem;margin:.25rem 0;color:{h};font-size:.9rem}}
.chat-meta{{font-size:.68rem;color:{sub};border-top:1px solid {bd};padding-top:.35rem;margin-top:.5rem}}
.chunk-box{{background:{card};border:1px solid {bd};border-radius:8px;padding:.6rem .9rem;margin:.25rem 0;font-size:.8rem;color:{p}}}
.node-active{{color:#3fb950;font-weight:600}}
.compare-a{{background:{ca_bg};border:1px solid {ca_bd};border-radius:10px;padding:1rem}}
.compare-b{{background:{cb_bg};border:1px solid {cb_bd};border-radius:10px;padding:1rem}}
div[data-testid="stFileUploader"]{{border:1.5px dashed {file_bd};border-radius:10px;background:{file_bg};padding:.5rem}}
</style>""", unsafe_allow_html=True)

# ── Session init ──────────────────────────────────────────────
def _init():
    for k, v in {
        "sid": str(uuid.uuid4())[:8], "chat": [], "loaded": False,
        "doc_name": None, "chunks": 0, "chars": 0, "pages": 0,
        "summary": None, "vs": None, "graph": None, "memory": None,
        "strategy": "recursive", "total_tokens": 0,
        "vs_by_strategy": {},   # strategy → vectorstore
        "documents": None,      # raw LangChain docs for on-demand strategy-B index
        "dark_mode": True,
    }.items():
        if k not in st.session_state: st.session_state[k] = v
_init()
_inject_css()

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️  Settings")
    st.divider()

    _dark = st.toggle(
        "🌙 Dark Mode" if st.session_state.dark_mode else "☀️ Light Mode",
        value=st.session_state.dark_mode,
        help="Switch between dark and light theme."
    )
    if _dark != st.session_state.dark_mode:
        st.session_state.dark_mode = _dark
        st.rerun()

    st.divider()
    strategy = st.selectbox(
        "📐 Chunking Strategy",
        ["recursive", "token", "character", "semantic"],
        index=["recursive","token","character","semantic"].index(st.session_state.strategy),
        help="Select how the PDF is split into chunks for embedding."
    )
    st.caption(get_strategy_description(strategy))

    model = st.selectbox("🤖 LLM", ["gpt-4o", "gpt-4o-mini"])
    embed_model = st.selectbox("🔢 Embedding Model",
        ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"])

    st.divider()
    chunk_size = st.slider("Chunk size (chars)", 300, 2000, 800, 100)
    overlap    = st.slider("Overlap (chars)", 50, 400, 150, 50)
    top_k      = st.slider("Top-K retrieval", 2, 10, 4)
    temperature= st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    st.divider()
    compare_mode = st.toggle("⚖️  Comparison Mode", value=False,
        help="Run two chunking strategies on the same question and compare answers side-by-side.")
    if compare_mode:
        strategy_b = st.selectbox("Strategy B", ["token","character","semantic","recursive"],
            help="Second strategy to compare against.")

    st.divider()
    if st.button("🗑️ Reset Session", use_container_width=True):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()

    if st.session_state.loaded:
        stats = get_session_stats(st.session_state.sid, DATA_DIR / "memory.db")
        st.markdown("**Memory Stats**")
        st.caption(f"Questions: `{stats['questions']}`")
        st.caption(f"Total tokens: `{st.session_state.total_tokens:,}`")
        st.caption(f"DB turns: `{stats['turns']}`")

    st.markdown("---")
    st.caption(f"Session: `{st.session_state.sid}`")
    st.caption("LangChain · LangGraph · FAISS · SQLite")

# ── HEADER ────────────────────────────────────────────────────
st.markdown('<div class="hero">🧠 PDF Intelligence · Multi-Agent + LangGraph</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">LangChain chunking · LangGraph agent graph · FAISS retrieval · SQLite memory</div>', unsafe_allow_html=True)
st.markdown(
    f'<span class="pill pill-blue">LangGraph</span>'
    f'<span class="pill pill-green">{model}</span>'
    f'<span class="pill pill-amber">{embed_model}</span>'
    f'<span class="pill pill-purple">{strategy.upper()}</span>'
    f'<span class="pill pill-red">FAISS</span>',
    unsafe_allow_html=True
)

# ── AGENT GRAPH DIAGRAM ───────────────────────────────────────
with st.expander("🔀 LangGraph Agent Flow", expanded=False):
    cols = st.columns(5)
    nodes = [
        ("📥", "Ingestor", "PyPDFLoader"),
        ("✂️",  "Chunker",  "4 strategies"),
        ("🔢", "Embedder", "FAISS index"),
        ("🎯", "Retriever","Top-K search"),
        ("🤖", "LLM Judge","Grade + Answer"),
    ]
    for col, (icon, name, desc) in zip(cols, nodes):
        with col:
            st.markdown(f"""
            <div class="chunk-box" style="text-align:center">
              <div style="font-size:1.4rem">{icon}</div>
              <div style="font-weight:700;color:#e6edf3;margin-top:4px">{name}</div>
              <div style="font-size:.72rem;color:#8b949e">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ── UPLOAD ────────────────────────────────────────────────────
st.divider()
st.markdown("### 📤 Step 1 — Upload PDF")

uploaded = st.file_uploader("Drop PDF here", type=["pdf"], label_visibility="collapsed")

if uploaded and (not st.session_state.loaded or uploaded.name != st.session_state.doc_name):
    prog = st.progress(0, "Starting…")
    try:
        # Agent 1: Ingest
        prog.progress(10, "📄 Agent 1: Loading PDF with PyPDFLoader…")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded.read()); tmp = f.name
        doc_data = load_pdf(tmp)
        os.unlink(tmp)

        if not doc_data["full_text"].strip():
            st.error("No text extracted — PDF may be image-only.")
            st.stop()

        # Agent 2: Chunk with selected strategy
        prog.progress(25, f"✂️ Agent 2: Chunking with [{strategy}] strategy…")
        chunks = chunk_documents(
            doc_data["documents"], strategy=strategy,
            chunk_size=chunk_size, overlap=overlap, embed_model=embed_model
        )

        # Agent 3: Embed + FAISS
        prog.progress(50, f"🔢 Agent 3: Embedding {len(chunks)} chunks → FAISS…")
        vs = build_faiss_index(chunks, embed_model=embed_model, session_id=st.session_state.sid)

        # If compare mode: also build the second strategy's index
        if compare_mode:
            prog.progress(65, f"🔢 Building second FAISS index [{strategy_b}]…")
            chunks_b = chunk_documents(
                doc_data["documents"], strategy=strategy_b,
                chunk_size=chunk_size, overlap=overlap, embed_model=embed_model
            )
            vs_b = build_faiss_index(chunks_b, embed_model=embed_model,
                                      session_id=st.session_state.sid + "_b")
            st.session_state.vs_by_strategy[strategy_b] = (vs_b, len(chunks_b))

        # Agent 4: LangGraph graph
        prog.progress(75, "🔀 Agent 4: Building LangGraph agent graph…")
        graph = build_qa_graph(vs)

        # Summary
        prog.progress(85, "📋 Summarizing document…")
        summary = summarize_document(doc_data["full_text"], model=model)

        # Clear stale state from any previously loaded document
        if st.session_state.get("memory"):
            st.session_state.memory.clear()

        # Memory
        memory = AgentMemory(session_id=st.session_state.sid, window_k=5,
                             db_path=DATA_DIR / "memory.db")
        upsert_session(st.session_state.sid, {
            "doc_name": uploaded.name, "strategy": strategy,
            "embed_model": embed_model, "chunk_count": len(chunks),
            "page_count": doc_data["page_count"],
        }, DATA_DIR / "memory.db")

        # Store — reset chat and strategy indexes so no cross-doc contamination
        st.session_state.update({
            "vs": vs, "graph": graph, "memory": memory,
            "loaded": True, "doc_name": uploaded.name,
            "strategy": strategy, "summary": summary,
            "chunks": len(chunks), "chars": doc_data["char_count"],
            "pages": doc_data["page_count"],
            "documents": doc_data["documents"],
            "chat": [],
            "vs_by_strategy": {strategy: (vs, len(chunks))},
        })

        prog.progress(100, "✅ Done!")
        st.success(f"✅ **{uploaded.name}** — {doc_data['page_count']} pages · {len(chunks)} chunks · [{strategy}] strategy")
        prog.empty()

    except Exception as e:
        prog.empty()
        st.error(f"Pipeline failed: {e}")
        raise

# ── STATS ROW ─────────────────────────────────────────────────
if st.session_state.loaded:
    c1,c2,c3,c4,c5 = st.columns(5)
    for col, val, lbl in [
        (c1, st.session_state.pages,   "Pages"),
        (c2, st.session_state.chunks,  "Chunks Indexed"),
        (c3, f"{st.session_state.chars:,}", "Characters"),
        (c4, strategy.upper(),         "Chunk Strategy"),
        (c5, f"{st.session_state.total_tokens:,}", "Tokens Used"),
    ]:
        col.markdown(f'<div class="sbox"><div class="sval">{val}</div><div class="slbl">{lbl}</div></div>', unsafe_allow_html=True)

    # Summary
    with st.expander("📋 AI Document Summary", expanded=True):
        st.markdown(st.session_state.summary)

    # Chunking comparison info
    with st.expander("📐 Chunking Strategy Comparison", expanded=False):
        st.markdown("### How the 4 strategies differ on your document")
        s_cols = st.columns(4)
        strategy_info = [
            ("recursive", "🔁 Recursive", "#58a6ff",
             "Splits on \\n\\n → \\n → '. ' → ' '. Best all-round. Produces natural paragraphs."),
            ("token", "🔢 Token", "#3fb950",
             "Splits on GPT-4o token boundaries. Never exceeds LLM context window per chunk."),
            ("character", "📝 Character", "#e3b341",
             "Splits only on \\n\\n. Fastest. Works well for structured reports."),
            ("semantic", "🧠 Semantic", "#a78bfa",
             "Uses embeddings to find meaning shifts. Slowest. Best for complex/mixed docs."),
        ]
        for col, (strat, label, color, desc) in zip(s_cols, strategy_info):
            active = "✅ Active" if strat == st.session_state.strategy else ""
            col.markdown(f"""
            <div style="background:#161b27;border:1px solid {color}33;border-radius:10px;padding:.75rem;min-height:140px">
              <div style="color:{color};font-weight:700;font-size:.85rem">{label}</div>
              <div style="font-size:.72rem;color:#8b949e;margin-top:.4rem">{desc}</div>
              <div style="font-size:.72rem;color:{color};margin-top:.5rem;font-weight:600">{active}</div>
            </div>""", unsafe_allow_html=True)

    # ── Q&A SECTION ───────────────────────────────────────────
    st.divider()

    question = st.chat_input("Ask anything about the PDF…")

    if question:
        if compare_mode and strategy_b not in st.session_state.vs_by_strategy:
            if st.session_state.get("documents"):
                with st.spinner(f"🔢 Building Strategy B index [{strategy_b}] on-the-fly…"):
                    chunks_b = chunk_documents(
                        st.session_state.documents, strategy=strategy_b,
                        chunk_size=chunk_size, overlap=overlap, embed_model=embed_model,
                    )
                    vs_b = build_faiss_index(chunks_b, embed_model=embed_model,
                                             session_id=st.session_state.sid + "_b")
                    st.session_state.vs_by_strategy[strategy_b] = (vs_b, len(chunks_b))
            else:
                st.warning("⚠️ Please re-upload the PDF with Comparison Mode already enabled.")

        if compare_mode and strategy_b in st.session_state.vs_by_strategy:
            # ── COMPARISON MODE ───────────────────────────────
            with st.spinner("⚖️ Running both strategies in parallel…"):
                vs_a, _ = st.session_state.vs_by_strategy.get(strategy, (st.session_state.vs, 0))
                vs_b, _ = st.session_state.vs_by_strategy[strategy_b]

                graph_a = build_qa_graph(vs_a)
                graph_b = build_qa_graph(vs_b)

                state_a: AgentState = {
                    "question": question, "retrieved_docs": [], "graded_docs": [],
                    "answer": "", "tokens_used": 0,
                    "chat_history": st.session_state.memory.get_window_as_text(),
                    "model": model, "temperature": temperature, "top_k": top_k,
                }
                state_b = {**state_a}

                result_a = graph_a.invoke(state_a)
                result_b = graph_b.invoke(state_b)

            save_comparison(st.session_state.sid, question, {
                "strategy_a": strategy,   "answer_a": result_a["answer"], "score_a": 0,
                "strategy_b": strategy_b, "answer_b": result_b["answer"], "score_b": 0,
            }, DATA_DIR / "memory.db")

            st.session_state.memory.add_turn(question, result_a["answer"])
            st.session_state.total_tokens += result_a.get("tokens_used", 0) + result_b.get("tokens_used", 0)
            st.session_state.chat.append({
                "role": "user", "content": question,
            })
            st.session_state.chat.append({
                "role": "compare",
                "answer_a": result_a["answer"], "strategy_a": strategy,
                "answer_b": result_b["answer"], "strategy_b": strategy_b,
                "tokens": result_a.get("tokens_used", 0) + result_b.get("tokens_used", 0),
            })

        else:
            # ── STANDARD Q&A with LangGraph ───────────────────
            with st.spinner("🔀 LangGraph: retrieve → grade → generate…"):
                # Show live agent steps
                step_placeholder = st.empty()
                step_placeholder.markdown(
                    '<div class="chunk-box">⏳ <span class="node-active">Node: retrieve</span> — searching FAISS…</div>',
                    unsafe_allow_html=True
                )

                init_state: AgentState = {
                    "question": question, "retrieved_docs": [], "graded_docs": [],
                    "answer": "", "tokens_used": 0,
                    "chat_history": st.session_state.memory.get_window_as_text(),
                    "model": model, "temperature": temperature, "top_k": top_k,
                }

                step_placeholder.markdown(
                    '<div class="chunk-box">⏳ <span class="node-active">Node: grade</span> — filtering by relevance…</div>',
                    unsafe_allow_html=True
                )
                result = st.session_state.graph.invoke(init_state)
                step_placeholder.empty()

            st.session_state.memory.add_turn(
                question, result["answer"],
                meta={
                    "pages": [d["page"] for d in result.get("graded_docs", [])],
                    "chunks": [d["chunk_id"] for d in result.get("graded_docs", [])],
                    "tokens": result.get("tokens_used", 0),
                    "strategy": strategy,
                }
            )
            st.session_state.total_tokens += result.get("tokens_used", 0)
            st.session_state.chat.append({"role": "user", "content": question})
            st.session_state.chat.append({
                "role": "assistant", "content": result["answer"],
                "pages":  [d["page"] for d in result.get("graded_docs", [])],
                "chunks": [d["chunk_id"] for d in result.get("graded_docs", [])],
                "tokens": result.get("tokens_used", 0),
                "graded": len(result.get("graded_docs", [])),
                "retrieved": len(result.get("retrieved_docs", [])),
            })

    # ── CHAT HISTORY ──────────────────────────────────────────
    if st.session_state.chat:
        st.markdown("---")
        # Group into (question, answer) pairs then reverse pairs so newest is on top,
        # but within each pair question always renders before answer.
        _pairs, _i = [], 0
        while _i < len(st.session_state.chat):
            _q = st.session_state.chat[_i]
            _a = st.session_state.chat[_i + 1] if _i + 1 < len(st.session_state.chat) else None
            _pairs.append((_q, _a))
            _i += 2

        for _q, _a in reversed(_pairs):
            st.markdown(f'<div class="chat-q">🧑 <strong>You</strong><br>{_q["content"]}</div>', unsafe_allow_html=True)
            if _a is None:
                continue
            if _a["role"] == "compare":
                ca2, cb2 = st.columns(2)
                with ca2:
                    st.markdown(f'<div class="compare-a"><div style="color:#58a6ff;font-weight:600;font-size:.8rem">{_a["strategy_a"].upper()}</div>{_a["answer_a"]}</div>', unsafe_allow_html=True)
                with cb2:
                    st.markdown(f'<div class="compare-b"><div style="color:#a78bfa;font-weight:600;font-size:.8rem">{_a["strategy_b"].upper()}</div>{_a["answer_b"]}</div>', unsafe_allow_html=True)
            else:
                pages_str  = ", ".join(f"p.{p}" for p in _a.get("pages", []))
                chunks_str = ", ".join(f"#{c}" for c in _a.get("chunks", []))
                g, r = _a.get("graded", "?"), _a.get("retrieved", "?")
                st.markdown(
                    f'<div class="chat-a">🤖 <strong>AI ({model})</strong><br>{_a["content"]}'
                    f'<div class="chat-meta">'
                    f'📖 Pages: <strong>{pages_str or "—"}</strong> &nbsp;|&nbsp; '
                    f'🧩 Chunks: {chunks_str or "—"} &nbsp;|&nbsp; '
                    f'✅ Graded: {g}/{r} &nbsp;|&nbsp; '
                    f'🔤 ~{_a.get("tokens", 0):,} tokens'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

        if st.button("🗑️ Clear chat"):
            st.session_state.chat = []
            if st.session_state.memory:
                st.session_state.memory.clear()
            st.rerun()

    # ── COMPARISON HISTORY ────────────────────────────────────
    past_comps = load_comparisons(st.session_state.sid, DATA_DIR / "memory.db")
    if past_comps:
        with st.expander(f"📊 Comparison History ({len(past_comps)} runs)", expanded=False):
            for comp in past_comps[:5]:
                st.markdown(f"**Q:** {comp['question']}")
                cc1, cc2 = st.columns(2)
                cc1.markdown(f"**{comp['strategy_a'].upper()}**\n\n{comp['answer_a'][:300]}…")
                cc2.markdown(f"**{comp['strategy_b'].upper()}**\n\n{comp['answer_b'][:300]}…")
                st.divider()

else:
    st.markdown("""
    <div style="text-align:center;padding:3.5rem;color:#8b949e">
      <div style="font-size:3.5rem">📄</div>
      <div style="font-size:1.1rem;font-weight:600;color:#c9d1d9;margin-top:1rem">Upload a PDF above to begin</div>
      <div style="font-size:.85rem;margin-top:.5rem">LangGraph agents will ingest · chunk · embed · index · answer</div>
    </div>""", unsafe_allow_html=True)
