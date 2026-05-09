# app.py
"""
Repo-Insight: Unified Coding Agent Interface.
Browser-first, chat-driven, with on-demand graph visualization.
"""

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import falkordb
import time
import json
import openai
from pathlib import Path
from collections import Counter

from config import (FALKORDB_HOST, FALKORDB_PORT, GRAPH_NAME,
                    SGLANG_BASE_URL, SGLANG_API_KEY, LLM_MODEL,
                    BASELINE_SGLANG_BASE_URL, BASELINE_LLM_MODEL)
from ingest import run_ingestion, get_connection
from agent import run_repo_agent
from sandbox import SandboxManager
from apply_changes import apply_to_original
from tools import get_macro_architecture, get_class_architecture
from watcher import start_watcher

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Repo-Insight",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark theme polish
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Dark theme accents */
    .stApp { background-color: #0e1117; }
    
    /* Sidebar branding */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1923 0%, #0e1117 100%);
        border-right: 1px solid #1e293b;
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
        border: 1px solid #1e293b;
        margin-bottom: 0.5rem;
    }
    
    /* Primary button gradient */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 100%);
        border: none;
        color: white;
        font-weight: 600;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: #1e293b;
        border-radius: 10px;
        padding: 12px;
        border: 1px solid #334155;
    }
    
    /* Success/error/warning boxes */
    .stAlert {
        border-radius: 10px;
    }
    
    /* Fair fight comparison cards */
    .fair-fight-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph_ready" not in st.session_state:
    st.session_state.graph_ready = False
if "repo_path" not in st.session_state:
    st.session_state.repo_path = ""
if "ingestion_report" not in st.session_state:
    st.session_state.ingestion_report = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_subgraph_data" not in st.session_state:
    st.session_state.last_subgraph_data = None
if "agent_mode" not in st.session_state:
    st.session_state.agent_mode = "c"
if "sandbox_manager" not in st.session_state:
    st.session_state.sandbox_manager = None
if "sandbox_path" not in st.session_state:
    st.session_state.sandbox_path = None
if "original_path" not in st.session_state:
    st.session_state.original_path = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "chat"
if "graph_watcher" not in st.session_state:
    st.session_state.graph_watcher = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def get_db_graph():
    db = falkordb.FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
    return db.select_graph(GRAPH_NAME)


def get_edge_color(rel_type):
    colors = {
        "CALLS": "#3b82f6",
        "INHERITS_FROM": "#8b5cf6",
        "IMPORTS": "#22c55e",
        "DEFINED_IN": "#f59e0b",
    }
    return colors.get(rel_type, "#6b7280")


def build_subgraph_viz(result):
    """Build agraph nodes/edges from a Mode C ChangeResult's subgraph."""
    if result is None or result.subgraph is None:
        return [], []

    sg = result.subgraph
    nodes = []
    edges = []
    seen_nodes = set()

    # Seed nodes — red
    for fqn in sg.seed_nodes:
        if fqn not in seen_nodes:
            nodes.append(Node(
                id=fqn, label=fqn.split(".")[-1],
                color="#ef4444", size=30,
                title=f"SEED: {fqn}",
            ))
            seen_nodes.add(fqn)

    # Blast radius — orange
    for node in sg.blast_radius_nodes:
        fqn = node.get("fqn", "")
        if fqn and fqn not in seen_nodes:
            nodes.append(Node(
                id=fqn, label=fqn.split(".")[-1],
                color="#f97316", size=22,
                title=f"BLAST RADIUS: {fqn}\nFile: {node.get('file_path', '?')}",
            ))
            seen_nodes.add(fqn)
        # Edge from blast node to seed
        for seed in sg.seed_nodes:
            edges.append(Edge(
                source=fqn, target=seed,
                color="#f97316", width=2,
                label="depends on",
            ))

    # Impact radius — blue
    for node in sg.impact_radius_nodes:
        fqn = node.get("fqn", "")
        if fqn and fqn not in seen_nodes:
            nodes.append(Node(
                id=fqn, label=fqn.split(".")[-1],
                color="#3b82f6", size=18,
                title=f"IMPACT: {fqn}\nFile: {node.get('file_path', '?')}",
            ))
            seen_nodes.add(fqn)
        for seed in sg.seed_nodes:
            edges.append(Edge(
                source=seed, target=fqn,
                color="#3b82f6", width=1,
                label="calls",
            ))

    # Callers — cyan
    for node in sg.caller_nodes:
        fqn = node.get("fqn", "")
        if fqn and fqn not in seen_nodes:
            nodes.append(Node(
                id=fqn, label=fqn.split(".")[-1],
                color="#06b6d4", size=16,
                title=f"CALLER: {fqn}",
            ))
            seen_nodes.add(fqn)

    # Callees — green
    for node in sg.callee_nodes:
        fqn = node.get("fqn", "")
        if fqn and fqn not in seen_nodes:
            nodes.append(Node(
                id=fqn, label=fqn.split(".")[-1],
                color="#22c55e", size=16,
                title=f"CALLEE: {fqn}",
            ))
            seen_nodes.add(fqn)

    return nodes, edges


def build_macro_viz(graph, min_weight=1, max_nodes=100):
    """Build full macro architecture graph (Highway Model thick edges)."""
    data = get_macro_architecture(graph)

    node_degrees = Counter()
    for e in data.get("modules", []):
        if e["weight"] >= min_weight:
            node_degrees[e["source"]] += 1
            node_degrees[e["target"]] += 1

    top_nodes = set(n for n, _ in node_degrees.most_common(max_nodes))

    nodes = []
    edges = []
    modules = set()

    for e in data.get("modules", []):
        if e["weight"] < min_weight:
            continue
        if e["source"] not in top_nodes or e["target"] not in top_nodes:
            continue

        modules.add(e["source"])
        modules.add(e["target"])

        label = " & ".join(e["types"])
        size = min(max(e["weight"] / 2, 1), 10)
        color = get_edge_color(e["types"][0] if e["types"] else "")
        edges.append(Edge(
            source=e["source"], target=e["target"],
            label=label, width=size, color=color,
        ))

    for m in modules:
        deg = min(node_degrees[m], 15)
        nodes.append(Node(
            id=m, label=m,
            color="#ef4444", size=18 + deg,
        ))

    return nodes, edges

def handle_engine_event(event_type: str, message: str):
    import streamlit as st
    # Render a highly visible warning inside the current st.status container
    st.warning(f"🤖 **Engine Intervention:** {message}", icon="⚡")
# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🔍 Repo-Insight")
    st.caption("Graph-Driven Coding Agent")
    st.divider()

    # --- Codebase Setup ---
    st.markdown("### 📁 Codebase")
    repo_path = st.text_input(
        "Repository path",
        value=st.session_state.repo_path or "./",
        placeholder="/path/to/your/repo",
    )
    st.session_state.repo_path = repo_path

    if st.button("🚀 Load Codebase", use_container_width=True, type="primary"):
        try:
            manager = SandboxManager(repo_path)
            progress_bar = st.progress(0, text="Creating sandbox...")
            sandbox_path = manager.create()
            st.session_state.sandbox_manager = manager
            st.session_state.sandbox_path = sandbox_path
            st.session_state.original_path = Path(repo_path).resolve()
            progress_bar.progress(20, text="Parsing files...")
            def _on_progress(stage, current, total, detail=""):
                pct = 20 + int(current / max(total, 1) * 75)
                progress_bar.progress(pct, text=f"{stage}... {detail[:50]}")
            report = run_ingestion(str(sandbox_path), on_progress=_on_progress)
            progress_bar.progress(100, text="✅ Graph ready")
            st.session_state.ingestion_report = report
            st.session_state.graph_ready = True
            if st.session_state.graph_watcher:
                st.session_state.graph_watcher.stop()
            st.session_state.graph_watcher = start_watcher(st.session_state.original_path, get_db_graph())
            st.success(
                f"✅ Sandbox [{manager.sandbox_id}]: "
                f"{report.get('functions',0)} functions · "
                f"{report.get('classes',0)} classes · "
                f"{report.get('call_edges',0)} edges"
            )
            st.caption(f"📁 `{sandbox_path}`")
        except Exception as e:
            st.error(f"Ingestion failed: {e}")

    if st.session_state.ingestion_report:
        r = st.session_state.ingestion_report
        st.caption(
            f"📊 {r.get('functions', 0)} funcs · "
            f"{r.get('classes', 0)} classes · "
            f"{r.get('call_edges', 0)} edges · "
            f"{r.get('files_parsed', 0)} files"
        )

    st.divider()

    # --- Agent Mode ---
    st.markdown("### 🔧 Agent Mode")
    mode = st.radio(
        "Select pipeline",
        options=["c", "b"],
        format_func=lambda x: {
            "c": "⚡ Mode C — Graph-Driven (6-Phase)",
            "b": "🔧 Mode B — Tool-Calling Agent",
        }[x],
        index=0,
        label_visibility="collapsed",
    )
    st.session_state.agent_mode = mode

    if mode == "c":
        st.caption("Deterministic graph traversal → validated plan → auto-apply + test")
    elif mode == "b":
        st.caption("ReAct loop with graph tools (non-deterministic)")

    st.divider()

    # --- Demo Prompts ---
    st.markdown("### 📖 Try a Demo")
    DEMO_PROMPTS = [
        "Add a `decorators` field to FunctionDef and update all code that creates or reads it.",
        "Rename `get_connection` to `connect_to_graph` everywhere.",
        "Add error handling to `parse_file` for syntax errors.",
        "Add a `timeout` param to `run_ingestion` and propagate it.",
    ]
    for i, prompt in enumerate(DEMO_PROMPTS, 1):
        if st.button(f"{i}. {prompt[:50]}...", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

    st.divider()

    # --- Quick Actions ---
    st.markdown("### ⚙️ Tools")
    if st.button("📊 Graph Health", use_container_width=True):
        try:
            from graph_health import get_graph_health
            health = get_graph_health(get_db_graph())
            st.json(health)
        except Exception as e:
            st.error(str(e))

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.session_state.last_subgraph_data = None
        st.rerun()

    st.divider()
    if "graph_watcher" in st.session_state:
        st.sidebar.success("● Live graph sync active")
    else:
        st.sidebar.info("○ No active repo — ingest to start live sync")


# ---------------------------------------------------------------------------
# Main area — Tabs: Chat | Graph
# ---------------------------------------------------------------------------

tab_chat, tab_graph, tab_bench = st.tabs([
    "💬 Agent Chat", "🔗 Graph Explorer", "📊 Benchmark"
])


# ===========================
# TAB 1: Chat
# ===========================
with tab_chat:
    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # If this message has attached result metadata, show expandables
            if msg.get("phases"):
                with st.expander("📋 Pipeline Phases"):
                    for phase in msg["phases"]:
                        st.markdown(f"✓ {phase}")

            if msg.get("edit_blocks"):
                with st.expander(f"✏️ Edit Blocks ({msg['edit_count']} edits)"):
                    st.code(msg["edit_blocks"], language="diff")

            if msg.get("test_info"):
                with st.expander("🧪 Test Results"):
                    st.markdown(msg["test_info"])

            if msg.get("timings"):
                with st.expander("⏱️ Timing Breakdown"):
                    st.json(msg["timings"])

    # Chat input
    if prompt := st.chat_input(
        "Ask about your codebase..." if st.session_state.graph_ready
        else "Load a codebase first (sidebar) →"
    ):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Run agent
        with st.chat_message("assistant"):
            if not st.session_state.graph_ready:
                response_text = ("⚠️ Please load a codebase first using the sidebar. "
                                 "Set the repo path and click **Load Codebase**.")
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

            elif st.session_state.agent_mode == "c":
                # --- Mode C: Graph-Driven Engine ---
                from change_engine import GraphDrivenEngine

                phase_status = st.status("🚀 Running 6-Phase Graph-Driven Pipeline...", expanded=True)
                phase_log = []

                def _on_phase(phase, data):
                    labels = {
                        "phase_0": "Phase 0: Graph Construction",
                        "phase_1": "Phase 1: Seed Localization",
                        "phase_2": "Phase 2: Structural Expansion",
                        "phase_3": "Phase 3: Graph-Constrained Planning",
                        "phase_4": "Phase 4: Surgical Editing",
                        "phase_5": "Phase 5: Verified Apply",
                        "phase_5_test": "Phase 5: Test Execution",
                    }
                    label = labels.get(phase, phase)
                    detail = ""
                    if phase == "phase_0":
                        detail = f"{data.get('functions', '?')} functions, {data.get('call_edges', '?')} edges"
                    elif phase == "phase_1":
                        seeds = data.get('seeds', [])
                        detail = f"{len(seeds)} seeds: {', '.join(s.split('.')[-1] for s in seeds[:3])}"
                    elif phase == "phase_2":
                        detail = f"{data.get('blast_radius_count', '?')} blast nodes across {data.get('files_affected', '?')} files"
                    elif phase == "phase_3":
                        v = data.get("is_validated", False)
                        m = data.get("missing_files", 0)
                        detail = f"{'✅ Validated' if v else f'⚠️ {m} missing files'}"
                    elif phase == "phase_4":
                        detail = f"{data.get('edit_blocks', 0)} SEARCH/REPLACE blocks"
                    elif phase == "phase_5":
                        detail = f"apply={'✅' if data.get('apply_success') else '❌'} tests={'✅' if data.get('tests_passed') else '❌'}"
                    elif phase == "phase_5_test":
                        detail = f"attempt {data.get('attempt')}: {data.get('passed', 0)}P / {data.get('failed', 0)}F"
                    elif phase == "self_correction":
                        missing = data.get("missing_files", [])
                        phase_status.write(
                            f"🔄 **Self-Correction** — LLM missed "
                            f"{len(missing)} file(s): "
                            f"{', '.join(f'`{f}`' for f in missing[:3])}\n"
                            f"Graph is recovering them now..."
                        )

                    phase_log.append(f"✓ **{label}** — {detail}")
                    phase_status.update(label=f"⏳ {label}...")
                    phase_status.write(f"✓ {label} — {detail}")

                try:
                    graph = get_db_graph()
                    sandbox_path = st.session_state.sandbox_path
                    original_path = st.session_state.original_path or Path(repo_path).resolve()
                    def handle_engine_event(event_type: str, message: str):
                        # Write to the expanding status container
                        phase_status.write(f"⚡ **Engine Intervention:** {message}")
                        # Also flash a warning on the main UI
                        st.warning(f"🤖 **Engine Intervention:** {message}", icon="⚡")

                    engine = GraphDrivenEngine(
                        original_path, graph,
                        sandbox_path=sandbox_path if sandbox_path != original_path else None
                    )
                    start = time.time()
                    
                    # 2. Pass BOTH callbacks to the engine
                    result = engine.run(
                        prompt, 
                        on_phase=_on_phase,
                        on_event=handle_engine_event
                    )
                    elapsed = time.time() - start

                    phase_status.update(label=f"✅ Pipeline complete ({elapsed:.1f}s)", state="complete")

                    # Store for graph tab
                    st.session_state.last_result = result
                    if result.subgraph:
                        st.session_state.last_subgraph_data = result

                    # Main answer
                    answer = result.answer or "(No answer generated)"
                    st.markdown(answer)

                    if result.timings:
                        st.subheader("⏱ Phase Timings")
                        timing_rows = []
                        for phase_key, secs in result.timings.items():
                            label = phase_key.replace("phase_", "Phase ").replace("_", " ").title()
                            timing_rows.append({"Phase": label, "Time": f"{secs:.2f}s"})
                        st.table(timing_rows)

                    if result.post_edit_analysis and result.post_edit_analysis.get("status") == "analyzed":
                        analysis = result.post_edit_analysis
                        st.subheader("🔗 Graph Delta (Post-Edit)")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Files Re-ingested", analysis.get("files_reingested", 0))
                        col2.metric("New Call Edges", analysis.get("new_edge_count", 0), 
                                    delta=analysis.get("new_edge_count", 0))
                        col3.metric("Removed Edges", analysis.get("removed_edge_count", 0),
                                    delta=-analysis.get("removed_edge_count", 0))
                        
                        if analysis.get("new_call_edges"):
                            with st.expander("New call edges introduced"):
                                for edge in analysis["new_call_edges"][:20]:
                                    st.code(f"{edge['from']}  →  {edge['to']}")
                        
                        if analysis.get("removed_call_edges"):
                            with st.expander("Call edges removed"):
                                for edge in analysis["removed_call_edges"][:20]:
                                    st.code(f"{edge['from']}  →  {edge['to']}")

                    # Build assistant message with metadata
                    msg_data = {
                        "role": "assistant",
                        "content": answer,
                        "phases": result.phases_completed,
                        "timings": result.timings,
                    }

                    # Edit blocks
                    if result.edits:
                        edit_text = "\n\n".join(
                            f"FILE: {eb.file_path}\n<<<<<<< SEARCH\n{eb.search_text[:300]}\n=======\n{eb.replace_text[:300]}\n>>>>>>> REPLACE"
                            for eb in result.edits[:10]
                        )
                        msg_data["edit_blocks"] = edit_text
                        msg_data["edit_count"] = len(result.edits)

                        with st.expander(f"✏️ {len(result.edits)} Edit Blocks"):
                            st.code(edit_text, language="diff")

                    # Test results
                    if result.test_result:
                        tr = result.test_result
                        status_icon = "✅" if tr.all_passed else "❌"
                        test_info = f"{status_icon} **{tr.passed}** passed, **{tr.failed}** failed, **{tr.errors}** errors"
                        msg_data["test_info"] = test_info
                        with st.expander("🧪 Test Results"):
                            st.markdown(test_info)
                            if tr.stdout:
                                st.code(tr.stdout[-1500:], language="text")

                    # Validation gate
                    if result.plan:
                        with st.expander("🛡️ Validation Gate"):
                            st.markdown(f"**Blast radius files:** {len(result.plan.blast_radius_files)}")
                            st.markdown(f"**Planned files:** {len(result.plan.planned_files)}")
                            st.markdown(f"**Missing files:** {len(result.plan.missing_files)}")
                            st.markdown(f"**Validated:** {'✅ Yes' if result.plan.is_validated else '❌ No'}")
                            if result.plan.justifications:
                                st.json(result.plan.justifications)

                    # Timings
                    if result.timings:
                        with st.expander("⏱️ Timing"):
                            st.json(result.timings)



                    # Nudge to graph tab
                    if result.subgraph and result.subgraph.blast_radius_nodes:
                        st.info("💡 Switch to the **Graph Explorer** tab to see the blast radius visualization.")

                    if st.session_state.sandbox_manager:
                        st.divider()
                        with st.expander("🗂️ Sandbox Review", expanded=True):
                            mgr = st.session_state.sandbox_manager
                            changed = mgr.get_changed_files()
                            if changed:
                                st.markdown(f"**{len(changed)} file(s) modified in sandbox:**")
                                for f in changed:
                                    added = f.get('lines_added', 0)
                                    removed = f.get('lines_removed', 0)
                                    st.markdown(
                                        f"✏️ `{f['path']}` "
                                        f"<span style='color:#4ade80'>+{added}</span> "
                                        f"<span style='color:#f87171'>-{removed}</span>",
                                        unsafe_allow_html=True
                                    )
                            else:
                                st.info("No file changes yet.")
                            diff = mgr.get_diff()
                            if diff:
                                with st.expander("📄 Unified Diff", expanded=False):
                                    st.code(diff[:6000], language="diff")
                            st.divider()
                            st.caption(
                                f"💡 Run your own tests: `cd {st.session_state.sandbox_path} && pytest`"
                            )
                            ca, cd = st.columns(2)
                            with ca:
                                if st.button("✅ Apply to Real Repo", type="primary", use_container_width=True):
                                    res = apply_to_original(
                                        st.session_state.sandbox_path,
                                        st.session_state.original_path,
                                    )
                                    if res.success:
                                        st.success(f"Applied {len(res.files_applied)} file(s)")
                                        for f in res.files_applied:
                                            st.caption(f"  ✓ {f}")
                                    else:
                                        st.error(res.error or "Apply failed")
                            with cd:
                                if st.button("🗑️ Discard Sandbox", use_container_width=True):
                                    mgr.discard()
                                    st.session_state.sandbox_manager = None
                                    st.session_state.sandbox_path = None
                                    st.session_state.original_path = None
                                    st.session_state.graph_ready = False
                                    st.warning("Sandbox discarded.")
                                    st.rerun()

                    if result.error:
                        st.warning(f"Pipeline error: {result.error}")

                    st.session_state.messages.append(msg_data)

                except Exception as e:
                    error_msg = f"❌ Pipeline error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

            elif st.session_state.agent_mode == "fair":
                # --- Fair Fight: Graph-Driven vs Blind Baseline ---
                from change_engine import GraphDrivenEngine

                st.markdown("### ⚔️ Fair Fight: Structure vs. Scale")
                col_graph, col_blind = st.columns(2)

                # --- Left column: Graph-Driven (Mode C) ---
                with col_graph:
                    st.markdown(f"#### 🧠 {LLM_MODEL} + Graph")
                    phase_status = st.status("Running Graph Pipeline...", expanded=True)
                    phase_log_fair = []

                    def _on_phase_fair(phase, data):
                        labels = {
                            "phase_0": "Graph Construction",
                            "phase_1": "Seed Localization",
                            "phase_2": "Structural Expansion",
                            "phase_3": "Validated Planning",
                            "phase_4": "Surgical Editing",
                            "phase_5": "Verified Apply",
                        }
                        label = labels.get(phase, phase)
                        phase_log_fair.append(label)
                        phase_status.update(label=f"⏳ {label}...")
                        phase_status.write(f"✓ {label}")

                    try:
                        graph = get_db_graph()
                        sb_path = st.session_state.sandbox_path
                        orig_path = st.session_state.original_path or Path(repo_path).resolve()
                        engine = GraphDrivenEngine(
                            orig_path, graph,
                            sandbox_path=sb_path if sb_path != orig_path else None
                        )
                        start_c = time.time()
                        result_c = engine.run(prompt, on_phase=_on_phase_fair)
                        elapsed_c = time.time() - start_c
                        phase_status.update(label=f"✅ Done ({elapsed_c:.1f}s)", state="complete")

                        st.session_state.last_result = result_c
                        if result_c.subgraph:
                            st.session_state.last_subgraph_data = result_c

                        answer_c = result_c.answer or "(No answer)"
                        st.markdown(answer_c[:2000])

                        if result_c.edits:
                            st.metric("Edit Blocks", len(result_c.edits))
                        if result_c.plan:
                            st.metric("Validated", "✅ Yes" if result_c.plan.is_validated else "❌ No")
                    except Exception as e:
                        answer_c = f"Error: {e}"
                        elapsed_c = 0
                        result_c = None
                        st.error(answer_c)

                # --- Right column: Blind Baseline ---
                with col_blind:
                    st.markdown(f"#### 🤖 {BASELINE_LLM_MODEL} (Blind)")
                    try:
                        baseline_client = openai.OpenAI(
                            base_url=BASELINE_SGLANG_BASE_URL,
                            api_key=SGLANG_API_KEY,
                        )
                        with st.spinner(f"Querying {BASELINE_LLM_MODEL}..."):
                            start_b = time.time()
                            response = baseline_client.chat.completions.create(
                                model=BASELINE_LLM_MODEL,
                                messages=[
                                    {"role": "system", "content": (
                                        "You are a helpful coding assistant. When asked to make code changes, "
                                        "list every file that needs modification. Be thorough."
                                    )},
                                    {"role": "user", "content": prompt},
                                ],
                            )
                            elapsed_b = time.time() - start_b

                        answer_b = response.choices[0].message.content or "(empty)"
                        st.success(f"Done ({elapsed_b:.1f}s)")
                        st.markdown(answer_b[:2000])
                        st.metric("Graph Tools Used", "0")
                        st.metric("Validation Gate", "❌ None")
                    except Exception as e:
                        answer_b = f"Error: {e}"
                        elapsed_b = 0
                        st.error(answer_b)

                # --- Comparison ---
                st.divider()
                st.markdown("### 📊 Head-to-Head Results")
                m1, m2, m3 = st.columns(3)
                with m1:
                    graph_edits = len(result_c.edits) if result_c and result_c.edits else 0
                    st.metric("Graph Pipeline Edits", graph_edits, delta="surgical")
                with m2:
                    validated = result_c.plan.is_validated if result_c and result_c.plan else False
                    st.metric("Validation Gate", "✅ Passed" if validated else "❌ Failed")
                with m3:
                    st.metric("Baseline Has Graph", "❌ No", delta="blind guess")

                st.info(
                    f"💡 **{LLM_MODEL}** with the graph pipeline produced "
                    f"**{graph_edits}** surgical edit blocks with structural validation. "
                    f"**{BASELINE_LLM_MODEL}** provided prose guidance without any structural guarantee. "
                    f"Run `python demo_cli.py --score` for quantitative Precision/Recall/F1 proof."
                )

                msg_data = {
                    "role": "assistant",
                    "content": f"**⚔️ Fair Fight Complete**\n\n"
                               f"- Graph-Driven ({LLM_MODEL}): {elapsed_c:.1f}s, {graph_edits} edits\n"
                               f"- Blind Baseline ({BASELINE_LLM_MODEL}): {elapsed_b:.1f}s, prose only\n",
                }
                st.session_state.messages.append(msg_data)

            else:
                # --- Mode B: Tool-Calling Agent ---
                try:
                    graph = get_db_graph()
                    tool_log_container = st.expander("🔧 Graph Queries (Live)", expanded=True)
                    tool_entries = []

                    with st.spinner("Graph-RAG Agent thinking..."):
                        start = time.time()
                        result = run_repo_agent(prompt, graph)
                        elapsed = time.time() - start

                    answer = result.get("answer", "")
                    st.markdown(answer)

                    with st.expander(f"🔧 {len(result.get('tool_calls_log', []))} Tool Calls ({elapsed:.1f}s)"):
                        for entry in result.get("tool_calls_log", []):
                            st.markdown(f"**{entry['tool']}**({json.dumps(entry['args'])})")

                    if result.get("diff"):
                        with st.expander("📝 Change Set"):
                            st.code(result["diff"], language="diff")

                    msg_data = {"role": "assistant", "content": answer}
                    st.session_state.messages.append(msg_data)

                except Exception as e:
                    error_msg = f"❌ Agent error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


# ===========================
# TAB 2: Graph Explorer (75%+ screen)
# ===========================
with tab_graph:
    st.markdown("## 🔗 Dependency Graph Explorer")

    # Graph type selector
    graph_type = st.radio(
        "View",
        options=["blast_radius", "macro", "class"],
        format_func=lambda x: {
            "blast_radius": "🎯 Blast Radius (from last query)",
            "macro": "🏗️ Macro Architecture (modules)",
            "class": "🔬 Class Architecture (within module)",
        }[x],
        horizontal=True,
    )

    # Legend
    with st.expander("📖 Legend", expanded=False):
        cols = st.columns(5)
        cols[0].markdown("🔴 **Seed** (change target)")
        cols[1].markdown("🟠 **Blast Radius** (will break)")
        cols[2].markdown("🔵 **Impact** (downstream)")
        cols[3].markdown("🔷 **Caller** (direct)")
        cols[4].markdown("🟢 **Callee** (direct)")

    # Graph rendering area — full width, large height
    if graph_type == "blast_radius":
        if st.session_state.last_subgraph_data:
            result = st.session_state.last_subgraph_data
            nodes, edges = build_subgraph_viz(result)

            if nodes:
                st.caption(
                    f"Showing {len(nodes)} nodes, {len(edges)} edges · "
                    f"Seeds: {', '.join(result.subgraph.seed_nodes)} · "
                    f"Files affected: {len(result.subgraph.all_affected_files)}"
                )

                config = Config(
                    width=1200,
                    height=700,
                    directed=True,
                    nodeHighlightBehavior=True,
                    highlightColor="#fbbf24",
                    collapsible=False,
                    physics=True,
                    hierarchical=False,
                )
                agraph(nodes=nodes, edges=edges, config=config)

                # Files affected list below graph
                with st.expander("📂 All Affected Files"):
                    for fp in sorted(result.subgraph.all_affected_files):
                        st.markdown(f"- `{fp}`")
            else:
                st.info("No subgraph data to display. Run a query in the Chat tab first.")
        else:
            st.info("No blast radius data yet. Ask a code change question in the **Chat** tab, then come back here.")

    elif graph_type == "macro":
        try:
            graph = get_db_graph()

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                min_weight = st.slider("Min edge weight", 1, 50, 3)
            with col_s2:
                max_nodes = st.slider("Max nodes", 10, 200, 80)

            nodes, edges = build_macro_viz(graph, min_weight, max_nodes)

            if nodes:
                st.caption(f"Showing {len(nodes)} modules, {len(edges)} connections")
                config = Config(
                    width=1200,
                    height=700,
                    directed=True,
                    nodeHighlightBehavior=True,
                    highlightColor="#fbbf24",
                    collapsible=False,
                    physics=False,
                )
                agraph(nodes=nodes, edges=edges, config=config)
            else:
                st.info("No edges match the current filter. Lower the min weight.")
        except Exception as e:
            st.error(f"Could not load graph: {e}")

    elif graph_type == "class":
        mod_name = st.text_input("Module name", placeholder="e.g. parser, ingest, agent")
        if mod_name:
            try:
                graph = get_db_graph()
                data = get_class_architecture(mod_name, graph)

                node_degrees = Counter()
                for e in data.get("class_edges", []):
                    node_degrees[e["source"]] += 1
                    node_degrees[e["target"]] += 1

                nodes = []
                edges = []
                classes = set()

                for e in data.get("class_edges", []):
                    classes.add(e["source"])
                    classes.add(e["target"])
                    label = " & ".join(e["types"])
                    size = min(max(e["weight"] / 2, 1), 10)
                    color = get_edge_color(e["types"][0] if e["types"] else "")
                    edges.append(Edge(
                        source=e["source"], target=e["target"],
                        label=label, width=size, color=color,
                    ))

                for c in classes:
                    deg = min(node_degrees.get(c, 0), 15)
                    nodes.append(Node(id=c, label=c, color="#818cf8", size=18 + deg))

                if nodes:
                    config = Config(
                        width=1200,
                        height=700,
                        directed=True,
                        nodeHighlightBehavior=True,
                        highlightColor="#fbbf24",
                        collapsible=False,
                        physics=True,
                    )
                    agraph(nodes=nodes, edges=edges, config=config)
                else:
                    st.info(f"No class relationships found in module '{mod_name}'.")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
                    st.info("Enter a module name to explore its class architecture.")


# ===========================
# TAB 3: Benchmark
# ===========================
with tab_bench:
    st.markdown("## 📊 Precision / Recall / F1 Benchmark")
    st.caption(
        "How precisely does the graph-guided agent identify the correct files vs. "
        "a blind LLM? Uses the internal scoring harness against the loaded codebase."
    )

    if not st.session_state.graph_ready:
        st.info("Load a codebase first to run the benchmark.")
    else:
        col_cfg, col_res = st.columns([1, 2])
        with col_cfg:
            n_cases = st.number_input("Test cases", min_value=1, max_value=20, value=4)
            if st.button("▶ Run Scoring", type="primary", use_container_width=True):
                with st.spinner("Running scoring harness..."):
                    try:
                        from scoring import run_scoring_suite
                        from change_engine import GraphDrivenEngine
                        GraphDrivenEngine  # keep import reference

                        tasks = __import__("scoring", fromlist=["GROUND_TRUTH_TASKS"]).GROUND_TRUTH_TASKS[:n_cases]
                        scores = run_scoring_suite(
                            graph=get_db_graph(),
                            repo_root=Path(st.session_state.sandbox_path or repo_path),
                            modes=["c", "baseline"],
                            tasks=tasks,
                        )
                        st.session_state.bench_scores = scores
                    except Exception as e:
                        st.error(f"Scoring failed: {e}")

        with col_res:
            if scores_data := st.session_state.get("bench_scores"):
                avgs = scores_data.mode_averages
                mode_c = avgs.get("Mode C", {})
                mode_b = avgs.get("Mode BASELINE", {})

                m1, m2, m3 = st.columns(3)
                m1.metric("Graph-RAG F1", f"{mode_c.get('avg_f1', 0)*100:.1f}%")
                m2.metric("Baseline F1", f"{mode_b.get('avg_f1', 0)*100:.1f}%")
                m3.metric("Improvement", f"+{(mode_c.get('avg_f1',0) - mode_b.get('avg_f1',0))*100:.1f}%")

                metrics = ["Precision", "Recall", "F1"]
                keys = ["avg_precision", "avg_recall", "avg_f1"]
                graph_vals = [mode_c.get(k, 0)*100 for k in keys]
                baseline_vals = [mode_b.get(k, 0)*100 for k in keys]

                import pandas as pd
                df = pd.DataFrame({
                    "Metric": metrics,
                    "Graph-RAG": graph_vals,
                    "Blind Baseline": baseline_vals,
                }).set_index("Metric")
                st.bar_chart(df, y=["Graph-RAG", "Blind Baseline"], horizontal=True)
