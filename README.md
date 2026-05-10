<div align="center">
  <a href="https://git.io/typing-svg">
    <img src="https://readme-typing-svg.demolab.com?font=DM+Sans&weight=800&size=36&pause=1000&color=6366F1&center=true&width=800&height=70&lines=REPO-INSIGHT;GRAPH-DRIVEN+CODING+AGENT;AUTONOMOUS+BLAST-RADIUS+ANALYSIS" alt="Repo-Insight">
  </a>

  <h3><i>Eliminate RAG cascade failures. Prove change impact deterministically. Edit autonomously.</i></h3>
  <p><b>A deterministic, structural knowledge-graph agent for multi-file refactoring.</b></p>

  [![AMD MI300X](https://img.shields.io/badge/AMD-MI300X-ED1C24?style=for-the-badge&logo=amd)](https://www.amd.com/)
  [![ROCm](https://img.shields.io/badge/ROCm-6.x-blue?style=for-the-badge&logo=amd)](https://rocm.docs.amd.com/)
  [![Qwen](https://img.shields.io/badge/Powered%20by-Qwen3-6366F1?style=for-the-badge)](https://huggingface.co/Qwen)
  [![FalkorDB](https://img.shields.io/badge/Graph%20DB-FalkorDB-FF5A00?style=for-the-badge)](https://falkordb.com/)
  [![Tree-sitter](https://img.shields.io/badge/AST-Tree--sitter-10B981?style=for-the-badge)](https://tree-sitter.github.io/)
  [![MCP](https://img.shields.io/badge/Protocol-MCP%20Native-8A2BE2?style=for-the-badge)](https://modelcontextprotocol.io/)
  [![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
</div>

---

## 🚨 The Fatal Flaw of Standard RAG Coding Agents

Standard RAG-based coding agents rely on semantic similarity to fetch context. For codebase modifications, this is the wrong primitive.

Ask an agent to *"rename the user authentication function"* and vector search will find `auth.py` perfectly. It will completely miss the five disparate files upstream that **call** that function — because *call-site usage* shares almost no semantic similarity with the functional intent of the change.

```text
❌  BLIND RAG CASCADE FAILURE

    [User Request: Rename Auth]
              │
              ▼
    Semantic Search → auth.py  ✅  (retrieved)
              │
              ├──▶ Upstream caller A     ❌  missing  →  runtime crash
              ├──▶ Upstream caller B     ❌  missing  →  un-updated args
              └──▶ Integration tests     ❌  missing  →  broken pipeline
```

The LLM acts as a blind surgeon — modifying the focal point perfectly while leaving a wake of broken imports, un-updated arguments, and failing tests upstream because those files were never placed in its context window. **Scale cannot fix this. Structural context is required.**

---

## ⚡ The Solution: Structural Graph Traversal

**Repo-Insight** abandons blind RAG in favor of a deterministic, structural knowledge graph. A two-tier static-analysis engine — **Tree-sitter** for high-speed syntax extraction and **Jedi** for precise cross-module inference — ingests the entire codebase into **FalkorDB**.

Before the LLM is ever asked to write code, the engine executes recursive Cypher queries to compute the **exact blast radius** of the change, ensuring every downstream dependency and upstream caller is deterministically captured in the context window.

---

## ⚙️ The Orchestrated 6-Phase Pipeline

Repo-Insight orchestrates multi-file edits through a rigorous, verified loop:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 0  ·  INGESTION & GRAPH BUILD                                    │
│  Extract AST syntax + verify repository fingerprint cache               │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1  ·  SEED LOCALIZATION                                          │
│  Semantic vector search mapped to precise entry-point nodes             │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 2  ·  STRUCTURAL EXPANSION                                       │
│  Deterministic Cypher traversal pulls full blast radius (no LLM)        │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 3  ·  CONSTRAINED PLANNING                                       │
│  LLM devises change plan + strict validation gate intercept             │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 4  ·  SURGICAL EDITING                                           │
│  Generates SEARCH/REPLACE blocks with fast fuzzy-match fallback         │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 5  ·  VERIFIED APPLY                                             │
│  Isolated sandbox execution, pytest harness, and graph diffs            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Core Architecture

The backend is built for extreme performance, strong verification, and zero-friction client connectivity:

| Module | Responsibility |
| :--- | :--- |
| `parser.py` | High-speed syntax extraction and AST parsing |
| `ingest.py` | Serializes AST data and builds call/import edges |
| **FalkorDB** | High-performance graph database (the source of truth) |
| `tools.py` | Exposes structural Cypher queries and targeted semantic search |
| `change_engine.py` | Orchestrates the strict, deterministic 6-phase pipeline |
| `apply_changes.py` | Handles surgical injection of edit blocks |
| `sandbox.py` | Manages isolated, temporary repository clones for safe test execution |
| `app.py` / `mcp_server.py` | Interactive Streamlit UI and native MCP entry points |

---

## 🥊 The Fair Fight: Why It's Different

| Feature | **Repo-Insight** | GitHub Copilot | Cursor | aider |
| :--- | :--- | :--- | :--- | :--- |
| **Change-impact detection** | Deterministic graph traversal | None (blind context) | Semantic / keyword search | AST-aware grep |
| **Multi-file coordination** | Strict 6-phase pipeline | Developer-driven | Developer-driven | Agentic loop |
| **Post-edit verification** | Sandbox + pytest execution | None | None | Test-runner execution |
| **MCP integration** | Native `mcp_server.py` | No | Client only | No |
| **Graph feedback loop** | Post-edit graph-delta analysis | No | No | No |

---

## 🔌 Native MCP Integration

Repo-Insight natively supports the **Model Context Protocol (MCP)**. Connect it directly to **Claude Desktop** or **Cursor** to expose its graph-traversal tools inside your daily editing environment via `mcp_server.py`.

Configure your preferred MCP client to launch the server as a child process over standard input/output (`stdio`) transport.

---

## 🕹️ Interactive Execution Modes

- **Mode B — Advisory.** Agent graph analysis with a proposed-diff visualization. Completely read-only.
- **Mode C — Autonomous.** Full 6-phase automated change pipeline with sandbox verification.

---

## 🚀 Quick Start

Spin up the stack, populate the graph, and launch the dashboard in three commands:

```bash
# 1. Start the FalkorDB graph instance
docker-compose up -d

# 2. Install core dependencies
pip install -r requirements.txt

# 3. Launch the visual analytics interface
streamlit run app.py
```

---

## 📊 Benchmarks

> SWE-bench Lite and open-source performance evaluations are in progress. Empirical metrics and quantitative scoring runs will be published here.

---

<div align="center">
  <sub>Built for the AMD Developer Hackathon · Powered by AMD MI300X + ROCm</sub>
</div>
