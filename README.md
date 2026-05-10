<div align="center">
  <a href="https://git.io/typing-svg">
    <img src="https://readme-typing-svg.demolab.com?font=DM+Sans&weight=800&size=42&pause=1000&color=6366F1&center=true&width=600&height=70&lines=REPO-INSIGHT;GRAPH-DRIVEN+CODING+AGENT;AUTONOMOUS+BLAST-RADIUS+ANALYSIS" alt="Typing SVG">
  </a>
  <h3><i>Eliminate RAG cascade failures. Prove change impact deterministically. Edit autonomously.</i></h3>
  <p><b>A Deterministic, Structural Knowledge Graph Agent for Multi-File Refactoring</b></p>

  [![AMD MI300X](https://img.shields.io/badge/AMD-MI300X-ED1C24?style=for-the-badge&logo=amd)](https://www.amd.com/)
  [![ROCm](https://img.shields.io/badge/ROCm-6.x-blue?style=for-the-badge&logo=amd)](https://rocm.docs.amd.com/)
  [![Qwen](https://img.shields.io/badge/Powered%20by-Qwen3-6366F1?style=for-the-badge)](https://huggingface.co/Qwen)
  [![FalkorDB](https://img.shields.io/badge/Graph%20DB-FalkorDB-FF5A00?style=for-the-badge)](https://falkordb.com/)
  [![Tree-sitter](https://img.shields.io/badge/AST-Tree--sitter-10B981?style=for-the-badge)](https://tree-sitter.github.io/)
  [![Model Context Protocol](https://img.shields.io/badge/Protocol-MCP%20Native-8A2BE2?style=for-the-badge)](https://modelcontextprotocol.io/)
  [![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
</div>

---

## 🚨 The Fatal Flaw of Standard RAG Coding Agents

Standard RAG-based coding agents rely heavily on semantic similarity to fetch context. This is fundamentally the wrong primitive for codebase modifications. 

If you ask an agent to *"rename the user authentication function,"* vector search easily finds the file where the function is defined. However, it completely misses the five disparate files upstream that call that function, because **"call site usage"** shares little semantic similarity with the functional intent of the change.
❌ BLIND RAG CASCADE FAILURE:
[User Request: Rename Auth] ──> Semantic Search retrieves auth.py perfectly.
│
├──> Upstream caller A (MISSING FROM CONTEXT) = Crashes at runtime.
├──> Upstream caller B (MISSING FROM CONTEXT) = Un-updated arguments.
└──> Integration tests (MISSING FROM CONTEXT) = Broken pipeline.


The LLM acts as a blind surgeon—modifying the focal point perfectly, but leaving a wake of broken imports, un-updated arguments, and broken tests upstream because those files were never placed in its context window. **Scale cannot fix this; structural context is required.**

---

## ⚡ The Solution: Structural Graph Traversal

**Repo-Insight** abandons blind RAG in favor of a deterministic, structural knowledge graph. We use a two-tier static analysis engine—**Tree-sitter** for high-speed syntax extraction, and **Jedi** for precise cross-module inference—to ingest the entire codebase into **FalkorDB**. 

Before the LLM is ever asked to write code, the engine executes recursive Cypher queries to compute the **exact blast radius** of the change, ensuring all downstream dependencies and upstream callers are deterministically captured in the context window.

---

## ⚙️ The Orchestrated 6-Phase Pipeline

Repo-Insight orchestrates multi-file edits via a highly rigorous, verified loop:

┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 0: INGESTION & GRAPH BUILD                     │
│        Extract AST syntax + verify repository fingerprint cache         │
└─────────────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: SEED LOCALIZATION                           │
│        Semantic vector search mapped to precise entry-point nodes       │
└─────────────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: STRUCTURAL EXPANSION                        │
│        Deterministic Cypher traversal pulls full blast radius (NO LLM)  │
└─────────────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 3: CONSTRAINED PLANNING                        │
│        LLM devises change plan + strict validation gate intercept       │
└─────────────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 4: SURGICAL EDITING                            │
│        Generates SEARCH/REPLACE blocks with fast fuzzy-match fallback   │
└─────────────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 5: VERIFIED APPLY                              │
│        Isolated sandbox execution, pytest harness, and graph diffs      │
└─────────────────────────────────────────────────────────────────────────┘


---

## 🏗️ Core Architecture

The backend is built for extreme performance, strong verification, and zero-friction client connectivity:

* `parser.py` &mdash; High-speed syntax extraction and AST parsing.
* `ingest.py` &mdash; Serializes AST data and builds call/import edge connections.
* **FalkorDB** &mdash; The core high-performance graph database.
* `tools.py` &mdash; Exposes structural Cypher queries and targeted semantic search.
* `change_engine.py` &mdash; Orchestrates the strict, deterministic 6-phase pipeline.
* `apply_changes.py` &mdash; Handles surgical injection of edit blocks.
* `sandbox.py` &mdash; Manages isolated, temporary repository clones for safe test execution.
* `app.py` / `mcp_server.py` &mdash; Interactive Streamlit UI and native Model Context Protocol entry points.

---

## 🥊 The Fair Fight: Why It's Different

| Feature | ⚡ Repo-Insight | GitHub Copilot | Cursor | aider |
| :--- | :--- | :--- | :--- | :--- |
| **Change Impact Detection** | **Deterministic Graph Traversal** | None (Blind Context) | Semantic/Keyword search | AST-aware grep |
| **Multi-File Coordination** | **Strict 6-Phase Pipeline** | Developer-driven | Developer-driven | Agentic Loop |
| **Post-Edit Verification** | **Sandbox + Pytest Execution** | None | None | Test runner execution |
| **MCP Integration** | **Native `mcp_server.py`** | No | Client | No |
| **Graph Feedback Loop** | **Post-edit Graph Delta Analysis** | No | No | No |

---

## 🔌 Native MCP Integration

Repo-Insight natively supports the **Model Context Protocol (MCP)**. You can connect it directly to **Claude Desktop** or **Cursor** to expose its powerful graph-traversal tools directly to your daily editing environment via `mcp_server.py`. 

Simply configure your preferred MCP client to launch the server as a child process using standard input/output (`stdio`) transport.

---

## 🕹️ Interactive Execution Modes

* **Mode B (Advisory):** Agent graph analysis + proposed diff visualization (completely read-only).
* **Mode C (Autonomous):** Full 6-phase automated change pipeline with automated sandbox verification.

---

## 🚀 Quick Start

Spin up the entire stack, populate the graph, and launch the interactive dashboard in three commands:

```bash
# 1. Start the connected FalkorDB graph instance
docker-compose up -d

# 2. Install core dependencies
pip install -r requirements.txt

# 3. Launch the visual analytics interface
streamlit run app.py

📊 Benchmarks

    SWE-bench Lite / Open-Source Performance: Empirical evaluation metrics and quantitative scoring runs loading soon.


***

