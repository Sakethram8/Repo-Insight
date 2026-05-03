# Repo-Insight

## The Problem With RAG-Based Coding Agents

Standard RAG-based coding agents rely heavily on semantic similarity to fetch context. This is fundamentally the wrong primitive for codebase modifications. If you ask an agent to "rename the user authentication function," vector search easily finds the file where the function is defined. However, it completely misses the five disparate files upstream that call that function, because "call site usage" shares little semantic similarity with the functional intent of the change. 

This leads to catastrophic cascade failures during automated refactoring. The LLM acts as a blind surgeon—modifying the focal point perfectly, but leaving a wake of broken imports, un-updated arguments, and broken tests upstream because those files were never placed in its context window. Scale cannot fix this; structural context is required.

## The Solution: Structural Graph Traversal

Repo-Insight abandons blind RAG in favor of a deterministic, structural knowledge graph. We use a two-tier static analysis engine (Tree-sitter for high-speed syntax extraction, Jedi for precise cross-module inference) to ingest the entire codebase into FalkorDB. Before the LLM is ever asked to write code, the engine executes recursive Cypher queries to compute the exact blast radius of the change, ensuring all downstream dependencies and upstream callers are deterministically captured in the context window. 

## How It Works — The 6-Phase Pipeline

1. Phase 0: Graph Construction (fingerprint-based skip-if-fresh)
2. Phase 1: Seed Localization (semantic search → LLM filtering)
3. Phase 2: Structural Expansion (deterministic Cypher traversal, NO LLM)
4. Phase 3: Graph-Constrained Planning (LLM + validation gate, up to 3 rounds)
5. Phase 4: Surgical Editing (SEARCH/REPLACE blocks, fuzzy matching fallback)
6. Phase 5: Verified Apply (sandbox, pytest, post-edit graph re-analysis)

## Architecture

* parser.py → High-speed syntax extraction and AST parsing
* ingest.py → Serializes AST data and edge connections
* FalkorDB → The core graph database
* tools.py → Exposes structural Cypher queries and semantic search
* change_engine.py → Orchestrates the strict 6-phase pipeline
* apply_changes.py → Handles surgical injection of edits
* sandbox.py → Manages isolated test execution
* app.py / mcp_server.py → UI and Model Context Protocol entry points

## Why It's Different

| Feature | Repo-Insight | GitHub Copilot | Cursor | aider |
| :--- | :--- | :--- | :--- | :--- |
| **Change impact detection** | Deterministic Graph Traversal | None (Blind Context) | Semantic/Keyword search | AST-aware grep |
| **Multi-file coordination** | Strict 6-Phase Pipeline | Developer-driven | Developer-driven | Agentic Loop |
| **Post-edit verification** | Sandbox + Pytest Execution | None | None | Test runner execution |
| **MCP integration** | Native `mcp_server.py` | No | Client | No |
| **Graph feedback loop** | Post-edit Graph Delta Re-Analysis | No | No | No |

## Quick Start

```bash
docker-compose up -d
pip install -r requirements.txt
streamlit run app.py
```

## MCP Integration

Repo-Insight natively supports the Model Context Protocol (MCP). You can connect it directly to Claude Desktop or Cursor to expose its graph-traversal tools to your daily editor via `mcp_server.py`. Configure your MCP client to launch the server as a child process using stdio transport.

## Modes

* **Mode B:** Agent analysis + proposed diff (read-only)
* **Mode C:** Full 6-phase automated change pipeline

## Benchmark

SWE-bench Lite results: coming soon
