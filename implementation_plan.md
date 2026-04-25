# Phase 1: Architectural Blueprint for Repo-Insight (Hackathon Edition)

This document finalizes the architectural details for transforming Repo-Insight from a standard graph-RAG proof-of-concept into the "Anti-RAG" precision tool designed to win the AMD hackathon. 

This architecture is designed to be built and tested locally on your Dell G15 SE (Phase 2) before scaling to the MI300X on Digital Ocean (Phase 3).

## User Review Required

> [!IMPORTANT]
> Please review this architectural plan. Once approved, we will transition into Phase 2 and begin executing these changes locally on your machine. Let me know if any aspect is missing or if you want to alter the priorities.

## Proposed Architecture Changes

We will restructure the application across four core domains: The Graph Schema (Hierarchical "Highway" Model), The Agent, The Graph Tools, and The UI.

---

### 1. The Graph Schema: The Hybrid Structural Graph
To capture the full data and logic flow of the codebase, we cannot rely solely on function calls. We must physically store the different architectural layers and their unique relationships.

#### [MODIFY] `ingest.py` & `parser.py` (The Full Flow Model)
*   **Physical Nodes:** We will retain explicit `Function`, `Class`, and `Module` nodes.
*   **Diverse Thin/Medium Edges:** The graph will store multiple specific types of connections:
    *   `[:CALLS]` (Function -> Function): The runtime logic flow.
    *   `[:INHERITS_FROM]` (Class -> Class): The structural object-oriented flow.
    *   `[:IMPORTS]` (Module -> Module): The high-level file dependency flow.
    *   `[:DEFINED_IN]` (Function->Class, Class->Module): The containment structure.
*   **Aggregating the Highways (UI Zooming):** 
    *   When the user zooms out to the **Class level**, the UI will draw a "Medium Highway" between Class A and Class B by summing the weights of all underlying `CALLS` between their functions, PLUS an immediate heavy weight if there is an `INHERITS_FROM` edge.
    *   When zooming out to the **Module level**, the "Thick Highway" is the amalgamation of all underlying function `CALLS`, class inheritances, AND direct `IMPORTS`.
*   **Benefit:** The Agent and the user see a flawless, complete picture. If two classes never call each other but share an inheritance tree, or if two files only share constants via an import, the connection is mathematically preserved and visually displayed.

### 2. The Agent: Two-Phase Execution Loop
Currently, `agent.py` allows the LLM to call any tool at any time. We will enforce a strict "Map Before You Drive" architecture.

#### [MODIFY] `agent.py`
*   **Architect Phase (Mapping):** The system prompt will enforce that the agent *must* first map the blast radius using `get_blast_radius`, `get_callers`, and `semantic_search`. It will output a JSON or Markdown plan of the dependency topology.
*   **Surgeon Phase (Editing):** Only after the map is validated internally will the agent be allowed to use `get_source_code` on the specific files identified in the blueprint. 
*   **Benefit:** This entirely prevents the agent from falling into "context stuffing" where it reads too many files and loses track of the goal.

### 2. The Graph Tools: Sub-Graph Serialization & Weighted Search
Currently, tools return flat lists of affected files. We need them to return topological maps and rank importance based on graph metrics.

#### [MODIFY] `tools.py`
*   **Update `get_blast_radius` & `get_impact_radius`:** Instead of `RETURN DISTINCT affected.name`, we will modify the Cypher query to use `paths` (`RETURN p = (affected)-[:CALLS*1..x]->(target)`). We will serialize this path into a minimal JSON tree. The LLM will receive the exact call stack (e.g., `A -> B -> C`), not just a list of names.
*   **Graph-Weighted Semantic Search:** We will update the `semantic_search` Cypher query to also calculate the **In-Degree** (how many functions call this node). The final score will be a combination of `cosine_similarity` + `(log(in_degree) * weight)`. Core utilities will automatically rank higher than random test scripts.

### 3. The Extraction & Ingestion Pipeline: Robustness
The foundation of the graph must be flawlessly accurate to trust the agent.

#### [MODIFY] `parser.py`
*   **Fix Namespace Collisions:** On line 191 (`caller_simple = call.caller_name.split(".")[-1]`), stripping class prefixes merges all identically named methods. We will change this to preserve fully qualified names (`ClassName.method_name`) to ensure graph edges do not cross-pollinate incorrectly.

#### [MODIFY] `ingest.py`
*   **Incremental Ingestion:** Hard-flushing the entire FalkorDB on every run (`FLUSH_GRAPH_ON_INGEST=True`) is not viable for real-world use. We will implement a file-modification state tracker.
*   **LLM Edge-Resolution Phase (The AMD Hackathon Flex):** Static analysis cannot resolve dynamic dispatch (e.g., `obj.save()`). During ingestion, we will flag ambiguous nodes. We will introduce a lightweight pre-processing pass where the LLM analyzes these ambiguous nodes and inserts "AI-Inferred" edges into the graph.
*   **AI Node Summarization (New):** Docstrings are often missing or poorly written. During ingestion, we will pass the raw code of each node to the local LLM to generate a concise, 1-2 sentence summary of its exact behavior. This `summary` property is stored on the node. When the Agent is in Phase 1 (Mapping), it reads these summaries alongside the graph topology, granting it immediate semantic understanding of the blast radius without ever having to trigger `get_source_code`.

### 5. The Presentation: Visual Demo UI with Semantic Zoom
A command-line interface will not impress hackathon judges as much as a visual representation of the codebase.

#### [NEW] `app.py`
*   Build a fast **Streamlit** application.
*   **Left Panel:** The chat interface communicating with `agent.py`.
*   **Right Panel:** An interactive network graph using `pyvis` or `streamlit-agraph`. 
*   **Semantic Zooming (The Wow Factor):** Leveraging our Highway Model, the visualizer will support macro-to-micro views. At the top level, it displays the Modules and "Thick" edges. As the user or agent clicks into a cluster, it explodes into Classes (Medium edges), and finally Functions (Thin edges). As the agent triggers `get_blast_radius`, the UI will actively highlight the path across these highways.

---

## Verification Plan

### Local Testing (Phase 2)
1.  **Parser Accuracy:** Run unit tests after the `parser.py` namespace fix to ensure `CALLS` edges are correctly attributed to classes.
2.  **Incremental Ingest:** Modify a single file and verify that `ingest.py` updates only that file's nodes in FalkorDB without a full flush.
3.  **Two-Phase Agent:** Monitor the tool call sequence in `demo_cli.py` or the new `app.py` to confirm the LLM correctly maps the topology *before* pulling source code.
4.  **UI Visualization:** Ensure the Streamlit app successfully binds to the local FalkorDB instance and renders the graph in under 2 seconds.
