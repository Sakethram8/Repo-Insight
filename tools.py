# tools.py
"""
Deterministic, testable query functions over FalkorDB.
No LLM calls. These are the functions the agent will invoke.
"""

import json
import logging
import math
import time
from functools import lru_cache
from pathlib import Path
import numpy as np
from config import IMPACT_RADIUS_MAX_DEPTH, BLAST_RADIUS_MAX_DEPTH, IMPACT_RADIUS_WARN_THRESHOLD
import falkordb
from embedder import embed_text

logger = logging.getLogger(__name__)


def get_function_context(fqn: str, graph: falkordb.Graph) -> dict:
    # Try exact FQN match first, then fallback to name match
    result = graph.query(
        """MATCH (f:Function)
           WHERE f.fqn = $query OR f.name = $query
           OPTIONAL MATCH (f)-[:DEFINED_IN]->(m)
           RETURN f.fqn, f.name, f.file_path, f.start_line, f.end_line,
                  f.summary, f.is_method, f.class_name, m.name
           LIMIT 1""",
        {"query": fqn},
    )

    if len(result.result_set) == 0:
        return {"found": False, "query": fqn}

    row = result.result_set[0]
    summary = row[5] if row[5] else None
    class_name = row[7] if row[7] else None

    return {
        "found": True,
        "fqn": row[0],
        "name": row[1],
        "file_path": row[2],
        "start_line": row[3],
        "end_line": row[4],
        "summary": summary,
        "is_method": bool(row[6]),
        "class_name": class_name,
        "defined_in_module": row[8] or "",
    }


def get_callers(fqn: str, graph: falkordb.Graph) -> dict:
    result = graph.query(
        """MATCH (caller:Function)-[:CALLS]->(target:Function {fqn: $fqn})
           RETURN caller.fqn, caller.file_path, caller.start_line""",
        {"fqn": fqn},
    )
    callers = [{"fqn": r[0], "file_path": r[1], "start_line": r[2]} for r in result.result_set]
    return {"target_fqn": fqn, "caller_count": len(callers), "callers": callers}


def get_callees(fqn: str, graph: falkordb.Graph) -> dict:
    result = graph.query(
        """MATCH (src:Function {fqn: $fqn})-[:CALLS]->(callee:Function)
           RETURN callee.fqn, callee.file_path, callee.start_line""",
        {"fqn": fqn},
    )
    callees = [{"fqn": r[0], "file_path": r[1], "start_line": r[2]} for r in result.result_set]
    return {"target_fqn": fqn, "callee_count": len(callees), "callees": callees}


def get_downstream_deps(fqn: str, graph: falkordb.Graph, max_depth: int = IMPACT_RADIUS_MAX_DEPTH) -> dict:
    max_depth = int(max_depth)
    cypher = (
        f"MATCH p = (src:Function {{fqn: $fqn}})-[:CALLS*1..{max_depth}]->(impacted:Function) "
        f"RETURN DISTINCT impacted.fqn, impacted.file_path, length(p) "
        f"ORDER BY length(p) ASC"
    )
    result = graph.query(cypher, {"fqn": fqn})
    impacted = [{"fqn": r[0], "file_path": r[1], "distance": r[2]} for r in result.result_set]
    return {
        "source_fqn": fqn,
        "direction": "downstream",
        "depth": max_depth,
        "impacted_count": len(impacted),
        "warning": len(impacted) > IMPACT_RADIUS_WARN_THRESHOLD,
        "impacted": impacted,
    }


def get_upstream_callers(fqn: str, graph: falkordb.Graph, max_depth: int = BLAST_RADIUS_MAX_DEPTH) -> dict:
    max_depth = int(max_depth)
    cypher = (
        f"MATCH p = (affected:Function)-[:CALLS*1..{max_depth}]->(target:Function {{fqn: $fqn}}) "
        f"RETURN DISTINCT affected.fqn, affected.file_path, length(p) "
        f"ORDER BY length(p) ASC"
    )
    result = graph.query(cypher, {"fqn": fqn})
    affected = [{"fqn": r[0], "file_path": r[1], "distance": r[2]} for r in result.result_set]
    return {
        "target_fqn": fqn,
        "direction": "upstream",
        "depth": max_depth,
        "affected_count": len(affected),
        "warning": len(affected) > IMPACT_RADIUS_WARN_THRESHOLD,
        "affected": affected,
    }


def get_macro_architecture(graph: falkordb.Graph, inherits_weight: int = 10, imports_weight: int = 5) -> dict:
    # Sums CALLS, INHERITS_FROM, and IMPORTS edges between modules to generate Thick Edges
    query = f"""
    MATCH (f1:Function)-[c:CALLS]->(f2:Function)
    WHERE f1.module_name <> f2.module_name
    WITH f1.module_name AS src, f2.module_name AS tgt, count(c) AS weight
    RETURN src, tgt, weight, 'CALLS' as type
    UNION ALL
    MATCH (c1:Class)-[i:INHERITS_FROM]->(c2:Class)
    MATCH (c1)-[:DEFINED_IN]->(m1:Module)
    MATCH (c2)-[:DEFINED_IN]->(m2:Module)
    WHERE m1.name <> m2.name
    RETURN m1.name AS src, m2.name AS tgt, {inherits_weight} AS weight, 'INHERITS_FROM' as type
    UNION ALL
    MATCH (m1:Module)-[i:IMPORTS]->(m2:Module)
    WHERE m1.name <> m2.name
    RETURN m1.name AS src, m2.name AS tgt, {imports_weight} AS weight, 'IMPORTS' as type
    """
    result = graph.query(query)
    edges = {}
    for row in result.result_set:
        src, tgt, weight, etype = row[0], row[1], row[2], row[3]
        key = (src, tgt)
        if key not in edges:
            edges[key] = {"source": src, "target": tgt, "weight": 0, "types": set()}
        edges[key]["weight"] += weight
        edges[key]["types"].add(etype)
        
    for k in edges:
        edges[k]["types"] = list(edges[k]["types"])
        
    return {"modules": list(edges.values())}


def get_class_architecture(module_name: str, graph: falkordb.Graph, inherits_weight: int = 10) -> dict:
    query = f"""
    MATCH (f1:Function)-[c:CALLS]->(f2:Function)
    WHERE f1.module_name = $module AND f1.class_name <> '' AND f2.class_name <> ''
    WITH f1.class_name AS src, f2.class_name AS tgt, count(c) AS weight
    RETURN src, tgt, weight, 'CALLS' as type
    UNION ALL
    MATCH (c1:Class)-[i:INHERITS_FROM]->(c2:Class)
    MATCH (c1)-[:DEFINED_IN]->(m1:Module) WHERE m1.name = $module
    RETURN c1.name AS src, c2.name AS tgt, {inherits_weight} AS weight, 'INHERITS_FROM' as type
    """
    result = graph.query(query, {"module": module_name})
    edges = {}
    for row in result.result_set:
        src, tgt, weight, etype = row[0], row[1], row[2], row[3]
        if src == tgt:
            continue
        key = (src, tgt)
        if key not in edges:
            edges[key] = {"source": src, "target": tgt, "weight": 0, "types": set()}
        edges[key]["weight"] += weight
        edges[key]["types"].add(etype)
        
    for k in edges:
        edges[k]["types"] = list(edges[k]["types"])
        
    return {"module": module_name, "class_edges": list(edges.values())}


def get_source_code(fqn: str, graph: falkordb.Graph,
                    repo_root_override: str | None = None) -> dict:
    result = graph.query(
        "MATCH (n:Function) WHERE n.fqn = $fqn RETURN n.file_path, n.start_line, n.end_line\n"
        "UNION\n"
        "MATCH (n:Class) WHERE n.fqn = $fqn RETURN n.file_path, n.start_line, n.end_line",
        {"fqn": fqn},
    )
    if len(result.result_set) == 0:
        return {"found": False, "fqn": fqn}
    
    row = result.result_set[0]
    file_path, start_line, end_line = row[0], row[1], row[2]

    source = "<file not readable>"
    candidates = []

    if repo_root_override:
        candidates.append(Path(repo_root_override) / file_path)
    else:
        meta_result = graph.query("MATCH (m:Meta {key: 'repo_root'}) RETURN m.value LIMIT 1")
        repo_root = meta_result.result_set[0][0] if meta_result.result_set else None
        if repo_root:
            candidates.append(Path(repo_root) / file_path)
        candidates.append(Path(file_path))
        candidates.append(Path(".") / file_path)

    for candidate in candidates:
        try:
            lines = candidate.read_text(encoding="utf-8").splitlines()
            source = "\n".join(lines[max(0, start_line - 1):end_line])
            break
        except (OSError, UnicodeDecodeError):
            continue

    if source == "<file not readable>":
        return {
            "found": False, "fqn": fqn, "file_path": file_path,
            "error": "Could not read file from disk. Ensure repo_root is correct or path is accessible."
        }

    return {
        "found": True, "fqn": fqn, "file_path": file_path,
        "start_line": start_line, "end_line": end_line, "source": source,
    }

@lru_cache(maxsize=5000)
def _get_cached_embedding(fqn: str, embedding_json: str) -> tuple[float, ...]:
    """Deserialize and cache a node embedding. Returns tuple for lru_cache hashability."""
    import json as _json
    return tuple(_json.loads(embedding_json))


def semantic_search(query: str, graph: falkordb.Graph, top_k: int = 5) -> dict:
    query_embedding = embed_text(query)

    # Calculate in-degree to rank core utilities higher
    func_query = """
    MATCH (f:Function) WHERE f.embedding IS NOT NULL
    OPTIONAL MATCH ()-[c:CALLS]->(f)
    RETURN 'Function', f.fqn, f.file_path, f.summary, f.embedding, count(c) as in_degree
    """
    func_results = graph.query(func_query)

    class_query = """
    MATCH (c:Class) WHERE c.embedding IS NOT NULL
    OPTIONAL MATCH ()-[i:INHERITS_FROM]->(c)
    RETURN 'Class', c.fqn, c.file_path, c.summary, c.embedding, count(i) as in_degree
    """
    class_results = graph.query(class_query)

    module_query = """
    MATCH (m:Module) WHERE m.embedding IS NOT NULL
    OPTIONAL MATCH ()-[d:DEFINED_IN]->(m)
    RETURN 'Module', m.name, m.file_path, 'Module definition', m.embedding, count(d) as in_degree
    """
    module_results = graph.query(module_query)

    all_rows = list(func_results.result_set) + list(class_results.result_set) + list(module_results.result_set)

    # Filter rows with valid embeddings and deserialize via lru_cache
    valid_rows = []
    for row in all_rows:
        label, fqn, file_path, summary, embedding_str, in_degree = row
        if not embedding_str or not fqn:
            continue
        try:
            emb = list(_get_cached_embedding(fqn, embedding_str))
            valid_rows.append((row, emb))
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse embedding for %s", fqn)
            continue

    if not valid_rows:
        return {"query": query, "results": []}

    # Vectorized cosine similarity via NumPy
    matrix = np.array([emb for _, emb in valid_rows], dtype=np.float32)
    q_vec = np.array(query_embedding, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    q_norm = np.linalg.norm(q_vec)
    scores = (matrix @ q_vec) / (norms * q_norm + 1e-8)

    # Apply in-degree re-ranking bonus
    scored = []
    for i, (row, _emb) in enumerate(valid_rows):
        label, fqn, file_path, summary, _embedding_str, in_degree = row
        base_score = float(scores[i])
        weight = 0.05
        final_score = base_score + (math.log(in_degree + 1) * weight)
        scored.append({
            "label": label, "fqn": fqn, "file_path": file_path,
            "summary": summary, "in_degree": in_degree, "score": round(final_score, 4),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"query": query, "results": scored[:top_k]}
