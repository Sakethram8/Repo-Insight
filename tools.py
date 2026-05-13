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
        f"WHERE ALL(n IN nodes(p)[1..-1] WHERE size((n)<-[:CALLS]-()) < 50) "
        f"RETURN DISTINCT impacted.fqn, impacted.file_path, length(p) "
        f"ORDER BY length(p) ASC "
        f"LIMIT 300"
    )
    result = graph.query(cypher, {"fqn": fqn})
    impacted = [{"fqn": r[0], "file_path": r[1], "distance": r[2]} for r in result.result_set]
    return {
        "source_fqn": fqn,
        "direction": "downstream",
        "depth": max_depth,
        "impacted_count": len(impacted),
        "truncated": len(impacted) >= 300,   # Fix 5: warn when LIMIT was hit
        "warning": len(impacted) >= IMPACT_RADIUS_WARN_THRESHOLD,
        "impacted": impacted,
    }


def get_upstream_callers(fqn: str, graph: falkordb.Graph, max_depth: int = BLAST_RADIUS_MAX_DEPTH) -> dict:
    max_depth = int(max_depth)
    cypher = (
        f"MATCH p = (caller:Function)-[:CALLS*1..{max_depth}]->(src:Function {{fqn: $fqn}}) "
        f"RETURN DISTINCT caller.fqn, caller.file_path, length(p) "
        f"ORDER BY length(p) ASC LIMIT 300"
    )
    result = graph.query(cypher, {"fqn": fqn})
    affected = [{"fqn": r[0], "file_path": r[1], "distance": r[2]} for r in result.result_set]
    return {
        "target_fqn": fqn,
        "direction": "upstream",
        "depth": max_depth,
        "affected_count": len(affected),
        "truncated": len(affected) >= 300,   # Fix 5: warn when LIMIT was hit
        "warning": len(affected) >= IMPACT_RADIUS_WARN_THRESHOLD,
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


def semantic_search(query: str, graph: falkordb.Graph, top_k: int = 5,
                    index=None) -> dict:
    """Find functions/classes semantically similar to *query*.

    When *index* (a GraphIndex) is provided, embeddings and in-degree are
    served from memory — zero Redis I/O after the initial index build.
    Falls back to querying FalkorDB directly when no index is available.
    Returns {"query": query, "results": []} gracefully when no embedding
    backend is available (Tier 3 degradation).
    """
    query_embedding = embed_text(query)
    if not query_embedding:
        return {"query": query, "results": [],
                "note": "Embedding backend unavailable — semantic search disabled"}
    q_vec = np.array(query_embedding, dtype=np.float32)

    # --- Fast path: use in-memory GraphIndex ---
    if index is not None and index.embeddings:
        fqns   = list(index.embeddings.keys())
        matrix = np.array(list(index.embeddings.values()), dtype=np.float32)
        scores = (matrix @ q_vec) / (
            np.linalg.norm(matrix, axis=1) * np.linalg.norm(q_vec) + 1e-8
        )
        top_indices = np.argsort(scores)[::-1][: top_k * 10]
        scored = []
        for i in top_indices:
            fqn        = fqns[i]
            meta       = index.fn_meta.get(fqn, {})
            in_degree  = index.in_degree.get(fqn, 0)
            final      = float(scores[i]) + math.log(in_degree + 1) * 0.05
            scored.append({
                "label": "Function",
                "fqn": fqn,
                "file_path": meta.get("file_path", ""),
                "summary": index.summaries.get(fqn, ""),
                "in_degree": in_degree,
                "score": round(final, 4),
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return {"query": query, "results": scored[:top_k]}

    # --- Fallback: query FalkorDB directly (no index available) ---
    func_results = graph.query(
        "MATCH (f:Function) WHERE f.embedding IS NOT NULL "
        "RETURN 'Function', f.fqn, f.file_path, f.summary, f.embedding"
    )
    class_results = graph.query(
        "MATCH (c:Class) WHERE c.embedding IS NOT NULL "
        "RETURN 'Class', c.fqn, c.file_path, c.summary, c.embedding"
    )

    all_rows = list(func_results.result_set) + list(class_results.result_set)
    valid_rows = []
    for row in all_rows:
        label, fqn, file_path, summary, embedding_str = row
        if not embedding_str or not fqn:
            continue
        try:
            emb = list(_get_cached_embedding(fqn, embedding_str))
            valid_rows.append((row, emb))
        except Exception:
            continue

    if not valid_rows:
        return {"query": query, "results": []}

    matrix = np.array([emb for _, emb in valid_rows], dtype=np.float32)
    scores = (matrix @ q_vec) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q_vec) + 1e-8)

    top_indices = np.argsort(scores)[::-1][:50]
    scored = []
    for idx in top_indices:
        row = valid_rows[idx][0]
        label, fqn, file_path, summary, _ = row
        base_score = float(scores[idx])
        deg_res = graph.query(
            "MATCH ()-[r:CALLS]->(n) WHERE n.fqn = $fqn RETURN count(r)", {"fqn": fqn}
        )
        in_degree = deg_res.result_set[0][0] if deg_res.result_set else 0
        final_score = base_score + (math.log(in_degree + 1) * 0.05)
        scored.append({
            "label": label, "fqn": fqn, "file_path": file_path,
            "summary": summary, "in_degree": in_degree, "score": round(final_score, 4),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"query": query, "results": scored[:top_k]}


# Aliases used by mcp_server.py
get_impact_radius = get_downstream_deps
get_blast_radius = get_upstream_callers


# ---------------------------------------------------------------------------
# Change impact tools
# ---------------------------------------------------------------------------

def get_cross_module_callers(fqn: str, graph) -> dict:
    """Callers of fqn that live in a different module.

    These are the only callers that matter when a function's interface changes —
    intra-module callers can always be updated in the same edit session.
    """
    result = graph.query(
        """MATCH (caller:Function)-[:CALLS]->(target:Function {fqn: $fqn})
           WHERE caller.module_name <> target.module_name
           RETURN caller.fqn, caller.file_path, caller.start_line, caller.module_name
           ORDER BY caller.module_name""",
        {"fqn": fqn},
    )
    callers = [
        {"fqn": r[0], "file_path": r[1], "start_line": r[2], "module_name": r[3]}
        for r in result.result_set
    ]
    return {
        "target_fqn": fqn,
        "cross_module_caller_count": len(callers),
        "callers": callers,
    }


def get_file_interface(file_path: str, graph) -> dict:
    """All functions in a file with their current stored signatures.

    Call this before editing a file to know which function contracts exist
    and which could be broken by a signature change.
    """
    import json as _json
    result = graph.query(
        """MATCH (f:Function)
           WHERE f.file_path = $file_path AND f.name <> '<module>'
           RETURN f.fqn, f.name, f.params, f.return_annotation,
                  f.is_method, f.class_name, f.start_line
           ORDER BY f.start_line""",
        {"file_path": file_path},
    )
    functions = []
    for r in result.result_set:
        fqn, name, params_raw, return_ann, is_method, class_name, start_line = r
        try:
            params = _json.loads(params_raw) if params_raw else []
        except Exception:
            params = []
        functions.append({
            "fqn": fqn,
            "name": name,
            "params": params,
            "return_annotation": return_ann or None,
            "is_method": is_method,
            "class_name": class_name or None,
            "start_line": start_line,
        })
    return {"file_path": file_path, "function_count": len(functions), "functions": functions}


def get_module_readers(module_name: str, graph) -> dict:
    """Functions in other modules that READ named values exported from this module.

    Captures the data-flow dependency that CALLS edges miss: when a function
    imports and uses a constant, config value, or any non-callable name from
    another module, that function may break if the exported name changes.
    """
    result = graph.query(
        """MATCH (f:Function)-[r:READS]->(m:Module {name: $module_name})
           WHERE f.module_name <> $module_name
           RETURN f.fqn, f.file_path, f.start_line, r.name
           ORDER BY r.name, f.module_name""",
        {"module_name": module_name},
    )
    readers = [
        {"fqn": r[0], "file_path": r[1], "start_line": r[2], "read_name": r[3]}
        for r in result.result_set
    ]
    return {
        "module_name": module_name,
        "reader_count": len(readers),
        "readers": readers,
    }


def _param_name_only(param: str) -> str:
    """Extract the bare parameter name from a signature fragment.

    Fix 7: strips type annotations and default values so that adding an
    annotation to an existing parameter ('x' → 'x: int') is not reported
    as a breaking interface change.

    Examples:
        'x'            → 'x'
        'x: int'       → 'x'
        'x: int = 5'   → 'x'
        '*args'        → '*args'
        '**kwargs'     → '**kwargs'
    """
    return param.split(":")[0].split("=")[0].strip()


def analyze_edit_impact(file_path: str, changed_signatures: list[dict], graph) -> dict:
    """Given functions whose signatures changed, return which external callers are at risk.

    Only interface changes (params / return type) are reported — body-only changes
    produce no output since callers cannot observe them.

    Args:
        file_path: The file that was edited (for context only).
        changed_signatures: List of dicts with keys:
            fqn, old_params (list), new_params (list),
            old_return (str|None), new_return (str|None)
        graph: FalkorDB graph connection.

    Returns:
        {file_path, interface_breaking_changes (int), impact: [{fqn, change, at_risk_callers, caller_count}]}
    """
    report = []
    for sig in changed_signatures:
        params_changed = (
            [_param_name_only(p) for p in sig.get("old_params", [])]
            != [_param_name_only(p) for p in sig.get("new_params", [])]
        )
        return_changed = (sig.get("old_return") or "") != (sig.get("new_return") or "")

        if not params_changed and not return_changed:
            continue  # body-only change — callers unaffected

        callers = get_cross_module_callers(sig["fqn"], graph)["callers"]
        if not callers:
            continue  # interface changed but no external callers — safe

        report.append({
            "fqn": sig["fqn"],
            "change": {
                "params_changed": params_changed,
                "return_changed": return_changed,
                "old_params": sig.get("old_params", []),
                "new_params": sig.get("new_params", []),
                "old_return": sig.get("old_return"),
                "new_return": sig.get("new_return"),
            },
            "at_risk_callers": callers,
            "caller_count": len(callers),
        })

    return {
        "file_path": file_path,
        "interface_breaking_changes": len(report),
        "impact": report,
    }
