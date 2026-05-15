# ingest.py
"""
Take parsed data and write it into FalkorDB.
Uses MERGE (not CREATE) for node upserts. Idempotent within a run.
Optionally flushes graph before run based on config.FLUSH_GRAPH_ON_INGEST.
"""

import json
import logging
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional
import falkordb
import os as _os
import signal as _signal
import concurrent.futures as _cf
SKIP_JEDI = _os.getenv("SKIP_JEDI", "false").lower() in ("true", "1")

from resolver import (
    build_module_fqn_map, build_symbol_table, resolve_callee, resolve_base_class,
    build_reexport_map, canonicalize_fqn, enrich_star_imports,
)
from config import (FALKORDB_HOST, FALKORDB_PORT, GRAPH_NAME,
                    FLUSH_GRAPH_ON_INGEST, INGEST_CONCURRENCY)
from parser import ParsedFile, parse_file, SKIP_DIRS
from embedder import embed_text, embed_texts, build_embedding_text

# Directories to skip during ingestion — no useful code here
# SKIP_DIRS = {
#     "tests", "test", "testing",
#     "build", "dist",
#     "__pycache__",
#     ".git", ".tox",
#     "benchmarks", "examples",
#     "migrations",          # Django migrations are auto-generated
#     "node_modules",
# }

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_to_module(file_path: str) -> str:
    """Convert a relative file path like 'foo/bar.py' to module name 'foo.bar'."""
    mod = file_path.replace("/", ".").replace("\\", ".")
    return mod[:-3] if mod.endswith(".py") else mod


# ---------------------------------------------------------------------------
# Connection with retry
# ---------------------------------------------------------------------------

_MAX_CONNECT_RETRIES = 3
_CONNECT_BACKOFF_BASE = 1.0  # seconds


def get_connection(graph_name: str | None = None) -> falkordb.Graph:
    """Connect to FalkorDB with exponential-backoff retry.

    graph_name overrides the GRAPH_NAME config — used by parallel SWE-bench
    workers so each worker has its own isolated graph namespace.
    """
    name = graph_name or GRAPH_NAME
    last_err: Exception | None = None
    for attempt in range(_MAX_CONNECT_RETRIES):
        try:
            db = falkordb.FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT,
                                   socket_timeout=None,
                                   socket_connect_timeout=10)
            return db.select_graph(name)
        except Exception as e:
            last_err = e
            wait = _CONNECT_BACKOFF_BASE * (2 ** attempt)
            logger.warning(
                "FalkorDB connection attempt %d/%d failed: %s  — retrying in %.1fs",
                attempt + 1, _MAX_CONNECT_RETRIES, e, wait,
            )
            time.sleep(wait)
    raise ConnectionError(
        f"Failed to connect to FalkorDB at {FALKORDB_HOST}:{FALKORDB_PORT} "
        f"after {_MAX_CONNECT_RETRIES} attempts: {last_err}"
    )


def drop_graph(graph_name: str | None = None) -> None:
    """Delete the graph key via raw Redis DEL — removes nodes, edges AND schema."""
    import redis as _redis
    name = graph_name or GRAPH_NAME
    try:
        r = _redis.Redis(host=FALKORDB_HOST, port=FALKORDB_PORT,
                         socket_timeout=None, socket_connect_timeout=10)
        removed = r.delete(name)
        logger.info("Graph key deleted (Redis DEL, keys removed: %d)", removed)
    except Exception as e:
        logger.warning("drop_graph failed (non-fatal): %s", e)


def create_indices(graph: falkordb.Graph) -> None:
    """Create FalkorDB indices for fast lookups. Idempotent."""
    index_queries = [
        "CREATE INDEX FOR (f:Function) ON (f.fqn)",
        "CREATE INDEX FOR (c:Class) ON (c.fqn)",
        "CREATE INDEX FOR (m:Module) ON (m.name)",
        "CREATE INDEX FOR (s:FileState) ON (s.file_path)",
        "CREATE INDEX FOR ()-[r:READS]-() ON (r.name)",
    ]
    for query in index_queries:
        try:
            graph.query(query)
        except Exception as e:
            err_msg = str(e).lower()
            if "already" in err_msg or "exists" in err_msg or "index" in err_msg:
                continue
            raise


import json


def _file_content_hash(file_path: Path) -> str:
    """MD5 of file bytes — used instead of mtime for change detection.
    git checkout sets every file's mtime to 'now', so mtime-based comparison
    always marks every file as changed even when content is identical between
    two commits. Content hash only changes when bytes actually change."""
    import hashlib
    try:
        return hashlib.md5(file_path.read_bytes()).hexdigest()
    except Exception:
        return ""


def extract_source_code(file_path: Path, start_line: int, end_line: int) -> str:
    """Read source lines from a file. Returns empty string on failure."""
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[start_line - 1 : end_line])
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Signature hashing
# ---------------------------------------------------------------------------

def _signature_hash(name: str, params: list[str], return_annotation: str | None) -> str:
    """16-char hex hash of a function's observable interface (name + params + return type).

    Unchanged if only the body changes. Changes when the contract changes.
    Used to detect interface-breaking changes during incremental re-ingestion.
    """
    import hashlib
    raw = json.dumps([name, params, return_annotation or ""], sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------
def _bulk_write(graph: falkordb.Graph, query: str, items: list[dict],
                chunk_size: int = 500, param: str = "nodes") -> None:
    """Write items in chunks to avoid memory pressure on large UNWIND."""
    for i in range(0, len(items), chunk_size):
        _write_no_alarm(graph, query, {param: items[i:i + chunk_size]})
def _write_no_alarm(graph, query, params):
    """Perform a graph write with SIGALRM temporarily disabled."""
    # signal.alarm is only available on Unix. Gracefully fallback if on Windows.
    try:
        prev = _signal.alarm(0)   # pause alarm
    except AttributeError:
        prev = 0

    try:
        graph.query(query, params)
    finally:
        if prev > 0:
            _signal.alarm(prev)  # restore remaining time

def ingest_parsed_files(
    parsed_files: list[ParsedFile],
    graph: falkordb.Graph,
    repo_root: Path,
    on_progress=None,
) -> None:
    """Write parsed AST entities into FalkorDB as nodes and edges.

    Phase order:
      1. Module nodes
      2. Symbol tables (in-memory, no graph writes)
      3. Class nodes  (FQNs from qualname)
      4. Function nodes (FQNs from qualname) + <module> pseudo-nodes
      5. DEFINED_IN + INHERITS_FROM edges (exact FQN via symbol table)
      6. IMPORTS edges
      7. CALLS edges — symbol-table resolved (exact FQN match)
      8. CALLS edges — Jedi precision pass on unresolved calls only
    """
    # --- Phase 2: symbol tables (before any node writes) ---
    module_fqn_map = build_module_fqn_map(parsed_files)
    symbol_tables = {
        pf.file_path: build_symbol_table(pf, module_fqn_map[pf.file_path], module_fqn_map)
        for pf in parsed_files
    }
    # Fix 1: build reexport map from __init__.py files so alias FQNs are
    # canonicalized to their definition-site FQNs before CALLS edges are written.
    reexport_map = build_reexport_map(parsed_files, module_fqn_map)
    # Fix 2: resolve within-repo star imports by enumerating source exports.
    enrich_star_imports(parsed_files, symbol_tables, module_fqn_map)
    # known_fqns is populated as nodes are written; used for callee validation
    known_fqns: set[str] = set()

    # --- Step 1: Upsert Module nodes ---
    file_modules: dict[str, str] = {}   # mod_fqn → file_path
    import_modules: set[str] = set()

    for pf in parsed_files:
        mod_name = module_fqn_map[pf.file_path]
        file_modules[mod_name] = pf.file_path
        for imp in pf.imports:
            if imp.module and imp.module != ".":
                import_modules.add(imp.module)

    mod_emb_texts = [f"{mod_name}. Module defined in {fpath}"
                     for mod_name, fpath in file_modules.items()]
    mod_embeddings = embed_texts(mod_emb_texts) if mod_emb_texts else []

    module_nodes = []
    for i, (mod_name, fpath) in enumerate(file_modules.items()):
        module_nodes.append({
            "name": mod_name,
            "file_path": fpath,
            "embedding": json.dumps(mod_embeddings[i]) if mod_embeddings else None,
        })
    for mod_name in import_modules:
        if mod_name and mod_name not in file_modules:
            module_nodes.append({"name": mod_name, "file_path": "", "embedding": None})

    module_nodes = [n for n in module_nodes if n.get("name") and n["name"].strip() not in ("", ".")]
    if module_nodes:
        _bulk_write(graph,
            """UNWIND $nodes AS n
               MERGE (m:Module {name: n.name})
               SET m.file_path = n.file_path
               WITH m, n WHERE n.file_path <> "" AND n.embedding IS NOT NULL
               SET m.embedding = n.embedding""",
            module_nodes,
        )

    # --- Step 2: Upsert Class nodes ---
    class_nodes: list[dict] = []
    class_emb_texts: list[str] = []
    for pf in parsed_files:
        mod_name = module_fqn_map[pf.file_path]
        for cls in pf.classes:
            emb_text = build_embedding_text(cls.name, cls.docstring, pf.file_path)
            class_emb_texts.append(emb_text)
            fqn = f"{mod_name}.{cls.qualname}"
            known_fqns.add(fqn)
            class_nodes.append({
                "fqn": fqn, "name": cls.name, "file_path": pf.file_path,
                "start_line": cls.start_line, "end_line": cls.end_line,
                "docstring": cls.docstring or "",
                "summary": cls.docstring or "",   # docstring IS the summary
            })

    if on_progress:
        on_progress("Embeddings", 2, 3, "Generating vectors...")

    if class_emb_texts:
        class_embeddings = embed_texts(class_emb_texts)
        for i, node in enumerate(class_nodes):
            if class_embeddings and class_embeddings[i]:  # Tier 3: skip if empty
                node["embedding"] = json.dumps(class_embeddings[i])

    if class_nodes:
        _bulk_write(graph,
            """UNWIND $nodes AS n
               MERGE (c:Class {fqn: n.fqn})
               SET c.name = n.name,
                   c.file_path = n.file_path,
                   c.start_line = n.start_line,
                   c.end_line = n.end_line,
                   c.docstring = n.docstring,
                   c.summary = n.summary,
                   c.embedding = n.embedding""",
            class_nodes,
        )

    # --- Step 3: Upsert Function nodes ---
    func_nodes: list[dict] = []
    func_emb_texts: list[str] = []
    for pf in parsed_files:
        mod_name = module_fqn_map[pf.file_path]
        for func in pf.functions:
            # Embedding text: docstring first, then typed signature, then name only
            if func.docstring:
                emb_source = func.docstring
            elif func.params or func.return_annotation:
                params_str = ", ".join(func.params)
                ret = f" -> {func.return_annotation}" if func.return_annotation else ""
                emb_source = f"{func.name}({params_str}){ret}"
            else:
                emb_source = None
            func_emb_texts.append(build_embedding_text(func.name, emb_source, pf.file_path))
            fqn = f"{mod_name}.{func.qualname}"
            known_fqns.add(fqn)
            func_nodes.append({
                "fqn": fqn, "name": func.name, "file_path": pf.file_path,
                "start_line": func.start_line, "end_line": func.end_line,
                "docstring": func.docstring or "", "is_method": func.is_method,
                "class_name": func.class_name or "", "module_name": mod_name,
                "summary": func.docstring or "",   # docstring IS the summary
                "params": json.dumps(func.params),
                "decorators": json.dumps(func.decorators),
                "return_annotation": func.return_annotation or "",
                "signature_hash": _signature_hash(func.name, func.params, func.return_annotation),
            })

    if func_emb_texts:
        func_embeddings = embed_texts(func_emb_texts)
        for i, node in enumerate(func_nodes):
            if func_embeddings and func_embeddings[i]:  # Tier 3: skip if empty
                node["embedding"] = json.dumps(func_embeddings[i])

    if func_nodes:
        _bulk_write(graph,
            """UNWIND $nodes AS n
               MERGE (f:Function {fqn: n.fqn})
               SET f.name = n.name,
                   f.file_path = n.file_path,
                   f.start_line = n.start_line,
                   f.end_line = n.end_line,
                   f.docstring = n.docstring,
                   f.is_method = n.is_method,
                   f.class_name = n.class_name,
                   f.module_name = n.module_name,
                   f.summary = n.summary,
                   f.embedding = n.embedding,
                   f.params = n.params,
                   f.decorators = n.decorators,
                   f.return_annotation = n.return_annotation,
                   f.signature_hash = n.signature_hash""",
            func_nodes,
        )

    # Add <module> pseudo-function per file — represents module-level execution
    module_pseudo_nodes = []
    for pf in parsed_files:
        mod_name = module_fqn_map[pf.file_path]
        pseudo_fqn = f"{mod_name}.<module>"
        known_fqns.add(pseudo_fqn)
        module_pseudo_nodes.append({
            "fqn": pseudo_fqn, "name": "<module>",
            "file_path": pf.file_path, "start_line": 1, "end_line": 0,
            "docstring": "", "is_method": False, "class_name": "",
            "module_name": mod_name,
            "summary": f"Module-level initialization code for {mod_name}",
            "params": "[]", "decorators": "[]", "return_annotation": "",
            "embedding": None,
        })
    if module_pseudo_nodes:
        _bulk_write(graph,
            """UNWIND $nodes AS n
               MERGE (f:Function {fqn: n.fqn})
               SET f.name = n.name, f.file_path = n.file_path,
                   f.start_line = n.start_line, f.end_line = n.end_line,
                   f.docstring = n.docstring, f.is_method = n.is_method,
                   f.class_name = n.class_name, f.module_name = n.module_name,
                   f.summary = n.summary, f.params = n.params,
                   f.decorators = n.decorators, f.return_annotation = n.return_annotation""",
            module_pseudo_nodes,
        )

    # --- Step 4: DEFINED_IN and INHERITS_FROM edges ---
    func_to_class: list[dict] = []
    func_to_mod: list[dict] = []
    class_to_mod: list[dict] = []
    inherits_resolved: list[dict] = []

    for pf in parsed_files:
        mod_name = module_fqn_map[pf.file_path]
        table = symbol_tables[pf.file_path]

        for func in pf.functions:
            fqn = f"{mod_name}.{func.qualname}"
            if func.is_method:
                # class_name is immediate enclosing class; qualname may have deeper nesting
                # DEFINED_IN points to the immediate class
                class_qualname = func.qualname.rsplit(".", 1)[0] if "." in func.qualname else func.class_name
                class_fqn = f"{mod_name}.{class_qualname}"
                func_to_class.append({"fqn": fqn, "cfqn": class_fqn})
            else:
                func_to_mod.append({"fqn": fqn, "mname": mod_name})

        # <module> pseudo-node → DEFINED_IN → Module
        func_to_mod.append({"fqn": f"{mod_name}.<module>", "mname": mod_name})

        for cls in pf.classes:
            class_fqn = f"{mod_name}.{cls.qualname}"
            # DEFINED_IN → Module for top-level; → outer Class for nested
            if "." in cls.qualname:
                outer_qualname = cls.qualname.rsplit(".", 1)[0]
                class_to_mod.append({"cfqn": class_fqn, "mname": outer_qualname,
                                     "is_nested": True, "mod_name": mod_name})
            else:
                class_to_mod.append({"cfqn": class_fqn, "mname": mod_name,
                                     "is_nested": False, "mod_name": mod_name})

            # INHERITS_FROM — resolved via symbol table (exact FQN)
            for base_raw in cls.bases:
                base_fqn = resolve_base_class(base_raw, table)
                if base_fqn:
                    if reexport_map:
                        base_fqn = canonicalize_fqn(base_fqn, reexport_map)
                    inherits_resolved.append({"cfqn": class_fqn, "base_fqn": base_fqn})
                else:
                    logger.debug("INHERITS_FROM unresolved: %s bases %s", class_fqn, base_raw)

    if func_to_class:
        _bulk_write(graph,
            """UNWIND $edges AS e
               MATCH (f:Function {fqn: e.fqn})
               MATCH (c:Class {fqn: e.cfqn})
               MERGE (f)-[:DEFINED_IN]->(c)""",
            func_to_class, chunk_size=500, param="edges")
    if func_to_mod:
        _bulk_write(graph,
            """UNWIND $edges AS e
               MATCH (f:Function {fqn: e.fqn})
               MATCH (m:Module {name: e.mname})
               MERGE (f)-[:DEFINED_IN]->(m)""",
            func_to_mod, chunk_size=500, param="edges")

    # For top-level classes: DEFINED_IN → Module
    top_level_classes = [e for e in class_to_mod if not e["is_nested"]]
    nested_classes = [e for e in class_to_mod if e["is_nested"]]
    if top_level_classes:
        _bulk_write(graph,
            """UNWIND $edges AS e
               MATCH (c:Class {fqn: e.cfqn})
               MATCH (m:Module {name: e.mname})
               MERGE (c)-[:DEFINED_IN]->(m)""",
            top_level_classes, chunk_size=500, param="edges")
    if nested_classes:
        # DEFINED_IN → outer Class (mname here is the outer class qualname)
        nested_with_fqn = [
            {"cfqn": e["cfqn"], "outer_fqn": f"{e['mod_name']}.{e['mname']}"}
            for e in nested_classes
        ]
        _bulk_write(graph,
            """UNWIND $edges AS e
               MATCH (c:Class {fqn: e.cfqn})
               MATCH (outer:Class {fqn: e.outer_fqn})
               MERGE (c)-[:DEFINED_IN]->(outer)""",
            nested_with_fqn, chunk_size=500, param="edges")

    # INHERITS_FROM — exact FQN match (no name collision possible)
    if inherits_resolved:
        _bulk_write(graph,
            """UNWIND $edges AS e
               MATCH (c:Class {fqn: e.cfqn})
               MATCH (base:Class {fqn: e.base_fqn})
               MERGE (c)-[:INHERITS_FROM]->(base)""",
            inherits_resolved, chunk_size=500, param="edges")

    # Fix A: build parent map for inherited method resolution in Step 6.
    # Must be built after inherits_resolved is populated (Step 4) and after
    # known_fqns is fully populated (Steps 2-3 above).
    parent_map = _build_parent_map(inherits_resolved)

    # --- Step 5: IMPORTS edges ---
    import_edges: list[dict] = []
    seen_import_pairs: set[tuple[str, str]] = set()
    for pf in parsed_files:
        src_mod = module_fqn_map[pf.file_path]
        for imp in pf.imports:
            tgt = imp.module
            if not tgt or tgt == ".":
                continue
            pair = (src_mod, tgt)
            if pair not in seen_import_pairs:
                seen_import_pairs.add(pair)
                import_edges.append({
                    "src_name": src_mod,
                    "tgt_name": tgt,
                    "alias": imp.alias or "",
                })
    if import_edges:
        _bulk_write(graph,
            """UNWIND $edges AS e
               MATCH (src:Module {name: e.src_name})
               MATCH (tgt:Module {name: e.tgt_name})
               MERGE (src)-[i:IMPORTS]->(tgt)
               SET i.alias = e.alias""",
            import_edges, chunk_size=500, param="edges")

    # --- Steps 6 + 6b: CALLS and READS edges ---
    pending_calls = _write_call_and_reads_edges(
        parsed_files, graph, module_fqn_map, symbol_tables,
        reexport_map, parent_map, known_fqns,
    )

    # --- Step 7: Jedi precision pass — only on unresolved calls ---
    if SKIP_JEDI:
        logger.info("SKIP_JEDI=true — skipping Jedi call resolution pass")
    else:
        # Group pending calls back by file for Jedi processing
        from collections import defaultdict
        pending_by_file: dict[str, list] = defaultdict(list)
        pf_by_path: dict[str, ParsedFile] = {pf.file_path: pf for pf in parsed_files}
        for pf, call in pending_calls:
            pending_by_file[pf.file_path].append(call)

        def _resolve_one_file(file_path: str) -> list[dict]:
            if not file_path.endswith(".py"):
                return []
            pf = pf_by_path.get(file_path)
            if pf is None:
                return []
            # Temporarily replace calls with only the pending ones
            import copy
            pf_subset = copy.copy(pf)
            pf_subset.calls = pending_by_file[file_path]
            try:
                return resolve_calls_with_jedi(pf_subset, repo_root)
            except Exception as e:
                logger.warning("Jedi failed for %s: %s", file_path, e)
                return []

        with ThreadPoolExecutor(max_workers=INGEST_CONCURRENCY) as jedi_pool:
            futures = [
                jedi_pool.submit(_resolve_one_file, fp)
                for fp in pending_by_file
            ]
            jedi_upgraded = 0
            for future in as_completed(futures):
                jedi_edges = [e for e in future.result() if e["resolution"] == "jedi"]
                jedi_upgraded += len(jedi_edges)
                if jedi_edges:
                    _bulk_write(graph,
                        """UNWIND $edges AS e
                           MATCH (caller:Function {fqn: e.caller_fqn})
                           MATCH (callee:Function {fqn: e.callee_fqn})
                           MERGE (caller)-[c:CALLS]->(callee)
                           SET c.line = e.line, c.file_path = e.file_path,
                               c.resolution = 'jedi'""",
                        jedi_edges, chunk_size=100, param="edges")
        logger.info("Jedi upgraded %d additional CALLS edges", jedi_upgraded)

    # --- Step 8: Generate and store Tier-0 static fingerprints ---
    # All CALLS and READS edges are now in the graph, so we can compute
    # accurate calls/reads/caller_count for each function.
    _write_fingerprints(parsed_files, graph, module_fqn_map, known_fqns)


def _write_fingerprints(
    parsed_files: list[ParsedFile],
    graph: falkordb.Graph,
    module_fqn_map: dict[str, str],
    known_fqns: set[str],
) -> None:
    """Generate and store Tier-0 static fingerprints for all ingested functions."""
    from fingerprinting import build_static_fingerprint

    # Batch-query caller counts for all known FQNs in one pass
    try:
        caller_counts: dict[str, int] = {}
        rows = graph.query(
            "MATCH ()-[:CALLS]->(f:Function) RETURN f.fqn, count(*)"
        ).result_set
        for row in rows:
            caller_counts[row[0]] = int(row[1])
    except Exception as e:
        logger.warning("Fingerprint: caller count query failed: %s", e)
        caller_counts = {}

    # Batch-query CALLS edges (callee short names) per function
    try:
        callees_map: dict[str, list[str]] = {}
        rows = graph.query(
            "MATCH (f:Function)-[:CALLS]->(t:Function) RETURN f.fqn, t.fqn"
        ).result_set
        for row in rows:
            callees_map.setdefault(row[0], []).append(row[1])
    except Exception as e:
        logger.warning("Fingerprint: callee query failed: %s", e)
        callees_map = {}

    # Batch-query READS edges per function
    try:
        reads_map: dict[str, list[str]] = {}
        rows = graph.query(
            "MATCH (f:Function)-[r:READS]->() RETURN f.fqn, r.name"
        ).result_set
        for row in rows:
            reads_map.setdefault(row[0], []).append(row[1])
    except Exception as e:
        logger.warning("Fingerprint: reads query failed: %s", e)
        reads_map = {}

    fingerprint_updates: list[dict] = []
    for pf in parsed_files:
        mod_name = module_fqn_map[pf.file_path]
        for func in pf.functions:
            fqn = f"{mod_name}.{func.qualname}" if func.qualname else f"{mod_name}.{func.name}"
            if fqn not in known_fqns:
                continue
            fp = build_static_fingerprint(
                name=func.name,
                qualname=func.qualname or func.name,
                params=func.params,
                return_annotation=func.return_annotation,
                docstring=func.docstring,
                raises=func.raises,
                calls=callees_map.get(fqn, []),
                reads=reads_map.get(fqn, []),
                caller_count=caller_counts.get(fqn, 0),
            )
            fingerprint_updates.append({"fqn": fqn, "fp": fp})

    if fingerprint_updates:
        _bulk_write(
            graph,
            "UNWIND $edges AS e MATCH (f:Function {fqn: e.fqn}) SET f.fingerprint = e.fp",
            fingerprint_updates, chunk_size=200, param="edges",
        )
        logger.info("Fingerprints written: %d functions", len(fingerprint_updates))


def run_ingestion(directory_path: str, *, on_progress=None, graph_name: str | None = None) -> dict:
    """Run full ingestion pipeline: parse, summarise, embed, and write to graph.

    Returns a summary dict with counts of all entities ingested.
    """
    dir_path = Path(directory_path).resolve()
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    graph = get_connection(graph_name)

    if FLUSH_GRAPH_ON_INGEST:
        try:
            graph.query("MATCH (n) DETACH DELETE n")
        except Exception:
            pass

    create_indices(graph)

    graph.query(
        "MERGE (m:Meta {key: 'repo_root'}) SET m.value = $root",
        {"root": str(dir_path)},
    )

    try:
        existing_states_res = graph.query("MATCH (s:FileState) RETURN s.file_path, s.mtime").result_set
        existing_states = {row[0]: row[1] for row in existing_states_res}
    except Exception:
        existing_states = {}

    files_to_parse: list[tuple[Path, str, str]] = []   # (abs_path, rel_path, content_hash)
    current_files: set[str] = set()
    INGEST_EXTENSIONS = {".py"}

    all_source_files = sorted(
        f for ext in INGEST_EXTENSIONS
        for f in dir_path.rglob(f"*{ext}")
    )
    for source_file in all_source_files:
        parts = source_file.relative_to(dir_path).parts
        if any(part in SKIP_DIRS for part in parts):
            continue

        rel_path = str(source_file.relative_to(dir_path))
        current_files.add(rel_path)
        content_hash = _file_content_hash(source_file)

        # Use content hash (not mtime) — git checkout resets mtimes on every
        # clone so mtime always differs even when file content is identical.
        if rel_path not in existing_states or existing_states[rel_path] != content_hash:
            files_to_parse.append((source_file, rel_path, content_hash))

    deleted_files = set(existing_states.keys()) - current_files
    changed_files = [f[1] for f in files_to_parse]
    files_to_delete = changed_files + list(deleted_files)

    # Batch delete instead of per-file loop (2.4)
    if files_to_delete:
        graph.query(
            "UNWIND $fps AS fp MATCH (n) WHERE n.file_path = fp DETACH DELETE n",
            {"fps": files_to_delete},
        )
        graph.query(
            "UNWIND $fps AS fp MATCH (s:FileState) WHERE s.file_path = fp DELETE s",
            {"fps": files_to_delete},
        )
    # Remove call edges pointing to functions whose files no longer exist in the graph
    try:
        graph.query("""
            MATCH (a:Function)-[r:CALLS]->(b:Function)
            OPTIONAL MATCH (fs:FileState)
            WHERE fs.file_path = b.file_path
            WITH r, fs
            WHERE fs IS NULL
            DELETE r
        """)
        logger.debug("Orphan edge pruning complete")
    except Exception as e:
        logger.warning("Orphan edge pruning failed (non-fatal): %s", e)

    if on_progress:
        on_progress("Scanning", 0, len(files_to_parse), f"{len(files_to_parse)} files found")

    parsed_with_meta: list[tuple[ParsedFile, str, str]] = []  # (parsed, rel_path, content_hash)
    for i, (py_file, rel_path, content_hash) in enumerate(files_to_parse):
        if on_progress:
            on_progress("Parsing", i + 1, len(files_to_parse), rel_path)
        try:
            parsed = parse_file(py_file, dir_path)
            parsed_with_meta.append((parsed, rel_path, content_hash))
        except Exception as e:
            logger.warning("Failed to parse %s: %s", rel_path, e)
            continue

    if on_progress:
        on_progress("Ingesting", 1, 3, "Writing graph nodes...")

    if parsed_with_meta:
        try:
            ingest_parsed_files([p for p, _, _ in parsed_with_meta], graph, dir_path, on_progress=on_progress)
        except Exception as e:
            logger.error("Ingest failed: %s", e)
            raise

        for _, rel_path, content_hash in parsed_with_meta:
            # Store content hash in the mtime field (reusing existing schema/index).
            graph.query("MERGE (s:FileState {file_path: $fp}) SET s.mtime = $h",
                        {"fp": rel_path, "h": content_hash})

    # Summary
    try:
        all_funcs = graph.query("MATCH (f:Function) RETURN count(f)").result_set[0][0]
        all_classes = graph.query("MATCH (c:Class) RETURN count(c)").result_set[0][0]
        all_modules = graph.query("MATCH (m:Module) RETURN count(m)").result_set[0][0]
        all_calls = graph.query("MATCH ()-[c:CALLS]->() RETURN count(c)").result_set[0][0]
        all_imports = graph.query("MATCH ()-[i:IMPORTS]->() RETURN count(i)").result_set[0][0]
    except Exception:
        all_funcs = all_classes = all_modules = all_calls = all_imports = 0

    return {
        "functions": all_funcs,
        "classes": all_classes,
        "modules": all_modules,
        "call_edges": all_calls,
        "import_edges": all_imports,
        "files_parsed": len(parsed_with_meta),
    }


# ---------------------------------------------------------------------------
# Jedi-based call resolution (Layer 3 precision)
# ---------------------------------------------------------------------------

try:
    import jedi as _jedi
    _JEDI_AVAILABLE = True
except ImportError:
    _JEDI_AVAILABLE = False
    logger.info("Jedi not installed; using tree-sitter fallback for call resolution")

# Shared jedi.Project cache — one project per repo_root, reused across all files
# in the same ingestion run. Creating a Project is expensive (~100ms+); sharing it
# cuts Jedi startup cost from O(files) to O(1) per repo.
_jedi_project_cache: dict[str, Any] = {}

def _get_jedi_project(repo_root: Path) -> Any:
    key = str(repo_root)
    if key not in _jedi_project_cache:
        _jedi_project_cache[key] = _jedi.Project(path=key)
    return _jedi_project_cache[key]


JEDI_FILE_TIMEOUT = int(_os.getenv("JEDI_FILE_TIMEOUT", "5"))
def resolve_calls_with_jedi(
    parsed_file: "ParsedFile",
    repo_root: Path,
) -> list[dict]:
    """Wrap jedi resolution with a hard per-file timeout."""
    timeout = JEDI_FILE_TIMEOUT  # was hardcoded to 300, ignoring the env var
    try:
        with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
            fut = _ex.submit(_resolve_calls_with_jedi_inner, parsed_file, repo_root)
            return fut.result(timeout=timeout)
    except _cf.TimeoutError:
        logger.debug("Jedi timed out on %s after %ds — skipping", parsed_file.file_path, timeout)
        return []
    except Exception as e:
        logger.debug("Jedi failed on %s: %s", parsed_file.file_path, e)
        return []

def _resolve_calls_with_jedi_inner(
    parsed_file: "ParsedFile",
    repo_root: Path,
) -> list[dict]:
    """Upgrade fuzzy call edges to precise ones using Jedi type inference.

    Returns a list of dicts with caller_fqn, callee_fqn, resolution method, etc.
    Falls back to tree-sitter name matching if Jedi can't resolve.
    """
    if not _JEDI_AVAILABLE:
        return []

    file_path = repo_root / parsed_file.file_path
    if not file_path.exists() or file_path.suffix != ".py":
        return []

    # Skip test files — they rarely contain the target code, and Jedi spends
    # most of its time resolving mock/fixture chains that don't become graph edges.
    rel_parts = Path(parsed_file.file_path).parts
    if any(p in ("tests", "test", "testing") or p.startswith("test_")
           for p in rel_parts):
        return []

    try:
        source = file_path.read_text(encoding="utf-8")
        project = _get_jedi_project(repo_root)   # cached — not recreated per file
        script = _jedi.Script(source, path=str(file_path), project=project)
    except Exception as e:
        logger.debug("Jedi init failed for %s: %s", parsed_file.file_path, e)
        return []

    mod_name = _file_to_module(parsed_file.file_path)
    precise_edges = []

    for call in parsed_file.calls:
        # caller_qualname replaces the old caller_name field
        caller_fqn = f"{mod_name}.{call.caller_qualname}"
        resolution = "tree-sitter"
        callee_fqn = call.callee_expr   # full expression, e.g. "self.connect"

        try:
            defs = script.goto(line=call.line, column=call.column)
            if defs:
                target = defs[0]
                if target.full_name:
                    callee_fqn = target.full_name
                    resolution = "jedi"
        except Exception:
            pass

        precise_edges.append({
            "caller_fqn": caller_fqn,
            "callee_fqn": callee_fqn,
            "resolution": resolution,
            "line": call.line,
            "file_path": call.file_path,
        })

    return precise_edges


# ---------------------------------------------------------------------------
# Fix A helpers — inherited method resolution
# ---------------------------------------------------------------------------

def _build_parent_map(inherits_resolved: list[dict]) -> dict[str, list[str]]:
    """Build class_fqn → [direct_parent_fqns, ...] from INHERITS_FROM edge list."""
    from collections import defaultdict as _dd
    parents: dict[str, list[str]] = _dd(list)
    for e in inherits_resolved:
        parents[e["cfqn"]].append(e["base_fqn"])
    return parents


def _resolve_via_inheritance(
    fqn: str,
    parent_map: dict[str, list[str]],
    known_fqns: set[str],
    _depth: int = 0,
) -> Optional[str]:
    """Walk the inheritance chain to find the method on an ancestor class.

    Example: Dog.breathe not in known_fqns → check Animal.breathe → found.
    Depth-limited to 8 to guard against circular inheritance.
    """
    if _depth > 8:
        return None
    if fqn in known_fqns:
        return fqn
    parts = fqn.rsplit(".", 1)
    if len(parts) < 2:
        return None
    class_fqn, method_name = parts
    for parent_fqn in parent_map.get(class_fqn, []):
        candidate = f"{parent_fqn}.{method_name}"
        result = _resolve_via_inheritance(candidate, parent_map, known_fqns, _depth + 1)
        if result:
            return result
    return None


# ---------------------------------------------------------------------------
# Incremental re-ingestion (for post-edit graph refresh)
# ---------------------------------------------------------------------------

def _write_call_and_reads_edges(
    parsed_files: list[ParsedFile],
    graph: falkordb.Graph,
    module_fqn_map: dict[str, str],
    symbol_tables: dict,
    reexport_map: dict[str, str],
    parent_map: dict[str, list[str]],
    known_fqns: set[str],
) -> list[tuple]:
    """Resolve and write CALLS + READS edges for the given parsed files.

    Returns the list of (ParsedFile, CallEdge) pairs that could not be resolved
    by the symbol table (pending for Jedi). In edges-only mode the caller
    discards this — Jedi is only run in the full ingestion path.
    """
    # --- CALLS edges ---
    resolved_call_edges: list[dict] = []
    pending_calls: list[tuple] = []

    for pf in parsed_files:
        mod_name = module_fqn_map[pf.file_path]
        table = symbol_tables[pf.file_path]

        for call in pf.calls:
            caller_fqn = f"{mod_name}.{call.caller_qualname}"
            callee_fqn = resolve_callee(call.callee_expr, call.caller_qualname, table, known_fqns)
            if callee_fqn and reexport_map:
                callee_fqn = canonicalize_fqn(callee_fqn, reexport_map)
            if callee_fqn and callee_fqn not in known_fqns and parent_map:
                inherited = _resolve_via_inheritance(callee_fqn, parent_map, known_fqns)
                if inherited:
                    callee_fqn = inherited
            if callee_fqn:
                resolved_call_edges.append({
                    "caller_fqn": caller_fqn,
                    "callee_fqn": callee_fqn,
                    "resolution": "symbol-table",
                    "line": call.line,
                    "file_path": call.file_path,
                })
            else:
                pending_calls.append((pf, call))

    if resolved_call_edges:
        _bulk_write(graph,
            """UNWIND $edges AS e
               MATCH (caller:Function {fqn: e.caller_fqn})
               MATCH (callee:Function {fqn: e.callee_fqn})
               MERGE (caller)-[c:CALLS]->(callee)
               SET c.line = e.line, c.file_path = e.file_path,
                   c.resolution = e.resolution""",
            resolved_call_edges, chunk_size=300, param="edges")
    logger.info(
        "CALLS: %d resolved via symbol-table, %d pending for Jedi",
        len(resolved_call_edges), len(pending_calls),
    )

    # --- READS edges ---
    reads_edges: list[dict] = []
    seen_reads: set[tuple] = set()

    for pf in parsed_files:
        mod_name = module_fqn_map[pf.file_path]
        table = symbol_tables[pf.file_path]

        for ref in pf.variable_refs:
            reader_fqn = f"{mod_name}.{ref.user_qualname}"
            if reader_fqn not in known_fqns:
                continue

            if "." in ref.name:
                head, attr_name = ref.name.split(".", 1)
                resolved_head = table.resolve(head)
                if resolved_head is None or resolved_head in known_fqns:
                    continue
                source_module = resolved_head
                if source_module == mod_name:
                    continue
                key = (reader_fqn, source_module, attr_name)
                if key not in seen_reads:
                    seen_reads.add(key)
                    reads_edges.append({"reader_fqn": reader_fqn,
                                        "source_module": source_module,
                                        "name": attr_name, "line": ref.line})
            else:
                resolved = table.resolve(ref.name)
                if resolved is None:
                    continue
                dot_pos = resolved.rfind(".")
                if dot_pos < 0:
                    continue
                source_module = resolved[:dot_pos]
                if source_module == mod_name:
                    continue
                key = (reader_fqn, source_module, ref.name)
                if key not in seen_reads:
                    seen_reads.add(key)
                    reads_edges.append({"reader_fqn": reader_fqn,
                                        "source_module": source_module,
                                        "name": ref.name, "line": ref.line})

    if reads_edges:
        _bulk_write(graph,
            """UNWIND $edges AS e
               MATCH (f:Function {fqn: e.reader_fqn})
               MATCH (m:Module {name: e.source_module})
               MERGE (f)-[r:READS]->(m)
               SET r.name = e.name, r.line = e.line""",
            reads_edges, chunk_size=200, param="edges")
        logger.info("READS edges written: %d", len(reads_edges))

    return pending_calls


def _find_importer_files(changed_modules: list[str], graph: falkordb.Graph) -> list[str]:
    """Return file_paths of all modules that IMPORT any of the changed modules.

    Used by reingest_files to find files whose outbound CALLS/READS edges may
    point to stale FQNs after a dependency changed.
    """
    if not changed_modules:
        return []
    try:
        result = graph.query(
            """UNWIND $mods AS m
               MATCH (src:Module)-[:IMPORTS]->(tgt:Module {name: m})
               WHERE src.file_path <> ''
               RETURN DISTINCT src.file_path""",
            {"mods": changed_modules},
        )
        return [row[0] for row in result.result_set]
    except Exception as e:
        logger.warning("_find_importer_files failed (non-fatal): %s", e)
        return []

def reingest_files(
    file_paths: list[str],
    graph: falkordb.Graph,
    repo_root: Path,
) -> dict:
    """Re-ingest specific files into the graph after edits.

    Used by Phase 5 of the change engine to update the graph after
    applying changes, so blast radius can be re-checked.

    Args:
        file_paths: Relative file paths to re-ingest.
        graph: FalkorDB graph connection.
        repo_root: Absolute path to the repository root.

    Returns:
        Dict with counts of re-ingested entities.
    """
    # Delete old nodes for these files
    if file_paths:
        graph.query(
            "UNWIND $fps AS fp MATCH (n) WHERE n.file_path = fp DETACH DELETE n",
            {"fps": file_paths},
        )

    # Re-parse and ingest
    parsed_files: list[ParsedFile] = []
    for rel_path in file_paths:
        abs_path = repo_root / rel_path
        if not abs_path.exists():
            continue
        try:
            parsed = parse_file(abs_path, repo_root)
            parsed_files.append(parsed)
        except Exception as e:
            logger.warning("Re-ingest parse failed for %s: %s", rel_path, e)

    if parsed_files:
        ingest_parsed_files(parsed_files, graph, repo_root)

        # Store content hash (same scheme as run_ingestion) so the watcher
        # can skip files that haven't changed since the last ingest.
        for pf in parsed_files:
            abs_path = repo_root / pf.file_path
            content_hash = _file_content_hash(abs_path)
            graph.query(
                "MERGE (s:FileState {file_path: $fp}) SET s.mtime = $h",
                {"fp": pf.file_path, "h": content_hash},
            )

    # Fix 3: cascade edge re-resolution — find all files that import from the
    # changed modules and re-resolve their CALLS/READS edges so they don't
    # point to stale FQNs.  We skip summaries/embeddings (those are stable).
    changed_modules = [_file_to_module(fp) for fp in file_paths]
    importer_paths = [
        fp for fp in _find_importer_files(changed_modules, graph)
        if fp not in file_paths   # already re-ingested above
    ]
    if importer_paths:
        logger.info(
            "reingest_files: cascade re-resolving edges for %d importer file(s)",
            len(importer_paths),
        )
        # Delete only CALLS and READS edges — nodes are unchanged
        graph.query(
            """UNWIND $fps AS fp
               MATCH (f:Function {file_path: fp})-[r:CALLS|READS]->()
               DELETE r""",
            {"fps": importer_paths},
        )
        # Parse importer files (fast — tree-sitter only, no LLM/embeddings)
        importer_parsed: list[ParsedFile] = []
        for rel_path in importer_paths:
            abs_path = repo_root / rel_path
            if not abs_path.exists():
                continue
            try:
                importer_parsed.append(parse_file(abs_path, repo_root))
            except Exception as e:
                logger.warning("Cascade re-parse failed for %s: %s", rel_path, e)

        if importer_parsed:
            # Build combined symbol tables (changed files + importers) so that
            # relative imports in importer files resolve against the new module FQNs.
            all_for_resolution = parsed_files + importer_parsed
            imp_fqn_map = build_module_fqn_map(all_for_resolution)
            imp_tables = {
                pf.file_path: build_symbol_table(
                    pf, imp_fqn_map[pf.file_path], imp_fqn_map
                )
                for pf in all_for_resolution
            }
            imp_reexport = build_reexport_map(all_for_resolution, imp_fqn_map)
            enrich_star_imports(all_for_resolution, imp_tables, imp_fqn_map)

            # known_fqns from graph (nodes are already written — no need to re-write them)
            imp_known: set[str] = set()
            for row in graph.query("MATCH (f:Function) RETURN f.fqn").result_set:
                imp_known.add(row[0])
            for row in graph.query("MATCH (c:Class) RETURN c.fqn").result_set:
                imp_known.add(row[0])

            # parent_map from graph (INHERITS_FROM edges already present)
            inh_rows = graph.query(
                "MATCH (c:Class)-[:INHERITS_FROM]->(b:Class) RETURN c.fqn, b.fqn"
            ).result_set
            imp_parent_map = _build_parent_map(
                [{"cfqn": r[0], "base_fqn": r[1]} for r in inh_rows]
            )

            # Write only CALLS and READS edges — skip all node writes and embeddings
            _write_call_and_reads_edges(
                importer_parsed, graph, imp_fqn_map, imp_tables,
                imp_reexport, imp_parent_map, imp_known,
            )

    return {
        "files_reingested": len(parsed_files),
        "cascade_files": len(importer_paths),
        "file_paths": file_paths,
    }

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    if len(sys.argv) < 2:
        print("Usage: python3 ingest.py <directory_path>")
        sys.exit(1)
        
    target_dir = sys.argv[1]
    logger.info(f"Starting graph ingestion for directory: {target_dir}")
    
    try:
        stats = run_ingestion(target_dir)
        logger.info("Ingestion completed successfully!")
        print(json.dumps(stats, indent=2))
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)
