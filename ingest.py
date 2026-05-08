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
import falkordb
import openai
import os as _os
SKIP_SUMMARIES = _os.getenv("SKIP_SUMMARIES", "false").lower() in ("true", "1")
from config import (FALKORDB_HOST, FALKORDB_PORT, GRAPH_NAME,
                    FLUSH_GRAPH_ON_INGEST, SGLANG_BASE_URL, SGLANG_API_KEY, LLM_MODEL,
                    INGEST_CONCURRENCY)
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
_NO_THINK = {"chat_template_kwargs":{"enable_thinking":False}}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_to_module(file_path: str) -> str:
    """Convert a relative file path like 'foo/bar.py' to module name 'foo.bar'."""
    mod = file_path.replace("/", ".").replace("\\", ".")
    return mod[:-3] if mod.endswith(".py") else mod


# Singleton OpenAI client for summary generation (avoids hundreds of TCP connections)
_summary_client: openai.OpenAI | None = None


def _get_summary_client() -> openai.OpenAI:
    """Lazy-initialise a single OpenAI client for the summary generator."""
    global _summary_client
    if _summary_client is None:
        _summary_client = openai.OpenAI(base_url=SGLANG_BASE_URL, api_key=SGLANG_API_KEY)
    return _summary_client


# ---------------------------------------------------------------------------
# Connection with retry
# ---------------------------------------------------------------------------

_MAX_CONNECT_RETRIES = 3
_CONNECT_BACKOFF_BASE = 1.0  # seconds


def get_connection() -> falkordb.Graph:
    """Connect to FalkorDB with exponential-backoff retry."""
    last_err: Exception | None = None
    for attempt in range(_MAX_CONNECT_RETRIES):
        try:
            db = falkordb.FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
            return db.select_graph(GRAPH_NAME)
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


def create_indices(graph: falkordb.Graph) -> None:
    """Create FalkorDB indices for fast lookups. Idempotent."""
    index_queries = [
        "CREATE INDEX FOR (f:Function) ON (f.fqn)",
        "CREATE INDEX FOR (c:Class) ON (c.fqn)",
        "CREATE INDEX FOR (m:Module) ON (m.name)",
        "CREATE INDEX FOR (s:FileState) ON (s.file_path)",
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

def generate_summaries_batch(batch: list[tuple[str, str, str, str]]) -> dict[str, str]:
    """Generate 1-2 sentence AI summaries for a batch of functions/classes.
    batch: list of (id_string, sig_or_code, docstring, has_docstring)
    Returns: dict mapping id_string to summary
    """
    if not batch:
        return {}
    try:
        client = _get_summary_client()
        prompt_parts = [
            "Summarize what each of these functions/classes does in 1-2 sentences.\n"
            "Return ONLY a valid JSON object mapping the ID to the summary string.\n"
            "No markdown, no explanation, no code fences.\n\n"
        ]
        for idx, content, docstring, has_docstring in batch:
            if has_docstring:
                # Docstring present — signature + docstring is sufficient
                prompt_parts.append(f"--- ID: {idx} ---\n{content}\nDocstring: {docstring[:300]}\n")
            else:
                # No docstring — send truncated code body for accuracy
                prompt_parts.append(f"--- ID: {idx} ---\n{content[:600]}\n")

        prompt = "".join(prompt_parts)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a code summarizer. Return ONLY valid JSON. No markdown fences, no explanation, no thinking. Output must be a single JSON object.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
            temperature=0,
            response_format={"type": "json_object"},
            extra_body=_NO_THINK,
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            logger.error("Batch summary: empty model content")
            return {}
        raw = content.strip()
        logger.debug("Batch summary raw content (first 300): %s", raw[:300])
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if raw.startswith("```"):
                raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(raw)
    except Exception as e:
        logger.error("Batch summary generation failed: %s", e)
        return {}


def extract_source_code(file_path: Path, start_line: int, end_line: int) -> str:
    """Read source lines from a file. Returns empty string on failure."""
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[start_line - 1 : end_line])
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------
def _bulk_write(graph: falkordb.Graph, query: str, nodes: list[dict], chunk_size: int = 500) -> None:
    """Write nodes in chunks to avoid FalkorDB timeouts on large UNWIND."""
    for i in range(0, len(nodes), chunk_size):
        graph.query(query, {"nodes": nodes[i:i + chunk_size]})
def ingest_parsed_files(
    parsed_files: list[ParsedFile],
    graph: falkordb.Graph,
    repo_root: Path,
    on_progress=None,
) -> None:
    """Write parsed AST entities into FalkorDB as nodes and edges."""
    file_modules: dict[str, str] = {}
    import_modules: set[str] = set()

    for pf in parsed_files:
        mod_name = _file_to_module(pf.file_path)
        file_modules[mod_name] = pf.file_path

        for imp in pf.imports:
            import_modules.add(imp.module)

    # Step 1: Upsert Module nodes
    mod_emb_texts = [f"{mod_name}. Module defined in {fpath}" 
                     for mod_name, fpath in file_modules.items()]
    mod_embeddings = embed_texts(mod_emb_texts) if mod_emb_texts else []
    
    module_nodes = []
    mod_items = list(file_modules.items())
    for i, (mod_name, fpath) in enumerate(mod_items):
        module_nodes.append({
            "name": mod_name, 
            "file_path": fpath,
            "embedding": json.dumps(mod_embeddings[i]) if mod_embeddings else None
        })

    for mod_name in import_modules:
        if mod_name and mod_name not in file_modules:
            module_nodes.append({"name": mod_name, "file_path": "", "embedding": None})

    module_nodes=[n for n in module_nodes if n.get("name") and n["name"].strip() not in ("",".")]        
    if module_nodes:
        _bulk_write(graph,
            """UNWIND $nodes AS n
               MERGE (m:Module {name: n.name})
               SET m.file_path = n.file_path
               WITH m, n WHERE n.file_path <> "" AND n.embedding IS NOT NULL
               SET m.embedding = n.embedding""",
            module_nodes,            
        )

    # Prepare for AI summaries
    items_to_summarize = []
    for pf in parsed_files:
        mod_name = _file_to_module(pf.file_path)
            
        abs_path = repo_root / pf.file_path
        for cls in pf.classes:
            code = extract_source_code(abs_path, cls.start_line, cls.end_line)
            items_to_summarize.append(("class", cls, pf.file_path, mod_name, code))
        for func in pf.functions:
            code = extract_source_code(abs_path, func.start_line, func.end_line)
            items_to_summarize.append(("function", func, pf.file_path, mod_name, code))

    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    
    summaries: dict[str, str] = {}

    if SKIP_SUMMARIES:
        logger.info("SKIP_SUMMARIES=true — skipping LLM summarization")
        for item in items_to_summarize:
            idx = f"{item[2]}::{item[1].name}::{item[1].start_line}"
            summaries[idx] = ""
        batches = []
    else:
        BATCH_SIZE = 60  # raised from 15 — adaptive input keeps tokens manageable
        batches = []
        current_batch = []

        for item in items_to_summarize:
            kind, entity, file_path, mod_name, code = item
            idx = f"{file_path}::{entity.name}::{entity.start_line}"
            docstring = (entity.docstring or "").strip()
            has_docstring = len(docstring) > 10

            if has_docstring:
                # Signature + docstring — compact, accurate
                if hasattr(entity, "params") and entity.params:
                    params = ",".join(entity.params)
                    sig = f"{entity.name}({params})"
                else:
                    sig = entity.name
                content=sig
            else:
                # No docstring — need code body
                content = code

            if len(code.strip()) > 0:
                current_batch.append((idx, content, docstring, has_docstring))
                if len(current_batch) >= BATCH_SIZE:
                    batches.append(current_batch)
                    current_batch = []
            else:
                summaries[idx] = ""

        if current_batch:
            batches.append(current_batch)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task_id = progress.add_task("[cyan]Generating AI Node Summaries...", total=len(batches))
        with ThreadPoolExecutor(max_workers=INGEST_CONCURRENCY) as executor:
            future_to_batch = {
                executor.submit(generate_summaries_batch, batch): batch
                for batch in batches
            }
            for future in as_completed(future_to_batch):
                batch_result = future.result()
                if batch_result:
                    for idx_str, summary in batch_result.items():
                        summaries[idx_str] = summary
                progress.advance(task_id)

    # Step 2: Upsert Class nodes
    class_nodes: list[dict] = []
    class_emb_texts: list[str] = []
    for item in items_to_summarize:
        if item[0] != "class": continue
        cls, file_path, mod_name, code = item[1], item[2], item[3], item[4]
        summary = summaries.get(f"{file_path}::{cls.name}::{cls.start_line}", "")
        emb_text = build_embedding_text(cls.name, cls.docstring, file_path)
        class_emb_texts.append(emb_text)
        fqn = f"{mod_name}.{cls.name}"
        class_nodes.append({
            "fqn": fqn, "name": cls.name, "file_path": file_path,
            "start_line": cls.start_line, "end_line": cls.end_line,
            "docstring": cls.docstring or "", "summary": summary,
        })
        
    if on_progress:
        on_progress("Embeddings", 2, 3, "Generating vectors...")

    if class_emb_texts:
        class_embeddings = embed_texts(class_emb_texts)
        for i, node in enumerate(class_nodes):
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

    # Step 3: Upsert Function nodes
    func_nodes: list[dict] = []
    func_emb_texts: list[str] = []
    for item in items_to_summarize:
        if item[0] != "function": continue
        func, file_path, mod_name, code = item[1], item[2], item[3], item[4]
        summary = summaries.get(f"{file_path}::{func.name}::{func.start_line}", "")
        emb_text = build_embedding_text(func.name, func.docstring, file_path)
        func_emb_texts.append(emb_text)
        fqn = f"{mod_name}.{func.class_name}.{func.name}" if func.is_method else f"{mod_name}.{func.name}"
        func_nodes.append({
            "fqn": fqn, "name": func.name, "file_path": file_path,
            "start_line": func.start_line, "end_line": func.end_line,
            "docstring": func.docstring or "", "is_method": func.is_method,
            "class_name": func.class_name or "", "module_name": mod_name,
            "summary": summary,
            "params": json.dumps(func.params),
            "decorators": json.dumps(func.decorators),
            "return_annotation": func.return_annotation or "",
        })
        
    if func_emb_texts:
        func_embeddings = embed_texts(func_emb_texts)
        for i, node in enumerate(func_nodes):
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
                   f.return_annotation = n.return_annotation""",
            func_nodes,
        )

    # Step 4: Create DEFINED_IN and INHERITS_FROM edges
    func_to_class: list[dict] = []
    func_to_mod: list[dict] = []
    class_to_mod: list[dict] = []
    inherits_edges: list[dict] = []
    
    for pf in parsed_files:
        mod_name = _file_to_module(pf.file_path)

        for func in pf.functions:
            fqn = f"{mod_name}.{func.class_name}.{func.name}" if func.is_method else f"{mod_name}.{func.name}"
            if func.is_method:
                class_fqn = f"{mod_name}.{func.class_name}"
                func_to_class.append({"fqn": fqn, "cfqn": class_fqn})
            else:
                func_to_mod.append({"fqn": fqn, "mname": mod_name})
                
        for cls in pf.classes:
            class_fqn = f"{mod_name}.{cls.name}"
            class_to_mod.append({"cfqn": class_fqn, "mname": mod_name})
            for base in cls.bases:
                inherits_edges.append({"cfqn": class_fqn, "base_name": base})

    if func_to_class:
        graph.query(
            """UNWIND $edges AS e
               MATCH (f:Function {fqn: e.fqn})
               MATCH (c:Class {fqn: e.cfqn})
               MERGE (f)-[:DEFINED_IN]->(c)""",
            {"edges": func_to_class},
        )
    if func_to_mod:
        graph.query(
            """UNWIND $edges AS e
               MATCH (f:Function {fqn: e.fqn})
               MATCH (m:Module {name: e.mname})
               MERGE (f)-[:DEFINED_IN]->(m)""",
            {"edges": func_to_mod},
        )
    if class_to_mod:
        graph.query(
            """UNWIND $edges AS e
               MATCH (c:Class {fqn: e.cfqn})
               MATCH (m:Module {name: e.mname})
               MERGE (c)-[:DEFINED_IN]->(m)""",
            {"edges": class_to_mod},
        )
    if inherits_edges:
        graph.query(
            """UNWIND $edges AS e
               MATCH (c:Class {fqn: e.cfqn})
               MATCH (base:Class {name: e.base_name})
               MERGE (c)-[:INHERITS_FROM]->(base)""",
            {"edges": inherits_edges},
        )

    # Step 5: Create IMPORTS edges
    import_edges: list[dict] = []
    for pf in parsed_files:
        src_mod = _file_to_module(pf.file_path)

        for imp in pf.imports:
            import_edges.append({
                "src_name": src_mod,
                "tgt_name": imp.module,
                "alias": imp.alias or "",
            })
    import_edges = [e for e in import_edges if e.get("src_name") and e.get("tgt_name") and e["tgt_name"] != "."]       
    if import_edges:
        graph.query(
            """UNWIND $edges AS e
               MATCH (src:Module {name: e.src_name})
               MATCH (tgt:Module {name: e.tgt_name})
               MERGE (src)-[i:IMPORTS]->(tgt)
               SET i.alias = e.alias""",
            {"edges": import_edges},
        )

    # Step 6: Create CALLS edges
    # Build a set of known imported modules per source module for scoped matching
    module_imports: dict[str, set[str]] = {}
    for pf in parsed_files:
        src_mod = _file_to_module(pf.file_path)
        module_imports[src_mod] = {imp.module for imp in pf.imports}

    call_edges: list[dict] = []
    for pf in parsed_files:
        mod_name = _file_to_module(pf.file_path)
            
        for call in pf.calls:
            if call.caller_name == "__classbody__" or call.caller_name == "<module>":
                continue  

            caller_fqn = f"{mod_name}.{call.caller_name}"
            callee_simple = call.callee_name.split(".")[-1] if "." in call.callee_name else call.callee_name
            
            # Build scoped module list: same module + imported modules
            scope_modules = [mod_name] + list(module_imports.get(mod_name, set()))

            call_edges.append({
                "caller_fqn": caller_fqn,
                "callee_name": "." + callee_simple,
                "callee_exact": callee_simple,
                "scope_modules": scope_modules,
                "line": call.line,
                "file_path": call.file_path,
            })
            
    if call_edges:
        graph.query(
            """UNWIND $edges AS e
               MATCH (caller:Function {fqn: e.caller_fqn})
               MATCH (callee:Function)
               WHERE (callee.fqn ENDS WITH e.callee_name OR callee.fqn = e.callee_exact)
                 AND callee.module_name IN e.scope_modules
               MERGE (caller)-[c:CALLS]->(callee)
               SET c.line = e.line, c.file_path = e.file_path, c.resolution = 'tree-sitter'""",
            {"edges": call_edges},
        )
    # Step 7: Jedi precision pass — parallelized across files
    jedi_upgraded = 0
    jedi_total = 0

    def _resolve_one_file(pf):
        if not pf.file_path.endswith(".py"):
            return []
        try:
            return resolve_calls_with_jedi(pf, repo_root)
        except Exception as e:
            logger.warning("Jedi failed for %s: %s", pf.file_path, e)
            return []

    with ThreadPoolExecutor(max_workers=INGEST_CONCURRENCY) as jedi_pool:
        futures = [jedi_pool.submit(_resolve_one_file, pf) for pf in parsed_files]
        for future in as_completed(futures):
            precise_edges = future.result()
            jedi_total += len(precise_edges)
            jedi_edges = [e for e in precise_edges if e["resolution"] == "jedi"]
            jedi_upgraded += len(jedi_edges)
            if jedi_edges:
                graph.query(
                    """UNWIND $edges AS e
                    MATCH (caller:Function {fqn: e.caller_fqn})
                    MATCH (callee:Function {fqn: e.callee_fqn})
                    MERGE (caller)-[c:CALLS]->(callee)
                    SET c.line = e.line, c.file_path = e.file_path, c.resolution = 'jedi'""",
                    {"edges": jedi_edges},
                )


def run_ingestion(directory_path: str, *, on_progress=None) -> dict:
    """Run full ingestion pipeline: parse, summarise, embed, and write to graph.

    Returns a summary dict with counts of all entities ingested.
    """
    dir_path = Path(directory_path).resolve()
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    graph = get_connection()

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

    files_to_parse: list[tuple[Path, str, float]] = []
    current_files: set[str] = set()
    # Supported languages. Architecture is language-agnostic (tree-sitter);
    # expand this set as precision resolvers are added per language.
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
        mtime = source_file.stat().st_mtime

        # Only parse if file changed or if FLUSH_GRAPH_ON_INGEST was True (which wipes existing_states)
        if rel_path not in existing_states or existing_states[rel_path] != mtime:
            files_to_parse.append((source_file, rel_path, mtime))

    deleted_files = set(existing_states.keys()) - current_files
    changed_files = [f[1] for f in files_to_parse]
    files_to_delete = changed_files + list(deleted_files)

    # Batch delete instead of per-file loop (2.4)
    if files_to_delete:
        graph.query(
            "UNWIND $fps AS fp MATCH (n {file_path: fp}) DETACH DELETE n",
            {"fps": files_to_delete},
        )
        graph.query(
            "UNWIND $fps AS fp MATCH (s:FileState {file_path: fp}) DELETE s",
            {"fps": files_to_delete},
        )
    # Remove call edges pointing to functions whose files no longer exist in the graph
    try:
        graph.query("""
            MATCH (a:Function)-[r:CALLS]->(b:Function)
            WHERE NOT EXISTS {
                MATCH (fs:FileState {file_path: b.file_path})
            }
            DELETE r
        """)
        logger.debug("Orphan edge pruning complete")
    except Exception as e:
        logger.warning("Orphan edge pruning failed (non-fatal): %s", e)

    if on_progress:
        on_progress("Scanning", 0, len(files_to_parse), f"{len(files_to_parse)} files found")

    parsed_with_meta: list[tuple[ParsedFile, str, float]] = []
    for i, (py_file, rel_path, mtime) in enumerate(files_to_parse):
        if on_progress:
            on_progress("Parsing", i + 1, len(files_to_parse), rel_path)
        try:
            parsed = parse_file(py_file, dir_path)
            parsed_with_meta.append((parsed, rel_path, mtime))
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

        for _, rel_path, mtime in parsed_with_meta:
            graph.query("MERGE (s:FileState {file_path: $fp}) SET s.mtime = $mtime", {"fp": rel_path, "mtime": mtime})

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


def resolve_calls_with_jedi(
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

    try:
        source = file_path.read_text(encoding="utf-8")
        project = _jedi.Project(path=str(repo_root))
        script = _jedi.Script(source, path=str(file_path), project=project)
    except Exception as e:
        logger.debug("Jedi init failed for %s: %s", parsed_file.file_path, e)
        return []

    mod_name = _file_to_module(parsed_file.file_path)
    precise_edges = []

    for call in parsed_file.calls:
        if call.caller_name in ("__classbody__", "<module>"):
            continue

        caller_fqn = f"{mod_name}.{call.caller_name}"
        resolution = "tree-sitter"
        callee_fqn = call.callee_name

        try:
            # Jedi goto resolves to the definition
            defs = script.goto(line=call.line, column=call.column)
            if defs:
                target = defs[0]
                if target.full_name:
                    callee_fqn = target.full_name
                    resolution = "jedi"
        except Exception:
            pass  # Fall through to tree-sitter

        precise_edges.append({
            "caller_fqn": caller_fqn,
            "callee_fqn": callee_fqn,
            "resolution": resolution,
            "line": call.line,
            "file_path": call.file_path,
        })

    return precise_edges


# ---------------------------------------------------------------------------
# Incremental re-ingestion (for post-edit graph refresh)
# ---------------------------------------------------------------------------

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
            "UNWIND $fps AS fp MATCH (n {file_path: fp}) DETACH DELETE n",
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

        # Update FileState mtime so the watcher knows this file is fresh
        import time as _time
        for pf in parsed_files:
            abs_path = repo_root / pf.file_path
            try:
                mtime = abs_path.stat().st_mtime
            except OSError:
                mtime = 0.0
            graph.query(
                """MERGE (fs:FileState {file_path: $path})
                   SET fs.mtime = $mtime, fs.last_ingested = $ts""",
                {"path": pf.file_path, "mtime": mtime, "ts": _time.time()},
            )

    return {
        "files_reingested": len(parsed_files),
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
