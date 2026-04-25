# ingest.py
"""
Take parsed data and write it into FalkorDB.
Uses MERGE (not CREATE) for node upserts. Idempotent within a run.
Optionally flushes graph before run based on config.FLUSH_GRAPH_ON_INGEST.
"""

import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import falkordb
import openai
from config import (FALKORDB_HOST, FALKORDB_PORT, GRAPH_NAME,
                    FLUSH_GRAPH_ON_INGEST, SGLANG_BASE_URL, SGLANG_API_KEY, LLM_MODEL,
                    INGEST_CONCURRENCY)
from parser import ParsedFile, parse_file, SKIP_DIRS
from embedder import embed_text, embed_texts, build_embedding_text

def get_connection() -> falkordb.Graph:
    try:
        db = falkordb.FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
        return db.select_graph(GRAPH_NAME)
    except Exception as e:
        raise ConnectionError(f"Failed to connect to FalkorDB at {FALKORDB_HOST}:{FALKORDB_PORT}: {e}")

def create_indices(graph: falkordb.Graph) -> None:
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

def generate_summary(name: str, code: str) -> str:
    if not code.strip():
        return ""
    try:
        client = openai.OpenAI(base_url=SGLANG_BASE_URL, api_key=SGLANG_API_KEY)
        prompt = f"Summarize what this Python function/class does in 1-2 sentences. Name: {name}\nCode:\n{code}"
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Summary unavailable: {e}"

def extract_source_code(file_path: Path, start_line: int, end_line: int) -> str:
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[start_line - 1 : end_line])
    except Exception:
        return ""

def ingest_parsed_files(parsed_files: list[ParsedFile], graph: falkordb.Graph, repo_root: Path) -> None:
    file_modules: dict[str, str] = {}
    import_modules: set[str] = set()

    for pf in parsed_files:
        mod_name = pf.file_path.replace("/", ".").replace("\\", ".")
        if mod_name.endswith(".py"):
            mod_name = mod_name[:-3]
        file_modules[mod_name] = pf.file_path

        for imp in pf.imports:
            import_modules.add(imp.module)

    # Step 1: Upsert Module nodes
    module_nodes = [{"name": mod_name, "file_path": fpath} for mod_name, fpath in file_modules.items()]
    for mod_name in import_modules:
        if mod_name not in file_modules:
            module_nodes.append({"name": mod_name, "file_path": ""})
            
    if module_nodes:
        graph.query(
            """UNWIND $nodes AS n
               MERGE (m:Module {name: n.name})
               SET m.file_path = n.file_path""",
            {"nodes": module_nodes},
        )

    # Prepare for AI summaries
    items_to_summarize = []
    for pf in parsed_files:
        mod_name = pf.file_path.replace("/", ".").replace("\\", ".")
        if mod_name.endswith(".py"):
            mod_name = mod_name[:-3]
            
        abs_path = repo_root / pf.file_path
        for cls in pf.classes:
            code = extract_source_code(abs_path, cls.start_line, cls.end_line)
            items_to_summarize.append(("class", cls, pf.file_path, mod_name, code))
        for func in pf.functions:
            code = extract_source_code(abs_path, func.start_line, func.end_line)
            items_to_summarize.append(("function", func, pf.file_path, mod_name, code))

    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    
    summaries = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task_id = progress.add_task("[cyan]Generating AI Node Summaries...", total=len(items_to_summarize))
        with ThreadPoolExecutor(max_workers=INGEST_CONCURRENCY) as executor:
            future_to_item = {
                executor.submit(generate_summary, item[1].name, item[4]): item
                for item in items_to_summarize
            }
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    summaries[id(item[1])] = future.result()
                except Exception:
                    summaries[id(item[1])] = ""
                progress.advance(task_id)

    # Step 2: Upsert Class nodes
    class_nodes = []
    class_emb_texts = []
    for item in items_to_summarize:
        if item[0] != "class": continue
        cls, file_path, mod_name, code = item[1], item[2], item[3], item[4]
        summary = summaries.get(id(cls), "")
        emb_text = build_embedding_text(cls.name, cls.docstring, file_path)
        class_emb_texts.append(emb_text)
        fqn = f"{mod_name}.{cls.name}"
        class_nodes.append({
            "fqn": fqn, "name": cls.name, "file_path": file_path,
            "start_line": cls.start_line, "end_line": cls.end_line,
            "docstring": cls.docstring or "", "summary": summary,
        })
        
    if class_emb_texts:
        class_embeddings = embed_texts(class_emb_texts)
        for i, node in enumerate(class_nodes):
            node["embedding"] = json.dumps(class_embeddings[i])

    if class_nodes:
        graph.query(
            """UNWIND $nodes AS n
               MERGE (c:Class {fqn: n.fqn})
               SET c.name = n.name,
                   c.file_path = n.file_path,
                   c.start_line = n.start_line,
                   c.end_line = n.end_line,
                   c.docstring = n.docstring,
                   c.summary = n.summary,
                   c.embedding = n.embedding""",
            {"nodes": class_nodes},
        )

    # Step 3: Upsert Function nodes
    func_nodes = []
    func_emb_texts = []
    for item in items_to_summarize:
        if item[0] != "function": continue
        func, file_path, mod_name, code = item[1], item[2], item[3], item[4]
        summary = summaries.get(id(func), "")
        emb_text = build_embedding_text(func.name, func.docstring, file_path)
        func_emb_texts.append(emb_text)
        fqn = f"{mod_name}.{func.class_name}.{func.name}" if func.is_method else f"{mod_name}.{func.name}"
        func_nodes.append({
            "fqn": fqn, "name": func.name, "file_path": file_path,
            "start_line": func.start_line, "end_line": func.end_line,
            "docstring": func.docstring or "", "is_method": func.is_method,
            "class_name": func.class_name or "", "module_name": mod_name,
            "summary": summary,
        })
        
    if func_emb_texts:
        func_embeddings = embed_texts(func_emb_texts)
        for i, node in enumerate(func_nodes):
            node["embedding"] = json.dumps(func_embeddings[i])
            
    if func_nodes:
        graph.query(
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
                   f.embedding = n.embedding""",
            {"nodes": func_nodes},
        )

    # Step 4: Create DEFINED_IN and INHERITS_FROM edges
    func_to_class = []
    func_to_mod = []
    class_to_mod = []
    inherits_edges = []
    
    for pf in parsed_files:
        mod_name = pf.file_path.replace("/", ".").replace("\\", ".")
        if mod_name.endswith(".py"):
            mod_name = mod_name[:-3]

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
    import_edges = []
    for pf in parsed_files:
        src_mod = pf.file_path.replace("/", ".").replace("\\", ".")
        if src_mod.endswith(".py"):
            src_mod = src_mod[:-3]

        for imp in pf.imports:
            import_edges.append({
                "src_name": src_mod,
                "tgt_name": imp.module,
                "alias": imp.alias or "",
            })
            
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
    call_edges = []
    for pf in parsed_files:
        mod_name = pf.file_path.replace("/", ".").replace("\\", ".")
        if mod_name.endswith(".py"):
            mod_name = mod_name[:-3]
            
        for call in pf.calls:
            if call.caller_name == "__classbody__" or call.caller_name == "<module>":
                continue  

            caller_fqn = f"{mod_name}.{call.caller_name}"
            callee_simple = call.callee_name.split(".")[-1] if "." in call.callee_name else call.callee_name
            
            call_edges.append({
                "caller_fqn": caller_fqn,
                "callee_name": "." + callee_simple,
                "callee_exact": callee_simple,
                "line": call.line,
                "file_path": call.file_path,
            })
            
    if call_edges:
        graph.query(
            """UNWIND $edges AS e
               MATCH (caller:Function {fqn: e.caller_fqn})
               MATCH (callee:Function) WHERE callee.fqn ENDS WITH e.callee_name OR callee.fqn = e.callee_exact
               MERGE (caller)-[c:CALLS]->(callee)
               SET c.line = e.line, c.file_path = e.file_path""",
            {"edges": call_edges},
        )

def run_ingestion(directory_path: str) -> dict:
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

    files_to_parse = []
    current_files = set()
    for py_file in sorted(dir_path.rglob("*.py")):
        parts = py_file.relative_to(dir_path).parts
        if any(part in SKIP_DIRS for part in parts):
            continue
            
        rel_path = str(py_file.relative_to(dir_path))
        current_files.add(rel_path)
        mtime = py_file.stat().st_mtime
        
        # Only parse if file changed or if FLUSH_GRAPH_ON_INGEST was True (which wipes existing_states)
        if rel_path not in existing_states or existing_states[rel_path] != mtime:
            files_to_parse.append((py_file, rel_path, mtime))

    deleted_files = set(existing_states.keys()) - current_files
    changed_files = [f[1] for f in files_to_parse]
    files_to_delete = changed_files + list(deleted_files)

    if files_to_delete:
        for f in files_to_delete:
            graph.query("MATCH (n {file_path: $fp}) DETACH DELETE n", {"fp": f})
            graph.query("MATCH (s:FileState {file_path: $fp}) DELETE s", {"fp": f})

    parsed_files = []
    for py_file, rel_path, mtime in files_to_parse:
        try:
            parsed = parse_file(py_file, dir_path)
            parsed_files.append(parsed)
            graph.query("MERGE (s:FileState {file_path: $fp}) SET s.mtime = $mtime", {"fp": rel_path, "mtime": mtime})
        except Exception:
            continue

    if parsed_files:
        ingest_parsed_files(parsed_files, graph, dir_path)

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
        "files_parsed": len(parsed_files),
    }
