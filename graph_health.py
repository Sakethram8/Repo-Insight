# graph_health.py
"""
Graph health and statistics module.
Provides diagnostics about the ingested codebase graph: node/edge counts,
orphan detection, hub analysis, and staleness metrics.
"""

import logging
from typing import Any
import falkordb
from ingest import get_connection

logger = logging.getLogger(__name__)


def get_graph_health(graph: falkordb.Graph | None = None) -> dict[str, Any]:
    """Return a comprehensive health report for the codebase graph.

    Args:
        graph: Optional FalkorDB graph connection. If None, creates one.

    Returns:
        Dict with keys: node_counts, edge_counts, orphans, hubs, staleness.
    """
    if graph is None:
        graph = get_connection()

    report: dict[str, Any] = {}

    # --- Node counts by label ---
    try:
        func_count = graph.query("MATCH (f:Function) RETURN count(f)").result_set[0][0]
        class_count = graph.query("MATCH (c:Class) RETURN count(c)").result_set[0][0]
        module_count = graph.query("MATCH (m:Module) RETURN count(m)").result_set[0][0]
        meta_count = graph.query("MATCH (m:Meta) RETURN count(m)").result_set[0][0]
        filestate_count = graph.query("MATCH (s:FileState) RETURN count(s)").result_set[0][0]
    except Exception as e:
        logger.error("Failed to count nodes: %s", e)
        func_count = class_count = module_count = meta_count = filestate_count = 0

    report["node_counts"] = {
        "Function": func_count,
        "Class": class_count,
        "Module": module_count,
        "Meta": meta_count,
        "FileState": filestate_count,
        "total": func_count + class_count + module_count + meta_count + filestate_count,
    }

    # --- Edge counts by type ---
    try:
        calls_count = graph.query("MATCH ()-[c:CALLS]->() RETURN count(c)").result_set[0][0]
        defined_count = graph.query("MATCH ()-[d:DEFINED_IN]->() RETURN count(d)").result_set[0][0]
        imports_count = graph.query("MATCH ()-[i:IMPORTS]->() RETURN count(i)").result_set[0][0]
        inherits_count = graph.query("MATCH ()-[i:INHERITS_FROM]->() RETURN count(i)").result_set[0][0]
        reads_count = graph.query("MATCH ()-[r:READS]->() RETURN count(r)").result_set[0][0]
    except Exception as e:
        logger.error("Failed to count edges: %s", e)
        calls_count = defined_count = imports_count = inherits_count = reads_count = 0

    report["edge_counts"] = {
        "CALLS": calls_count,
        "DEFINED_IN": defined_count,
        "IMPORTS": imports_count,
        "INHERITS_FROM": inherits_count,
        "READS": reads_count,
        "total": calls_count + defined_count + imports_count + inherits_count + reads_count,
    }

    # --- Orphan nodes (no edges at all) ---
    try:
        orphan_result = graph.query(
            """MATCH (n)
               WHERE NOT (n)--()
                 AND NOT n:Meta AND NOT n:FileState
               RETURN labels(n)[0] AS label, n.fqn AS fqn, n.name AS name
               LIMIT 20"""
        )
        orphans = [
            {"label": r[0], "fqn": r[1], "name": r[2]}
            for r in orphan_result.result_set
        ]
    except Exception as e:
        logger.error("Failed to find orphans: %s", e)
        orphans = []

    report["orphans"] = {
        "count": len(orphans),
        "sample": orphans,
    }

    # --- Hub functions (highest in-degree via CALLS) ---
    try:
        hub_result = graph.query(
            """MATCH (caller:Function)-[:CALLS]->(f:Function)
               RETURN f.fqn, f.file_path, count(caller) AS in_degree
               ORDER BY in_degree DESC
               LIMIT 10"""
        )
        hubs = [
            {"fqn": r[0], "file_path": r[1], "in_degree": r[2]}
            for r in hub_result.result_set
        ]
    except Exception as e:
        logger.error("Failed to find hubs: %s", e)
        hubs = []

    report["hubs"] = hubs

    # --- Staleness metrics ---
    try:
        staleness_result = graph.query(
            """MATCH (s:FileState)
               RETURN min(s.mtime) AS oldest, max(s.mtime) AS newest, count(s) AS tracked_files"""
        )
        row = staleness_result.result_set[0] if staleness_result.result_set else [None, None, 0]
        report["staleness"] = {
            "oldest_mtime": row[0],
            "newest_mtime": row[1],
            "tracked_files": row[2],
        }
    except Exception as e:
        logger.error("Failed to get staleness: %s", e)
        report["staleness"] = {"oldest_mtime": None, "newest_mtime": None, "tracked_files": 0}

    return report


def print_health_report(graph: falkordb.Graph | None = None) -> None:
    """Print a formatted health report to stdout using Rich."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()
    report = get_graph_health(graph)

    # Node counts table
    node_table = Table(title="Node Counts", show_header=True, header_style="bold cyan")
    node_table.add_column("Label", style="white")
    node_table.add_column("Count", style="green", justify="right")
    for label, count in report["node_counts"].items():
        if label == "total":
            node_table.add_row("[bold]Total[/bold]", f"[bold]{count}[/bold]")
        else:
            node_table.add_row(label, str(count))

    # Edge counts table
    edge_table = Table(title="Edge Counts", show_header=True, header_style="bold cyan")
    edge_table.add_column("Type", style="white")
    edge_table.add_column("Count", style="green", justify="right")
    for etype, count in report["edge_counts"].items():
        if etype == "total":
            edge_table.add_row("[bold]Total[/bold]", f"[bold]{count}[/bold]")
        else:
            edge_table.add_row(etype, str(count))

    console.print(node_table)
    console.print(edge_table)

    # Orphans
    orphan_count = report["orphans"]["count"]
    if orphan_count > 0:
        console.print(f"\n[yellow]⚠ {orphan_count} orphan node(s) found (no edges):[/yellow]")
        for o in report["orphans"]["sample"]:
            console.print(f"  [{o['label']}] {o.get('fqn') or o.get('name')}")
    else:
        console.print("\n[green]✓ No orphan nodes[/green]")

    # Hubs
    if report["hubs"]:
        hub_table = Table(title="Top Hub Functions (Highest In-Degree)", show_header=True, header_style="bold magenta")
        hub_table.add_column("FQN", style="cyan")
        hub_table.add_column("File", style="dim")
        hub_table.add_column("In-Degree", style="green", justify="right")
        for h in report["hubs"]:
            hub_table.add_row(h["fqn"], h["file_path"], str(h["in_degree"]))
        console.print(hub_table)

    # Staleness
    staleness = report["staleness"]
    console.print(Panel(
        f"Tracked files: {staleness['tracked_files']}\n"
        f"Oldest mtime:  {staleness['oldest_mtime']}\n"
        f"Newest mtime:  {staleness['newest_mtime']}",
        title="Staleness Metrics",
        border_style="dim",
    ))


if __name__ == "__main__":
    print_health_report()
