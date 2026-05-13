# demo_cli.py
"""
Rich terminal demo. Four modes:
  Mode A — Baseline: LLM answers with no tools (shows what AI coding agents do today)
  Mode B — Graph-Grounded: LLM uses graph tools (shows what Repo-Insight enables)
  Mode C — Graph-Driven: 6-phase pipeline with validation gate + auto-apply
  Fair  — Head-to-head: Graph-Driven (smaller model) vs Blind (bigger model)
"""

import sys
import time
import argparse
import json
import openai
import falkordb
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich import print as rprint
from contextlib import contextmanager
from config import (SGLANG_BASE_URL, SGLANG_API_KEY, LLM_MODEL,
                    FALKORDB_HOST, FALKORDB_PORT, GRAPH_NAME)
from agent import run_repo_agent
from ingest import run_ingestion, get_connection
from pathlib import Path


# Demo prompts designed to expose the A/B gap — these are CODE CHANGE requests,
# not analysis questions. Mode A will miss dependent files. Mode B won't.
DEMO_PROMPTS = [
    "Add a `decorators` field to the FunctionDef dataclass and update all code that creates or reads FunctionDef instances.",
    "Rename `get_connection` to `connect_to_graph` everywhere in the codebase. List every file that needs changing.",
    "Add error handling to `parse_file` so it returns a partial result on syntax errors. What other functions need to handle this new behavior?",
    "Add a `timeout` parameter to `run_ingestion` and propagate it to all database calls it makes.",
]


@contextmanager
def _noop_context():
    """No-op context manager used when diff output is short enough to not need paging."""
    yield


def _summarize_tool_result(res: dict) -> str:
    """Create a concise summary string for a tool result dict."""
    if not isinstance(res, dict):
        return str(res)[:50]

    if "error" in res:
        return f"ERROR: {res['error'][:40]}"
    if "found" in res:
        summary = f"found={res['found']}"
        if res.get("found"):
            summary += f", file={res.get('file_path', '?')}"
            if "source" in res:
                line_count = res["source"].count("\n") + 1
                summary += f", {line_count} lines"
        return summary
    if "caller_count" in res:
        return f"{res['caller_count']} callers"
    if "callee_count" in res:
        return f"{res['callee_count']} callees"
    if "impacted_count" in res:
        summary = f"{res['impacted_count']} impacted ({res.get('direction', '?')})"
        if res.get("warning"):
            summary += " ⚠️"
        return summary
    if "affected_count" in res:
        summary = f"{res['affected_count']} affected ({res.get('direction', '?')})"
        if res.get("warning"):
            summary += " ⚠️"
        return summary
    if "results" in res:
        return f"{len(res['results'])} results"
    return json.dumps(res)[:50]





def run_mode_b(query: str, graph: falkordb.Graph, console: Console) -> dict:
    """
    Fire query through run_repo_agent with live streaming of tool calls.
    Print tool calls as they happen via Rich Live, then final answer in a green panel.
    Returns timing and metadata for comparison.
    """
    from rich.live import Live

    console.print()
    console.print("[bold green]━━━ MODE B: Graph-Grounded (Repo-Insight) ━━━[/bold green]")
    console.print()

    mode_b_data = {"time": 0, "answer": "", "tool_count": 0, "files_traced": 0,
                   "functions_found": 0, "error": None}

    # Build streaming table for live display
    tool_table = Table(
        title="[bold cyan]Graph Queries (Live)[/bold cyan]",
        show_header=True,
        header_style="bold magenta",
    )
    tool_table.add_column("Iter", style="dim", width=4)
    tool_table.add_column("Tool", style="cyan", width=22)
    tool_table.add_column("Args", style="yellow", width=28)
    tool_table.add_column("Result", style="green", max_width=45)

    try:
        start = time.time()

        def _on_tool_call(entry: dict) -> None:
            """Callback invoked after each tool call — updates the live table."""
            args_str = json.dumps(entry["args"], indent=None)
            summary = _summarize_tool_result(entry["result"])
            tool_table.add_row(
                str(entry["iteration"]),
                entry["tool"],
                args_str,
                summary,
            )

        with Live(tool_table, console=console, refresh_per_second=4) as live:
            result = run_repo_agent(query, graph, on_tool_call=_on_tool_call)

        elapsed = time.time() - start

        mode_b_data["time"] = round(elapsed, 1)
        mode_b_data["answer"] = result["answer"]
        mode_b_data["tool_count"] = len(result["tool_calls_log"])

        # Count unique files and functions traced across all tool results
        files_seen = set()
        funcs_seen = set()
        for entry in result["tool_calls_log"]:
            res = entry["result"]
            if isinstance(res, dict):
                if res.get("file_path"):
                    files_seen.add(res["file_path"])
                for key in ("callers", "callees", "impacted", "affected", "results"):
                    if key in res:
                        for item in res[key]:
                            if isinstance(item, dict):
                                if item.get("file_path"):
                                    files_seen.add(item["file_path"])
                                if item.get("fqn"):
                                    funcs_seen.add(item["fqn"])
        mode_b_data["files_traced"] = len(files_seen)
        mode_b_data["functions_found"] = len(funcs_seen)

        console.print()

        # Display status info
        status_parts = [
            f"Iterations: {result['iterations']}",
            f"Tools called: {len(result['tool_calls_log'])}",
            f"Files traced: {mode_b_data['files_traced']}",
            f"Functions found: {mode_b_data['functions_found']}",
        ]
        if result["hit_max_iterations"]:
            status_parts.append("[red]⚠ Hit max iterations![/red]")
        console.print(" | ".join(status_parts))
        console.print()

        # Display final answer
        console.print(Panel(
            result["answer"],
            title="[bold green]MODE B: Graph-Grounded (Repo-Insight)[/bold green]",
            subtitle=f"[dim]{elapsed:.1f}s | {len(result['tool_calls_log'])} graph queries[/dim]",
            border_style="green",
            padding=(1, 2),
        ))

        # Display diff output if available (4.7)
        if result.get("diff"):
            with console.pager() if len(result["diff"]) > 2000 else _noop_context():
                console.print(Panel(
                    Syntax(result["diff"], "diff", theme="monokai", line_numbers=False),
                    title="[bold yellow]Change Set (Diff Format)[/bold yellow]",
                    border_style="yellow",
                    padding=(1, 2),
                ))

    except ConnectionError as e:
        mode_b_data["error"] = str(e)
        console.print(Panel(
            f"[red]FalkorDB connection error: {e}[/red]",
            title="[bold red]MODE B: Error[/bold red]",
            border_style="red",
        ))
    except Exception as e:
        mode_b_data["error"] = str(e)
        console.print(Panel(
            f"[red]Error in Mode B: {e}[/red]",
            title="[bold red]MODE B: Error[/bold red]",
            border_style="red",
        ))

    return mode_b_data


def run_mode_c(query: str, repo_path: str, graph: falkordb.Graph, console: Console) -> dict:
    """Run Mode C: 6-phase graph-driven pipeline with live phase display."""
    from change_engine import GraphDrivenEngine

    console.print()
    console.print("[bold bright_magenta]━━━ MODE C: Graph-Driven Engine (6-Phase Pipeline) ━━━[/bold bright_magenta]")
    console.print()

    mode_c_data = {"time": 0, "answer": "", "error": None, "phases": [],
                   "edits": 0, "tests_passed": False, "validated": False}

    # Phase status display
    phase_table = Table(
        title="[bold cyan]Pipeline Phases (Live)[/bold cyan]",
        show_header=True, header_style="bold magenta",
    )
    phase_table.add_column("Phase", style="cyan", width=40)
    phase_table.add_column("Status", style="green", width=15)
    phase_table.add_column("Details", style="yellow", max_width=40)

    def _on_phase(phase: str, data: dict) -> None:
        labels = {
            "phase_0": "Phase 0: Graph Construction",
            "phase_1": "Phase 1: Seed Localization",
            "phase_2": "Phase 2: Structural Expansion",
            "phase_3": "Phase 3: Graph-Constrained Planning",
            "phase_4": "Phase 4: Surgical Editing",
            "phase_5": "Phase 5: Verified Apply",
            "phase_5_test": "Phase 5: Test Run",
        }
        label = labels.get(phase, phase)
        detail = ""
        if phase == "phase_0":
            detail = f"{data.get('functions', '?')} functions, {data.get('call_edges', '?')} edges"
        elif phase == "phase_1":
            seeds = data.get('seeds', [])
            detail = f"{len(seeds)} seeds: {', '.join(seeds[:3])}"
        elif phase == "phase_2":
            detail = f"{data.get('blast_radius_count', '?')} blast nodes, {data.get('files_affected', '?')} files"
        elif phase == "phase_3":
            detail = f"validated={data.get('is_validated')}, missing={data.get('missing_files', 0)}"
            mode_c_data["validated"] = data.get("is_validated", False)
        elif phase == "phase_4":
            detail = f"{data.get('edit_blocks', 0)} SEARCH/REPLACE blocks"
            mode_c_data["edits"] = data.get("edit_blocks", 0)
        elif phase == "phase_5":
            detail = f"apply={data.get('apply_success')}, tests={data.get('tests_passed')}"
            mode_c_data["tests_passed"] = data.get("tests_passed", False)
        elif phase == "phase_5_test":
            detail = f"attempt {data.get('attempt')}: {data.get('passed', 0)} passed, {data.get('failed', 0)} failed"

        phase_table.add_row(label, "✓ Done", detail)

    try:
        from rich.live import Live
        engine = GraphDrivenEngine(Path(repo_path).resolve(), graph)
        start = time.time()

        with Live(phase_table, console=console, refresh_per_second=4):
            result = engine.run(query, on_phase=_on_phase)

        elapsed = time.time() - start
        mode_c_data["time"] = round(elapsed, 1)
        mode_c_data["answer"] = result.answer
        mode_c_data["phases"] = result.phases_completed

        console.print()

        # Show timing breakdown
        if result.timings:
            timing_parts = [f"{k.replace('phase_', 'P')}: {v:.1f}s" for k, v in result.timings.items()]
            console.print(f"[dim]Timing: {' | '.join(timing_parts)}[/dim]")

        # Show answer
        console.print(Panel(
            result.answer if result.answer else "(no answer generated)",
            title="[bold bright_magenta]MODE C: Graph-Driven Engine[/bold bright_magenta]",
            subtitle=f"[dim]{elapsed:.1f}s | {len(result.edits)} edits | validated={mode_c_data['validated']}[/dim]",
            border_style="bright_magenta",
            padding=(1, 2),
        ))

        # Show edit blocks if generated
        if result.edits:
            edit_text = "\n\n".join(
                f"FILE: {eb.file_path}\n<<<<<<< SEARCH\n{eb.search_text[:200]}...\n=======\n{eb.replace_text[:200]}...\n>>>>>>> REPLACE"
                for eb in result.edits[:5]
            )
            console.print(Panel(
                Syntax(edit_text, "diff", theme="monokai", line_numbers=False),
                title="[bold yellow]Generated Edit Blocks[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            ))

        # Show test results if available
        if result.test_result:
            status = "[green]✓ ALL PASSED[/green]" if result.test_result.all_passed else "[red]✗ FAILURES[/red]"
            console.print(Panel(
                f"Tests: {result.test_result.passed} passed, {result.test_result.failed} failed, "
                f"{result.test_result.errors} errors\nStatus: {status}",
                title="[bold]Test Results[/bold]",
                border_style="green" if result.test_result.all_passed else "red",
            ))

        if result.error:
            console.print(f"[red]Pipeline error: {result.error}[/red]")

    except Exception as e:
        mode_c_data["error"] = str(e)
        console.print(Panel(f"[red]Error in Mode C: {e}[/red]", title="MODE C: Error", border_style="red"))

    return mode_c_data


def run_mode_d(ref: str, repo_path: str, graph: falkordb.Graph, console: Console) -> dict:
    """Mode D: git-diff entry — automatically fix what broke since a git ref."""
    from change_engine import GraphDrivenEngine

    console.print()
    console.print(f"[bold bright_cyan]━━━ MODE D: Git-Diff Auto-Fix (ref: {ref}) ━━━[/bold bright_cyan]")
    console.print()

    mode_d_data = {"time": 0, "answer": "", "error": None, "phases": [], "edits": 0}

    phase_table = Table(
        title="[bold cyan]Git-Diff Pipeline (Live)[/bold cyan]",
        show_header=True, header_style="bold magenta",
    )
    phase_table.add_column("Phase", style="cyan", width=40)
    phase_table.add_column("Status", style="green", width=15)
    phase_table.add_column("Details", style="yellow", max_width=40)

    def _on_phase(phase: str, data: dict) -> None:
        labels = {
            "phase_0": "Phase 0: Git Diff Impact",
            "phase_1": "Phase 1: Seeds from Git Diff",
            "phase_2": "Phase 2: Structural Expansion",
            "phase_3": "Phase 3: Graph-Constrained Planning",
            "phase_4": "Phase 4: Surgical Editing",
            "phase_5": "Phase 5: Verified Apply",
        }
        label = labels.get(phase, phase)
        detail = ""
        if phase == "phase_0":
            detail = f"{data.get('total_at_risk', '?')} at-risk callers"
        elif phase == "phase_1":
            seeds = data.get("seeds", [])
            detail = f"{len(seeds)} seeds from diff"
        elif phase == "phase_2":
            detail = f"{data.get('blast_radius_count', '?')} blast nodes"
        elif phase == "phase_3":
            detail = f"validated={data.get('is_validated')}"
            mode_d_data["edits"] = data.get("edit_blocks", 0)
        elif phase == "phase_4":
            detail = f"{data.get('edit_blocks', 0)} SEARCH/REPLACE blocks"
            mode_d_data["edits"] = data.get("edit_blocks", 0)
        phase_table.add_row(label, "✓ Done", detail)

    try:
        from rich.live import Live
        engine = GraphDrivenEngine(Path(repo_path).resolve(), graph)
        start = time.time()

        with Live(phase_table, console=console, refresh_per_second=4):
            result = engine.run_from_diff(ref=ref, on_phase=_on_phase)

        elapsed = time.time() - start
        mode_d_data["time"] = round(elapsed, 1)
        mode_d_data["answer"] = result.answer
        mode_d_data["phases"] = result.phases_completed

        console.print()
        console.print(Panel(
            result.answer if result.answer else "(no answer generated)",
            title=f"[bold bright_cyan]MODE D: Git-Diff Auto-Fix ({ref})[/bold bright_cyan]",
            subtitle=f"[dim]{elapsed:.1f}s | {mode_d_data['edits']} edits[/dim]",
            border_style="bright_cyan",
            padding=(1, 2),
        ))

        if result.error:
            console.print(f"[red]Error: {result.error}[/red]")

    except Exception as e:
        mode_d_data["error"] = str(e)
        console.print(Panel(f"[red]Error in Mode D: {e}[/red]", title="MODE D: Error", border_style="red"))

    return mode_d_data


def _show_comparison(mode_b_data: dict, console: Console,
                     mode_c_data: dict = None) -> None:
    """Show a quantitative comparison table between modes."""
    console.print()

    has_c = mode_c_data is not None and mode_c_data

    comp_table = Table(
        title="[bold bright_blue]Mode Comparison[/bold bright_blue]",
        show_header=True, header_style="bold",
    )
    comp_table.add_column("Metric", style="white", width=25)
    comp_table.add_column("Mode B (Tool-Call)", style="green", justify="center", width=18)
    if has_c:
        comp_table.add_column("Mode C (Graph-Driven)", style="bright_magenta", justify="center", width=20)

    row = lambda m, a, b, c=None: comp_table.add_row(m, a, b, *([c] if has_c and c else []))

    row("Response Time",  f"{mode_b_data.get('time', '?')}s",
        f"{mode_c_data.get('time', '?')}s" if has_c else None)
    row("Graph Queries", str(mode_b_data.get("tool_count", 0)),
        "exhaustive" if has_c else None)
    row("Files Traced", str(mode_b_data.get("files_traced", 0)),
        "all in blast radius" if has_c else None)
    row("Validation Gate", "[red]✗ No[/red]", "[red]✗ No[/red]",
        "[green]✓ Yes[/green]" if has_c and mode_c_data.get("validated") else "[red]✗ No[/red]" if has_c else None)
    row("Auto-Apply + Test", "[red]✗ No[/red]", "[red]✗ No[/red]",
        "[green]✓ Yes[/green]" if has_c and mode_c_data.get("tests_passed") else "[yellow]attempted[/yellow]" if has_c else None)
    row("Structural Coverage", "[red]✗ None[/red]", "[yellow]~ Partial[/yellow]",
        "[green]✓ Guaranteed[/green]" if has_c else None)

    console.print(comp_table)
    console.print()
    console.print(Panel(
        "[bold]Key Insight:[/bold]\n\n"
        "• [red]Mode A[/red] guesses from training data — misses files, invents names.\n"
        "• [green]Mode B[/green] queries the graph — finds more files, but coverage is non-deterministic.\n"
        + ("• [bright_magenta]Mode C[/bright_magenta] uses the graph as the control plane — "
           "exhaustive traversal, validated plan, auto-applied and tested.\n" if has_c else "") +
        "\n[dim]The graph doesn't just improve accuracy. It guarantees structural completeness.[/dim]",
        title="[bold bright_blue]Why This Matters[/bold bright_blue]",
        border_style="bright_blue",
        padding=(1, 2),
    ))






def main() -> None:
    """
    CLI entry point.

    Arguments:
        --path PATH     : Path to repo to ingest. Default: ./target_repo
        --ingest        : Flag. If set, run ingestion before demo.
        --prompt TEXT   : Run a specific prompt directly (skip interactive selection).
        --mode {b,c,bc} : Which mode to run. Default: b.
        --score         : Run quantitative scoring suite.
    """
    parser = argparse.ArgumentParser(
        description="Repo-Insight: Graph-Driven Coding Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python demo_cli.py --ingest --path ./ --mode abc\n"
            "  python demo_cli.py --prompt 'Rename get_connection everywhere' --mode c\n"
            "  python demo_cli.py --mode fair --path ./   # Head-to-head: graph vs blind\n"
            "  python demo_cli.py --score --path ./\n"
        ),
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./target_repo",
        help="Path to the repository to ingest (default: ./target_repo)",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run ingestion before the demo",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Run a specific prompt directly (skip interactive selection)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="b",
        help="Which mode(s) to run: b, c, bc, d. Default: b. Mode d = git-diff auto-fix.",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Run quantitative scoring suite (precision/recall/F1)",
    )
    parser.add_argument(
        "--ref",
        type=str,
        default="HEAD",
        help="Git ref for Mode D (git-diff entry mode). Default: HEAD",
    )

    args = parser.parse_args()
    console = Console()

    # Header
    console.print()
    console.print(Panel(
        "[bold white]🔍 Repo-Insight[/bold white]\n"
        "[dim]Graph-RAG for Codebases — Complete Changes, Not Broken Code[/dim]",
        border_style="bright_blue",
        padding=(1, 2),
    ))

    # Step 1: Optional ingestion
    if args.ingest:
        console.print(f"\n[bold cyan]📦 Ingesting repository:[/bold cyan] {args.path}")
        try:
            start = time.time()
            console.print("[bold yellow]Parsing AST, generating embeddings, and building graph...[/bold yellow]")
            summary = run_ingestion(args.path)
            elapsed = time.time() - start

            # Display ingestion summary
            ingest_table = Table(title=f"[bold]Ingestion Complete[/bold] [dim]({elapsed:.1f}s)[/dim]")
            ingest_table.add_column("Metric", style="cyan")
            ingest_table.add_column("Count", style="green", justify="right")
            for key, value in summary.items():
                ingest_table.add_row(key.replace("_", " ").title(), str(value))
            console.print(ingest_table)
        except FileNotFoundError as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        except ConnectionError as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

    # Step 2: Select a prompt
    if args.prompt:
        selected_prompt = args.prompt
    else:
        console.print("\n[bold]Select a demo prompt (code change requests):[/bold]")
        for i, prompt in enumerate(DEMO_PROMPTS, 1):
            console.print(f"  [cyan]{i}.[/cyan] {prompt}")
        console.print(f"  [cyan]{len(DEMO_PROMPTS) + 1}.[/cyan] [dim]Enter a custom prompt[/dim]")

        choice = Prompt.ask(
            "\n[bold]Enter your choice[/bold]",
            default="1",
        )

        try:
            idx = int(choice)
            if 1 <= idx <= len(DEMO_PROMPTS):
                selected_prompt = DEMO_PROMPTS[idx - 1]
            else:
                selected_prompt = Prompt.ask("[bold]Enter your custom prompt[/bold]")
        except ValueError:
            selected_prompt = choice  # Treat raw text as prompt

    console.print(f"\n[bold]Prompt:[/bold] {selected_prompt}")
    console.print("─" * 60)

    # Scoring mode
    if args.score:
        from scoring import run_scoring_suite, print_scoring_report
        console.print("\n[bold cyan]📊 Running Scoring Suite...[/bold cyan]")
        try:
            graph = get_connection()
            score_modes = ["b", "c"]
            report = run_scoring_suite(graph, Path(args.path).resolve(), modes=score_modes)
            print_scoring_report(report)
        except Exception as e:
            console.print(f"[red]Scoring failed: {e}[/red]")
        return

    # Step 3: Run the selected mode(s)
    graph = None
    if "b" in args.mode or "c" in args.mode:
        try:
            graph = get_connection()
        except ConnectionError as e:
            console.print(f"[red]Cannot connect to FalkorDB: {e}[/red]")
            if args.mode in ("b", "c"):
                sys.exit(1)
            else:
                console.print("[yellow]Skipping graph modes due to connection error.[/yellow]")


    mode_b_data = {}
    mode_c_data = {}

    if "b" in args.mode and graph is not None:
        mode_b_data = run_mode_b(selected_prompt, graph, console)

    if "c" in args.mode and graph is not None:
        mode_c_data = run_mode_c(selected_prompt, args.path, graph, console)

    if "d" in args.mode and graph is not None:
        run_mode_d(args.ref, args.path, graph, console)
        return  # Mode D is standalone — no comparison table

    # Step 4: Comparison summary
    if len(args.mode) > 1 and mode_b_data:
        _show_comparison(
            mode_b_data,
            console,
            mode_c_data=mode_c_data if mode_c_data else None,
        )

    console.print()


if __name__ == "__main__":
    main()
