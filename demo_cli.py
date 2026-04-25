# demo_cli.py
"""
Rich terminal demo. Two modes:
  Mode A — Baseline: LLM answers with no tools (shows what AI coding agents do today)
  Mode B — Graph-Grounded: LLM uses graph tools (shows what Repo-Insight enables)
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
from config import (SGLANG_BASE_URL, SGLANG_API_KEY, LLM_MODEL,
                    FALKORDB_HOST, FALKORDB_PORT, GRAPH_NAME)
from agent import run_repo_agent
from ingest import run_ingestion, get_connection


# Demo prompts designed to expose the A/B gap — these are CODE CHANGE requests,
# not analysis questions. Mode A will miss dependent files. Mode B won't.
DEMO_PROMPTS = [
    "Add a `decorators` field to the FunctionDef dataclass and update all code that creates or reads FunctionDef instances.",
    "Rename `get_connection` to `connect_to_graph` everywhere in the codebase. List every file that needs changing.",
    "Add error handling to `parse_file` so it returns a partial result on syntax errors. What other functions need to handle this new behavior?",
    "Add a `timeout` parameter to `run_ingestion` and propagate it to all database calls it makes.",
]


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


def run_mode_a(query: str, console: Console) -> dict:
    """
    Fire query directly to SGLang with NO tools. Display raw response in a red-bordered panel.
    Returns timing and metadata for comparison.
    """
    console.print()
    console.print("[bold red]━━━ MODE A: Baseline (No Graph Context) ━━━[/bold red]")
    console.print()

    client = openai.OpenAI(
        base_url=SGLANG_BASE_URL,
        api_key=SGLANG_API_KEY,
    )

    mode_a_data = {"time": 0, "answer": "", "error": None}

    try:
        start = time.time()
        with console.status("[bold yellow]Querying LLM without tools...[/bold yellow]"):
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": (
                        "You are a helpful coding assistant. When asked to make code changes, "
                        "list every file that needs modification and describe what changes to make. "
                        "Be thorough — missing a file means the change will break things."
                    )},
                    {"role": "user", "content": query},
                ],
            )
        elapsed = time.time() - start
        answer = response.choices[0].message.content or "(empty response)"
        mode_a_data["time"] = round(elapsed, 1)
        mode_a_data["answer"] = answer

        console.print(Panel(
            answer,
            title="[bold red]MODE A: Baseline (No Graph Context)[/bold red]",
            subtitle=f"[dim]{elapsed:.1f}s[/dim]",
            border_style="red",
            padding=(1, 2),
        ))

    except Exception as e:
        mode_a_data["error"] = str(e)
        console.print(Panel(
            f"[red]Error in Mode A: {e}[/red]",
            title="[bold red]MODE A: Error[/bold red]",
            border_style="red",
        ))

    return mode_a_data


def run_mode_b(query: str, graph: falkordb.Graph, console: Console) -> dict:
    """
    Fire query through run_repo_agent.
    Print tool calls as a Rich Table, then final answer in a green panel.
    Returns timing and metadata for comparison.
    """
    console.print()
    console.print("[bold green]━━━ MODE B: Graph-Grounded (Repo-Insight) ━━━[/bold green]")
    console.print()

    mode_b_data = {"time": 0, "answer": "", "tool_count": 0, "files_traced": 0,
                   "functions_found": 0, "error": None}

    try:
        start = time.time()
        with console.status("[bold cyan]Running ReAct agent loop...[/bold cyan]"):
            result = run_repo_agent(query, graph)
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
                                if item.get("name"):
                                    funcs_seen.add(item["name"])
        mode_b_data["files_traced"] = len(files_seen)
        mode_b_data["functions_found"] = len(funcs_seen)

        # Display tool call log
        if result["tool_calls_log"]:
            tool_table = Table(
                title="[bold cyan]Graph Queries Executed[/bold cyan]",
                show_header=True,
                header_style="bold magenta",
            )
            tool_table.add_column("Iter", style="dim", width=4)
            tool_table.add_column("Tool", style="cyan", width=22)
            tool_table.add_column("Args", style="yellow", width=28)
            tool_table.add_column("Result", style="green", max_width=45)

            for entry in result["tool_calls_log"]:
                args_str = json.dumps(entry["args"], indent=None)
                summary = _summarize_tool_result(entry["result"])

                tool_table.add_row(
                    str(entry["iteration"]),
                    entry["tool"],
                    args_str,
                    summary,
                )

            console.print(tool_table)
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


def _show_comparison(mode_a_data: dict, mode_b_data: dict, console: Console) -> None:
    """Show a quantitative comparison table between Mode A and Mode B."""
    console.print()

    comp_table = Table(
        title="[bold bright_blue]A/B Comparison[/bold bright_blue]",
        show_header=True,
        header_style="bold",
    )
    comp_table.add_column("Metric", style="white", width=25)
    comp_table.add_column("Mode A (Blind)", style="red", justify="center", width=20)
    comp_table.add_column("Mode B (Graph)", style="green", justify="center", width=20)

    comp_table.add_row(
        "Response Time",
        f"{mode_a_data['time']}s",
        f"{mode_b_data['time']}s",
    )
    comp_table.add_row(
        "Graph Queries",
        "0",
        str(mode_b_data["tool_count"]),
    )
    comp_table.add_row(
        "Files Traced",
        "0",
        str(mode_b_data["files_traced"]),
    )
    comp_table.add_row(
        "Functions Found",
        "0",
        str(mode_b_data["functions_found"]),
    )
    comp_table.add_row(
        "Grounded in Code Graph",
        "[red]✗ No[/red]",
        "[green]✓ Yes[/green]",
    )

    console.print(comp_table)

    console.print()
    console.print(Panel(
        "[bold]Key Insight:[/bold]\n\n"
        "• [red]Mode A[/red] proposes changes from the LLM's training data — "
        "it will miss files, invent function names, and ignore dependency chains.\n\n"
        "• [green]Mode B[/green] queries the actual codebase graph before proposing changes — "
        "every file and function reference is verified against the real code structure.\n\n"
        "[dim]The graph doesn't just improve accuracy. It makes the AI's changes complete.[/dim]",
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
        --mode {a,b,ab} : Which mode to run. Default: ab (runs both).
    """
    parser = argparse.ArgumentParser(
        description="Repo-Insight: Graph-RAG for Codebases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python demo_cli.py --ingest --path ./ --mode ab\n"
            "  python demo_cli.py --prompt 'Rename get_connection everywhere' --mode b\n"
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
        choices=["a", "b", "ab"],
        default="ab",
        help="Which mode to run: a (baseline), b (graph), ab (both). Default: ab",
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

    # Step 3: Run the selected mode(s)
    graph = None
    if "b" in args.mode:
        try:
            graph = get_connection()
        except ConnectionError as e:
            console.print(f"[red]Cannot connect to FalkorDB: {e}[/red]")
            if args.mode == "b":
                sys.exit(1)
            else:
                console.print("[yellow]Skipping Mode B due to connection error.[/yellow]")

    mode_a_data = {}
    mode_b_data = {}

    if "a" in args.mode:
        mode_a_data = run_mode_a(selected_prompt, console)

    if "b" in args.mode and graph is not None:
        mode_b_data = run_mode_b(selected_prompt, graph, console)

    # Step 4: Comparison summary if running both modes
    if args.mode == "ab" and mode_a_data and mode_b_data:
        _show_comparison(mode_a_data, mode_b_data, console)

    console.print()


if __name__ == "__main__":
    main()
