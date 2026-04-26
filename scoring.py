# scoring.py
"""
Quantitative scoring harness for Repo-Insight.
Measures file-level precision, recall, and F1 for Mode A (blind LLM),
Mode B (tool-calling agent), and Mode C (graph-driven engine).
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import openai
import falkordb

from config import SGLANG_BASE_URL, SGLANG_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ground truth tasks
# ---------------------------------------------------------------------------

GROUND_TRUTH_TASKS = [
    {
        "prompt": "Add a `decorators` field to the FunctionDef dataclass and update all code that creates or reads FunctionDef instances.",
        "ground_truth_files": {"parser.py", "ingest.py", "tests/test_parser.py"},
    },
    {
        "prompt": "Rename `get_connection` to `connect_to_graph` everywhere in the codebase. List every file that needs changing.",
        "ground_truth_files": {"ingest.py", "mcp_server.py", "graph_health.py", "change_engine.py"},
    },
    {
        "prompt": "Add error handling to `parse_file` so it returns a partial result on syntax errors. What other functions need to handle this new behavior?",
        "ground_truth_files": {"parser.py", "ingest.py", "tests/test_parser.py"},
    },
    {
        "prompt": "Add a `timeout` parameter to `run_ingestion` and propagate it to all database calls it makes.",
        "ground_truth_files": {"ingest.py", "change_engine.py"},
    },
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaskScore:
    """Score for a single task under one mode."""
    task_prompt: str
    mode: str
    mentioned_files: set[str] = field(default_factory=set)
    ground_truth_files: set[str] = field(default_factory=set)
    true_positives: set[str] = field(default_factory=set)
    false_positives: set[str] = field(default_factory=set)
    false_negatives: set[str] = field(default_factory=set)
    hallucinated_files: set[str] = field(default_factory=set)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    time_seconds: float = 0.0


@dataclass
class ScoringReport:
    """Aggregate scoring results across all tasks and modes."""
    task_scores: list[TaskScore] = field(default_factory=list)
    mode_averages: dict[str, dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# File extraction from LLM output
# ---------------------------------------------------------------------------

def extract_mentioned_files(answer: str, known_files: set[str]) -> set[str]:
    """Parse an LLM answer to extract all file paths mentioned.

    Uses both exact matching against known files and regex patterns.
    """
    mentioned = set()

    # Strategy 1: Check if any known file is mentioned in the answer
    for f in known_files:
        # Check for the full path or just the basename
        if f in answer or Path(f).name in answer:
            mentioned.add(f)

    # Strategy 2: Regex for common file path patterns
    # Matches: path/to/file.py, `file.py`, "file.py"
    path_pattern = re.compile(r'[\w/\\]+\.(?:py|js|ts|jsx|tsx)')
    for match in path_pattern.finditer(answer):
        candidate = match.group().strip()
        # Check if it matches a known file
        for f in known_files:
            if candidate == f or candidate == Path(f).name or f.endswith(candidate):
                mentioned.add(f)

    return mentioned


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

def score_single(
    mentioned: set[str],
    ground_truth: set[str],
    all_repo_files: set[str],
) -> tuple[float, float, float, set[str]]:
    """Compute precision, recall, F1, and hallucinated files.

    Returns: (precision, recall, f1, hallucinated_files)
    """
    tp = mentioned & ground_truth
    fp = mentioned - ground_truth
    fn = ground_truth - mentioned
    hallucinated = mentioned - all_repo_files

    precision = len(tp) / len(mentioned) if mentioned else 0.0
    recall = len(tp) / len(ground_truth) if ground_truth else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1, hallucinated


def run_mode_a_for_scoring(prompt: str, client: openai.OpenAI) -> tuple[str, float]:
    """Run Mode A (blind LLM) and return the answer + time."""
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Error: {e}"
    elapsed = time.time() - start
    return answer, elapsed


def run_scoring_suite(
    graph: falkordb.Graph,
    repo_root: Path,
    modes: list[str] = None,
    tasks: list[dict] = None,
) -> ScoringReport:
    """Run all ground truth tasks through specified modes and score them.

    Args:
        graph: FalkorDB graph connection.
        repo_root: Path to the repository.
        modes: List of modes to test. Default: ["a", "b", "c"].
        tasks: Override tasks. Default: GROUND_TRUTH_TASKS.
    """
    if modes is None:
        modes = ["a", "b", "c"]
    if tasks is None:
        tasks = GROUND_TRUTH_TASKS

    # Collect all known files in the repo
    all_repo_files: set[str] = set()
    for f in repo_root.rglob("*.py"):
        try:
            all_repo_files.add(str(f.relative_to(repo_root)))
        except ValueError:
            pass

    client = openai.OpenAI(base_url=SGLANG_BASE_URL, api_key=SGLANG_API_KEY)
    report = ScoringReport()

    for task in tasks:
        prompt = task["prompt"]
        gt_files = task["ground_truth_files"]

        for mode in modes:
            if mode == "a":
                answer, elapsed = run_mode_a_for_scoring(prompt, client)
            elif mode == "b":
                from agent import run_repo_agent
                start = time.time()
                try:
                    result = run_repo_agent(prompt, graph)
                    answer = result["answer"]
                except Exception as e:
                    answer = f"Error: {e}"
                elapsed = time.time() - start
            elif mode == "c":
                from change_engine import GraphDrivenEngine
                engine = GraphDrivenEngine(repo_root, graph)
                start = time.time()
                try:
                    result = engine.run(prompt, skip_apply=True)
                    answer = result.answer
                except Exception as e:
                    answer = f"Error: {e}"
                elapsed = time.time() - start
            else:
                continue

            mentioned = extract_mentioned_files(answer, all_repo_files)
            precision, recall, f1, hallucinated = score_single(
                mentioned, gt_files, all_repo_files,
            )

            score = TaskScore(
                task_prompt=prompt[:80] + "...",
                mode=f"Mode {mode.upper()}",
                mentioned_files=mentioned,
                ground_truth_files=gt_files,
                true_positives=mentioned & gt_files,
                false_positives=mentioned - gt_files,
                false_negatives=gt_files - mentioned,
                hallucinated_files=hallucinated,
                precision=round(precision, 3),
                recall=round(recall, 3),
                f1=round(f1, 3),
                time_seconds=round(elapsed, 1),
            )
            report.task_scores.append(score)

    # Compute per-mode averages
    for mode in modes:
        mode_label = f"Mode {mode.upper()}"
        mode_scores = [s for s in report.task_scores if s.mode == mode_label]
        if mode_scores:
            report.mode_averages[mode_label] = {
                "avg_precision": round(sum(s.precision for s in mode_scores) / len(mode_scores), 3),
                "avg_recall": round(sum(s.recall for s in mode_scores) / len(mode_scores), 3),
                "avg_f1": round(sum(s.f1 for s in mode_scores) / len(mode_scores), 3),
                "avg_time": round(sum(s.time_seconds for s in mode_scores) / len(mode_scores), 1),
            }

    return report


# ---------------------------------------------------------------------------
# Rich output
# ---------------------------------------------------------------------------

def print_scoring_report(report: ScoringReport) -> None:
    """Display scoring results as Rich tables."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Per-task scores
    table = Table(
        title="[bold cyan]Scoring Results: File-Level Precision/Recall/F1[/bold cyan]",
        show_header=True, header_style="bold magenta",
    )
    table.add_column("Task", style="dim", max_width=40)
    table.add_column("Mode", style="cyan", width=10)
    table.add_column("Precision", justify="right", style="green")
    table.add_column("Recall", justify="right", style="yellow")
    table.add_column("F1", justify="right", style="bold white")
    table.add_column("Missed", justify="right", style="red")
    table.add_column("Time", justify="right", style="dim")

    for s in report.task_scores:
        table.add_row(
            s.task_prompt[:40],
            s.mode,
            f"{s.precision:.0%}",
            f"{s.recall:.0%}",
            f"{s.f1:.0%}",
            str(len(s.false_negatives)),
            f"{s.time_seconds:.1f}s",
        )

    console.print(table)
    console.print()

    # Averages
    avg_table = Table(
        title="[bold cyan]Aggregate Averages[/bold cyan]",
        show_header=True, header_style="bold magenta",
    )
    avg_table.add_column("Mode", style="cyan")
    avg_table.add_column("Avg Precision", justify="right", style="green")
    avg_table.add_column("Avg Recall", justify="right", style="yellow")
    avg_table.add_column("Avg F1", justify="right", style="bold white")
    avg_table.add_column("Avg Time", justify="right", style="dim")

    for mode_label, avgs in report.mode_averages.items():
        avg_table.add_row(
            mode_label,
            f"{avgs['avg_precision']:.0%}",
            f"{avgs['avg_recall']:.0%}",
            f"{avgs['avg_f1']:.0%}",
            f"{avgs['avg_time']:.1f}s",
        )

    console.print(avg_table)
