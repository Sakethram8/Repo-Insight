#!/usr/bin/env python3
"""
Compare A/B SWE-bench results and generate the Bob demo shortlist.

Usage:
  python compare_results.py \\
    --baseline ./results/ccli_baseline/results.json \\
    --graph    ./results/ccli_graph/results.json \\
    --output   ./results/comparison.md

Output:
  - Ranked table of all 50 instances showing which condition solved each one
  - Bob demo shortlist: top 5 instances with the biggest graph advantage
  - Token efficiency comparison (headline numbers for the submission)
  - Per-instance detail for each shortlisted instance
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def load(path: str) -> dict[str, dict]:
    data = json.loads(Path(path).read_text())
    return {r["instance_id"]: r for r in data}


def _tok(r: dict) -> int:
    return r.get("total_tokens", 0)


def _resolved(r: dict) -> bool:
    return r.get("status") == "patched"


def compare(baseline: dict, graph: dict, output_path: str) -> None:
    all_ids = sorted(set(baseline) | set(graph))

    # -----------------------------------------------------------------------
    # Per-instance comparison
    # -----------------------------------------------------------------------
    rows = []
    for iid in all_ids:
        b = baseline.get(iid, {})
        g = graph.get(iid, {})

        b_ok  = _resolved(b)
        g_ok  = _resolved(g)
        b_tok = _tok(b)
        g_tok = _tok(g)

        tok_saved  = b_tok - g_tok if b_tok > 0 and g_tok > 0 else None
        tok_pct    = round(100 * tok_saved / b_tok, 1) if tok_saved and b_tok else None

        # Score for Bob demo priority (higher = better demo candidate)
        #   +30  graph solved, baseline did NOT  (clearest win)
        #   +20  graph solved, baseline solved too but graph used fewer tokens
        #   +10  graph used significantly fewer files (precision advantage)
        #    0   both failed or both solved with similar tokens
        score = 0
        reason = []

        if g_ok and not b_ok:
            score += 30
            reason.append("graph solved, baseline failed")
        elif g_ok and b_ok and tok_saved and tok_saved > 0:
            score += 20
            efficiency = tok_saved / b_tok if b_tok else 0
            reason.append(f"{efficiency:.0%} fewer tokens with graph")
        elif not g_ok and not b_ok:
            pass  # both failed — not interesting for demo

        g_files = g.get("files_read_estimate", 0)
        b_files = b.get("files_read_estimate", 0)
        if b_files > 0 and g_files < b_files:
            score += 10
            reason.append(f"read {b_files - g_files} fewer files")

        tools = g.get("tools_called", [])
        if "run_failing_tests_and_localize" in tools:
            score += 5
            reason.append("used stack-trace localization")
        if "get_function_fingerprints" in tools:
            score += 3
            reason.append("used fingerprints")

        rows.append({
            "instance_id": iid,
            "baseline_status": b.get("status", "missing"),
            "graph_status":    g.get("status", "missing"),
            "baseline_tokens": b_tok,
            "graph_tokens":    g_tok,
            "tokens_saved":    tok_saved,
            "tokens_saved_pct": tok_pct,
            "baseline_files":  b_files,
            "graph_files":     g_files,
            "tools_called":    tools,
            "demo_score":      score,
            "demo_reason":     " | ".join(reason) if reason else "no advantage",
        })

    rows.sort(key=lambda x: x["demo_score"], reverse=True)

    # -----------------------------------------------------------------------
    # Headline metrics
    # -----------------------------------------------------------------------
    b_resolved = sum(1 for r in rows if r["baseline_status"] == "patched")
    g_resolved = sum(1 for r in rows if r["graph_status"]    == "patched")
    n = len(rows)

    b_toks = [_tok(baseline[r["instance_id"]]) for r in rows
              if r["baseline_status"] == "patched" and _tok(baseline.get(r["instance_id"], {})) > 0]
    g_toks = [_tok(graph[r["instance_id"]]) for r in rows
              if r["graph_status"] == "patched" and _tok(graph.get(r["instance_id"], {})) > 0]

    avg_b = sum(b_toks) / len(b_toks) if b_toks else 0
    avg_g = sum(g_toks) / len(g_toks) if g_toks else 0

    only_graph_solved = sum(1 for r in rows if r["graph_status"] == "patched" and r["baseline_status"] != "patched")
    both_solved       = sum(1 for r in rows if r["graph_status"] == "patched" and r["baseline_status"] == "patched")

    # -----------------------------------------------------------------------
    # Markdown report
    # -----------------------------------------------------------------------
    lines = [
        "# A/B Comparison: Claude Code Alone vs Claude Code + Repo-Insight",
        "",
        "## Headline Numbers",
        "",
        f"| Metric | Baseline (no graph) | Graph-Enhanced (Repo-Insight) | Delta |",
        f"|---|---|---|---|",
        f"| Resolution rate | {b_resolved}/{n} ({100*b_resolved/n:.1f}%) | **{g_resolved}/{n} ({100*g_resolved/n:.1f}%)** | **+{g_resolved-b_resolved} instances** |",
        f"| Avg tokens/resolved | {avg_b:,.0f} | **{avg_g:,.0f}** | **{100*(avg_b-avg_g)/max(avg_b,1):.0f}% fewer** |",
        f"| Solved only with graph | — | **{only_graph_solved}** | — |",
        f"| Solved by both | {both_solved} | {both_solved} | — |",
        "",
        "---",
        "",
        "## Bob Demo Shortlist (Top 5 — ranked by graph advantage)",
        "",
        "These are the best instances to demo with Bob. Run them manually with Bob + Repo-Insight.",
        "",
    ]

    shortlist = [r for r in rows if r["demo_score"] > 0][:5]
    for i, r in enumerate(shortlist, 1):
        iid = r["instance_id"]
        b_r = baseline.get(iid, {})
        g_r = graph.get(iid, {})

        lines += [
            f"### {i}. `{iid}`",
            f"**Demo score: {r['demo_score']} — {r['demo_reason']}**",
            "",
            f"| | Baseline | Graph-Enhanced |",
            f"|---|---|---|",
            f"| Status | {r['baseline_status']} | **{r['graph_status']}** |",
            f"| Tokens | {r['baseline_tokens']:,} | **{r['graph_tokens']:,}** |",
            f"| Files read | {r['baseline_files']} | **{r['graph_files']}** |",
            f"| Tools called | — | `{'`, `'.join(r['tools_called']) if r['tools_called'] else 'none'}` |",
            "",
            f"**To run with Bob:**",
            f"```",
            f"Instance: {iid}",
            f"Repo: {graph.get(iid, {}).get('repo', 'unknown')}",
            f"```",
            "",
            f"Log (graph): `{g_r.get('agent_output_log', 'see agent_logs/')}` ",
            f"Log (baseline): `{b_r.get('agent_output_log', 'see agent_logs/')}`",
            "",
        ]

    # Full comparison table
    lines += [
        "---",
        "",
        "## All Instances",
        "",
        "| Instance | Baseline | Graph | Tokens Saved | Tools Used | Demo Score |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        tools_short = ", ".join(r["tools_called"][:3]) + ("…" if len(r["tools_called"]) > 3 else "")
        tok_str = f"{r['tokens_saved_pct']}%" if r.get("tokens_saved_pct") else "—"
        lines.append(
            f"| `{r['instance_id']}` "
            f"| {r['baseline_status']} "
            f"| {r['graph_status']} "
            f"| {tok_str} "
            f"| {tools_short or '—'} "
            f"| {r['demo_score']} |"
        )

    report = "\n".join(lines)
    Path(output_path).write_text(report)
    print(report)
    print(f"\n✓ Report saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare A/B SWE-bench results")
    parser.add_argument("--baseline", required=True, help="Path to baseline results.json")
    parser.add_argument("--graph",    required=True, help="Path to graph-enhanced results.json")
    parser.add_argument("--output",   default="./results/comparison.md",
                        help="Output markdown report path")
    args = parser.parse_args()

    print("Loading results...")
    baseline = load(args.baseline)
    graph    = load(args.graph)
    print(f"Baseline: {len(baseline)} instances | Graph: {len(graph)} instances")

    compare(baseline, graph, args.output)


if __name__ == "__main__":
    main()
