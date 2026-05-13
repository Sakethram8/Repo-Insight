# change_engine.py
"""
6-Phase Graph-Driven Coding Agent Pipeline.

Phase 0: Graph Construction (ensure fresh graph)
Phase 1: Seed Localization (semantic search + LLM)
Phase 2: Structural Expansion (deterministic graph traversal)
Phase 3: Graph-Constrained Planning (LLM + validation gate)
Phase 4: Surgical Editing (LLM produces SEARCH/REPLACE blocks)
Phase 5: Verified Apply + Graph Re-Analysis
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import openai
import falkordb

from config import SGLANG_BASE_URL, SGLANG_API_KEY, LLM_MODEL, LLM_PROVIDER, BLAST_RADIUS_MAX_DEPTH, IMPACT_RADIUS_MAX_DEPTH
from graph_index import GraphIndex
from ingest import get_connection, run_ingestion, reingest_files
from sandbox import SandboxManager
from tools import (
    get_upstream_callers, get_downstream_deps, get_callers, get_callees,
    get_source_code, semantic_search,
)
from apply_changes import (
    EditBlock, ApplyResult, RunResult,
    parse_edit_blocks, apply_edits, run_tests,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ChangeSubgraph:
    """Graph-derived context assembled in Phase 2."""
    seed_nodes: list[str]
    blast_radius_nodes: list[dict] = field(default_factory=list)
    impact_radius_nodes: list[dict] = field(default_factory=list)
    caller_nodes: list[dict] = field(default_factory=list)
    callee_nodes: list[dict] = field(default_factory=list)
    all_affected_files: set[str] = field(default_factory=set)
    source_code: dict[str, str] = field(default_factory=dict)
    edges: list[dict] = field(default_factory=list)


@dataclass
class ChangePlan:
    """LLM's plan, validated against graph blast radius."""
    planned_files: set[str] = field(default_factory=set)
    blast_radius_files: set[str] = field(default_factory=set)
    missing_files: set[str] = field(default_factory=set)
    is_validated: bool = False
    justifications: dict[str, str] = field(default_factory=dict)
    actions: dict[str, str] = field(default_factory=dict)  # file → "modify"|"no_change"
    raw_plan: str = ""


@dataclass
class ChangeResult:
    """Full result of the 6-phase pipeline."""
    ingestion_report: dict = field(default_factory=dict)
    seeds: list[str] = field(default_factory=list)
    subgraph: Optional[ChangeSubgraph] = None
    plan: Optional[ChangePlan] = None
    edits: list[EditBlock] = field(default_factory=list)
    apply_result: Optional[ApplyResult] = None
    test_result: Optional[RunResult] = None
    post_edit_analysis: Optional[dict] = None
    answer: str = ""
    phases_completed: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

LOCALIZATION_PROMPT = """You are a code analysis expert. Given a user's change request and a list of candidate code entities found via semantic search, identify the PRIMARY SEED NODES — the specific functions or classes that are the direct targets of the change.

User request: {prompt}

Candidate entities from semantic search:
{candidates}

Return a JSON array of the FQNs (fully qualified names) that are the PRIMARY targets. Only include nodes that directly need to be modified — not their dependents (we will find those automatically via graph traversal).

Return ONLY the JSON array. Your entire response must start with [ and end with ]. No explanation, no prose, no markdown. Example: ["parser.FunctionDef", "parser.parse_file"]"""

PLANNING_PROMPT = """You are a structural code change planner. You MUST plan changes that cover ALL affected files.

## User Request
{prompt}

## Change Subgraph (from dependency graph analysis)
The following functions/classes are structurally connected to the change target:

### Blast Radius (upstream — THESE WILL BREAK if you don't update them):
{blast_radius}

### Impact Radius (downstream — these are called by the target):
{impact_radius}

### Source Code of Affected Entities:
{source_code_section}

## Instructions
1. For EACH file in the blast radius, decide: does it need code changes or not?
2. Output a JSON array of objects with this schema:
   [{{"file": "path/to/file.py", "action": "modify"|"no_change", "reason": "why"}}]
3. If a file needs no changes, set action to "no_change" and explain WHY.
4. You MUST address every file in the blast radius. Missing files will be flagged.

Return ONLY the JSON array. Your entire response must start with [ and end with ]. No prose, no markdown fences, no explanation outside the array."""

VALIDATION_PROMPT = """The dependency graph shows these files also depend on the code you're changing, but your plan didn't address them:

Missing files: {missing_files}

Their source code:
{missing_source}

For each missing file, either:
1. Add it to your plan with the required changes
2. Explain why it doesn't need changes

Return an updated JSON array covering ALL files (original plan + missing files):
[{{"file": "...", "action": "modify"|"no_change", "reason": "..."}}]"""

EDIT_PROMPT = """You are a precise code editor. Produce SEARCH/REPLACE blocks for each file that needs modification.

## User Request
{prompt}

## Change Plan
{plan}

## Source Code
{source_code_section}

## Output Format
For EACH file that needs changes, output one or more blocks in this EXACT format:

FILE: path/to/file.py
<<<<<<< SEARCH
exact existing code to find (copy from source above)
=======
replacement code with your changes
>>>>>>> REPLACE

RULES:
- The SEARCH block MUST be an EXACT copy of existing code from the source above
- Include 3+ lines of surrounding context for unique matching
- One block per change region. Multiple blocks per file are fine.
- Do NOT use line numbers. The system finds the text automatically.
- Produce blocks for ALL files marked \"modify\" in the plan.
- Do NOT write any explanation before or after the blocks. Output ONLY the FILE/SEARCH/REPLACE blocks.
- Do NOT wrap blocks in markdown code fences (no ```python or ``` around them).

EXAMPLE (do not copy — use actual code from the source above):
FILE: astropy/modeling/separable.py
<<<<<<< SEARCH
def _separable(transform):
    if isinstance(transform, CompoundModel):
        return _separable(transform.left) & _separable(transform.right)
    return np.ones((transform.n_outputs, transform.n_inputs), dtype=np.int32)
=======
def _separable(transform):
    if isinstance(transform, CompoundModel):
        left = _separable(transform.left)
        right = _separable(transform.right)
        return _operators[transform.op](left, right)
    return np.ones((transform.n_outputs, transform.n_inputs), dtype=np.int32)
>>>>>>> REPLACE"""

RETRY_PROMPT = """The changes you proposed failed testing. Here are the errors:

{test_errors}

The original source code of affected files:
{source_code_section}

Please produce corrected SEARCH/REPLACE blocks that fix these errors.
Use the same FILE: ... <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE format."""

#Helper
def _clean_seed(seed: str) -> str | None:
    """
    Validate and clean a seed FQN from Phase 1.
    Rejects raw traceback lines and exception messages.
    Extracts the function name if a traceback is detected.
    """
    if not seed or not isinstance(seed, str):
        return None

    # Detect traceback format: 'File "path.py", line N, in func_name'
    traceback_match = re.search(r'\bin (\w+)\s*$', seed.split('\n')[0])
    if 'File "' in seed and traceback_match:
        # Extract just the function name — return it for re-resolution
        return traceback_match.group(1)

    # Reject multi-line strings (raw exceptions, stack traces)
    if '\n' in seed:
        return None

    # Reject exception message patterns
    if any(kw in seed for kw in ['raise ', 'Error:', 'Exception:', 'Traceback']):
        return None

    return seed
# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class GraphDrivenEngine:
    """Orchestrates the 6-phase graph-driven coding pipeline."""

    def __init__(
        self,
        repo_root: Path,
        graph: Optional[falkordb.Graph] = None,
        sandbox_path: Path | None = None,
        swebench_tests: list[str] | None = None,
        index: Optional[GraphIndex] = None,
    ):
        self.original_root = Path(repo_root).resolve()
        self.repo_root = Path(sandbox_path).resolve() if sandbox_path is not None else self.original_root
        self.graph = graph
        self.index = index  # Optional in-memory adjacency cache — speeds up Phase 2
        # FAIL_TO_PASS test IDs from SWEbench — if set, Phase 5 runs only these.
        self.swebench_tests: list[str] = swebench_tests or []
        import httpx
        self.client = openai.OpenAI(
            base_url=SGLANG_BASE_URL,
            api_key=SGLANG_API_KEY,
            timeout=httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=5.0),
        )
        self._on_phase: Optional[callable] = None  # callback for streaming
        # Cache provider so _extra() doesn't re-read env every call
        self._provider = LLM_PROVIDER

    def _extra(self, thinking: bool) -> dict:
        """Return provider-specific extra_body for thinking mode.
        SGLang/Qwen3 uses chat_template_kwargs; standard OpenAI-compatible
        providers accept no extra params — returning {} is safe for all."""
        if self._provider == "sglang":
            return {"chat_template_kwargs": {"enable_thinking": thinking}}
        return {}

    def run_from_diff(
        self,
        ref: str = "HEAD",
        on_phase: Optional[callable] = None,
        skip_apply: bool = False,
        on_event=None,
    ) -> ChangeResult:
        """Mode D: seed the pipeline from git diff instead of a user prompt.

        Steps:
          1. Run git_diff_impact to find changed files and at-risk callers.
          2. Collect FQNs of functions in changed files as seeds.
          3. Build a user_prompt describing what changed.
          4. Run Phase 2–5 from those seeds (skip Phase 0–1).
        """
        from git_tools import git_diff_impact

        self._on_phase = on_phase
        result = ChangeResult()

        try:
            # --- Collect impact from git diff ---
            self._notify("phase_0", {"status": "running git diff impact..."})
            diff_report = git_diff_impact(
                ref=ref,
                repo_root=str(self.original_root),
                graph=self.graph,
            )
            result.ingestion_report = {"git_diff": diff_report, "ref": ref}
            result.phases_completed.append("Phase 0: Git Diff Impact")
            self._notify("phase_0", diff_report)

            # --- Build seeds from changed functions ---
            seeds: list[str] = []
            # Collect from interface_breaks (functions whose signatures changed)
            for fqn_info in diff_report.get("interface_breaks", {}).values():
                if isinstance(fqn_info, dict):
                    fqn = fqn_info.get("fqn") or fqn_info.get("seed", "")
                    if fqn:
                        seeds.append(fqn)
            # Collect from deleted_function_breaks
            for fqn in diff_report.get("deleted_function_breaks", {}).keys():
                if fqn not in seeds:
                    seeds.append(fqn)
            # Fall back: use at-risk callers as seeds if no breaks found
            if not seeds:
                for entry in diff_report.get("at_risk_callers", [])[:5]:
                    if isinstance(entry, dict) and entry.get("fqn"):
                        seeds.append(entry["fqn"])

            if not seeds:
                result.error = (
                    f"git diff against {ref} found no broken interfaces or at-risk callers. "
                    "Nothing to fix."
                )
                return result

            result.seeds = seeds
            result.phases_completed.append("Phase 1: Seeds from Git Diff")
            self._notify("phase_1", {"seeds": seeds, "source": f"git diff {ref}"})

            # Build a descriptive prompt from the diff report
            changed = diff_report.get("changed_files", [])
            deleted = diff_report.get("deleted_files", [])
            user_prompt = (
                f"Fix breaking changes introduced by `git diff {ref}`. "
                f"Changed files: {', '.join(changed[:5])}. "
                f"Deleted files: {', '.join(deleted[:3])}. "
                f"Total at-risk callers: {diff_report.get('total_at_risk', 0)}. "
                f"Update all callers to match the new interfaces."
            )

            # --- Phase 2–5 (reuse existing pipeline, skip Phase 0–1) ---
            result.subgraph = self._expand_subgraph(seeds)
            result.phases_completed.append("Phase 2: Structural Expansion")
            self._notify("phase_2", {
                "blast_radius_count": len(result.subgraph.blast_radius_nodes),
                "files_affected": len(result.subgraph.all_affected_files),
            })

            result.plan = self._plan_with_validation(user_prompt, result.subgraph, on_event=on_event)
            result.phases_completed.append("Phase 3: Graph-Constrained Planning")
            self._notify("phase_3", {
                "planned_files": len(result.plan.planned_files),
                "is_validated": result.plan.is_validated,
            })

            result.edits, result.answer = self._generate_edits(user_prompt, result.plan, result.subgraph)
            result.phases_completed.append("Phase 4: Surgical Editing")
            self._notify("phase_4", {"edit_blocks": len(result.edits)})

            if not skip_apply and result.edits:
                result.apply_result, result.test_result, result.post_edit_analysis = (
                    self._apply_and_verify(result.edits, user_prompt, result.subgraph, on_event=on_event)
                )
                result.phases_completed.append("Phase 5: Verified Apply")

        except Exception as e:
            logger.error("run_from_diff pipeline failed: %s", e, exc_info=True)
            result.error = str(e)

        return result

    def run(
        self,
        user_prompt: str,
        on_phase: Optional[callable] = None,
        skip_apply: bool = False,
        _skip_phase0: bool = False,
        on_event=None,
    ) -> ChangeResult:
        """Execute the full 6-phase pipeline.

        Args:
            user_prompt: The user's code change request.
            on_phase: Optional callback(phase_name, phase_data) for live UI updates.
            skip_apply: If True, skip Phase 5 (sandbox/test). Useful for scoring.
        """
        self._on_phase = on_phase
        result = ChangeResult()

        try:
            self._notify("sandbox_active", {
                "sandboxed": self.repo_root != self.original_root,
                "sandbox_path": str(self.repo_root),
                "original_path": str(self.original_root),
            })

            # Phase 0: Graph Construction
            if _skip_phase0:
                result.ingestion_report = {"skipped": True, "reason":"pre-ingested by harness"}
            else:
                t0 = time.time()
                result.ingestion_report = self._ensure_graph_fresh()
                result.timings["phase_0_graph"] = time.time() - t0
            result.phases_completed.append("Phase 0: Graph Construction")
            self._notify("phase_0", result.ingestion_report)

            # Phase 1: Seed Localization
            t1 = time.time()
            result.seeds = self._localize_seeds(user_prompt)
            result.timings["phase_1_localize"] = time.time() - t1
            result.phases_completed.append("Phase 1: Seed Localization")
            self._notify("phase_1", {"seeds": result.seeds})

            if not result.seeds:
                result.error = "Could not identify any seed nodes for this request."
                return result

            # Phase 2: Structural Expansion
            t2 = time.time()
            result.subgraph = self._expand_subgraph(result.seeds)
            result.timings["phase_2_expand"] = time.time() - t2
            result.phases_completed.append("Phase 2: Structural Expansion")
            self._notify("phase_2", {
                "blast_radius_count": len(result.subgraph.blast_radius_nodes),
                "files_affected": len(result.subgraph.all_affected_files),
                "source_files_loaded": len(result.subgraph.source_code),
            })
            # Guard: if Phase 2 found nothing, the seeds are wrong — fail fast
            if not result.subgraph.all_affected_files and not result.subgraph.source_code:
                result.error = (
                    f"Phase 2 found zero affected files for seeds: {result.seeds}. "
                    "Semantic search likely returned wrong entry points. Aborting."
                )
                logger.error("[Phase 2 Guard] %s", result.error)
                return result

            # Phase 3: Graph-Constrained Planning
            t3 = time.time()
            result.plan = self._plan_with_validation(user_prompt, result.subgraph, on_event=on_event)
            result.timings["phase_3_plan"] = time.time() - t3
            result.phases_completed.append("Phase 3: Graph-Constrained Planning")
            self._notify("phase_3", {
                "planned_files": len(result.plan.planned_files),
                "missing_files": len(result.plan.missing_files),
                "is_validated": result.plan.is_validated,
            })
            # Guard: if plan is empty, don't waste time on Phase 4
            if not result.plan.planned_files:
                logger.warning("Phase 3 JSON parse failed. Foricng modification of all blast radius files.")
                for fp in result.plan.blast_radius_files:
                    result.plan.planned_files.add(fp)
                    result.plan.actions[fp]='modify'
                    result.plan.justifications[fp]= "Emergency fallback : JSON planning unparseable"
                result.plan.is_validated=True
                result.error = (
                "Phase 3 returned an empty plan (no files to modify). "
                "LLM likely returned {} or unparseable JSON. Aborting."
                )
                return result

            # Phase 4: Surgical Editing
            t4 = time.time()
            result.edits, result.answer = self._generate_edits(
                user_prompt, result.plan, result.subgraph,
            )
            result.timings["phase_4_edit"] = time.time() - t4
            result.phases_completed.append("Phase 4: Surgical Editing")
            self._notify("phase_4", {"edit_blocks": len(result.edits)})

            # Phase 5: Verified Apply
            if not skip_apply and result.edits:
                t5 = time.time()
                result.apply_result, result.test_result, result.post_edit_analysis = (
                    self._apply_and_verify(result.edits, user_prompt, result.subgraph, on_event=on_event)
                )
                result.timings["phase_5_verify"] = time.time() - t5
                result.phases_completed.append("Phase 5: Verified Apply")
                self._notify("phase_5", {
                    "apply_success": result.apply_result.all_succeeded if result.apply_result else False,
                    "tests_passed": result.test_result.all_passed if result.test_result else False,
                    "new_dependencies": result.post_edit_analysis,
                })

        except Exception as e:
            logger.error("Pipeline failed: %s", e, exc_info=True)
            result.error = str(e)

        return result

    # ------------------------------------------------------------------
    # Phase 0: Graph Construction
    # ------------------------------------------------------------------

    def _ensure_graph_fresh(self) -> dict:
        """Build or refresh the code graph. Skips ingestion if no files have changed."""
        from config import SKIP_DIRS

        # Fingerprint from content hashes — not mtimes.
        # git checkout resets every file's mtime to now, so mtime-based
        # fingerprints always differ between clones even for identical content.
        from ingest import _file_content_hash
        current_hashes = {}
        for f in sorted(self.original_root.rglob("*.py")):
            parts = f.relative_to(self.original_root).parts
            if not any(p in SKIP_DIRS for p in parts):
                rel = str(f.relative_to(self.original_root))
                current_hashes[rel] = _file_content_hash(f)

        fingerprint = str(hash(frozenset(current_hashes.items())))

        # Check stored fingerprint in graph Meta node
        stored = None
        if self.graph is not None:
            try:
                res = self.graph.query(
                    "MATCH (m:Meta {key: 'repo_fingerprint'}) RETURN m.value LIMIT 1"
                ).result_set
                stored = res[0][0] if res else None
            except Exception:
                pass

        if stored == fingerprint:
            return {"skipped": True, "reason": "no file changes detected", "fingerprint": fingerprint}

        # Run ingestion and store new fingerprint
        report = run_ingestion(str(self.repo_root))
        self.graph = get_connection()

        try:
            self.graph.query(
                "MERGE (m:Meta {key: 'repo_fingerprint'}) SET m.value = $v",
                {"v": fingerprint}
            )
        except Exception as e:
            logger.warning("Could not store repo fingerprint: %s", e)

        # Rebuild in-memory index to reflect the fresh graph
        if self.index is not None:
            try:
                self.index.rebuild(self.graph)
            except Exception as e:
                logger.warning("GraphIndex rebuild after ingestion failed: %s", e)

        return report

    # ------------------------------------------------------------------
    # Phase 1: Seed Localization
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _localize_seeds(self, prompt: str) -> list[str]:
        """Use LLM + graph semantic search to identify entry points."""
        search_result = semantic_search(prompt, self.graph, top_k=40)
        candidates = search_result.get("results", [])

        if not candidates:
            return []

        candidates_text = "\n".join(
            f"- {c['fqn']} ({c['label']}, file: {c['file_path']}, score: {c.get('score', 1.0)})"
            for c in candidates
        )

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a code analysis expert. When a bug report mentions a crash "
                            "with a traceback, always prefer seed nodes from the TRACEBACK FILE PATHS "
                            "over test framework classes. The traceback shows where the bug IS, "
                            "not where it was triggered from."
                        )
                    },
                    {
                        "role": "user",
                        "content": LOCALIZATION_PROMPT.format(
                            prompt=prompt, candidates=candidates_text,
                        ),
                    }
                ],
                max_tokens=1024,
                # No response_format here — prompt asks for a JSON array [...], not object {...}.
                # json_object would force the model to wrap it, breaking direct array parsing.
                extra_body=self._extra(False),
            )
            raw = response.choices[0].message.content.strip()
            # Extract JSON array from response, tolerant of <think> blocks
            seeds = self._extract_json_array(raw)
            if seeds:
                raw_seeds = [s for s in seeds if isinstance(s,str)]
                #Clean each seed -reject tracebacks, extract function names
                cleaned = [_clean_seed(s) for s in raw_seeds]
                cleaned = [s for s in cleaned if s]
                if cleaned:
                    return cleaned
        except Exception as e:
            logger.error("Seed localization LLM call failed: %s", e)

        # Fallback: use top 3 semantic search results
        return [c["fqn"] for c in candidates[:3]]

    # ------------------------------------------------------------------
    # Phase 2: Structural Expansion (NO LLM)
    # ------------------------------------------------------------------

    def _expand_subgraph(self, seeds: list[str]) -> ChangeSubgraph:
        """Deterministically expand seeds into a full change subgraph.

        Uses GraphIndex (in-memory BFS) when available, otherwise falls back
        to FalkorDB Cypher queries — same results, zero round-trips with index.
        """
        subgraph = ChangeSubgraph(seed_nodes=seeds)
        seen_fqns: set[str] = set()
        fqn_depths: dict[str, int] = {}

        for seed in seeds:
            fqn_depths[seed] = 0

            # --- Blast radius (upstream — what breaks) ---
            if self.index is not None:
                blast_nodes_list = self.index.blast_radius(seed, max_depth=BLAST_RADIUS_MAX_DEPTH)
            else:
                blast_nodes_list = get_upstream_callers(
                    seed, self.graph, max_depth=BLAST_RADIUS_MAX_DEPTH
                ).get("affected", [])

            for node in blast_nodes_list:
                fqn = node.get("fqn", "")
                if fqn:
                    depth = node.get("depth", node.get("distance", 0))
                    fqn_depths[fqn] = min(fqn_depths.get(fqn, depth), depth)
                    if fqn not in seen_fqns:
                        subgraph.blast_radius_nodes.append(node)
                        seen_fqns.add(fqn)
                        if node.get("file_path"):
                            subgraph.all_affected_files.add(node["file_path"])

            # --- Impact radius (downstream) ---
            if self.index is not None:
                impact_nodes_list = self.index.impact_radius(seed, max_depth=IMPACT_RADIUS_MAX_DEPTH)
            else:
                impact_nodes_list = get_downstream_deps(
                    seed, self.graph, max_depth=IMPACT_RADIUS_MAX_DEPTH
                ).get("impacted", [])

            for node in impact_nodes_list:
                fqn = node.get("fqn", "")
                if fqn:
                    depth = node.get("depth", node.get("distance", 0))
                    fqn_depths[fqn] = min(fqn_depths.get(fqn, depth), depth)
                    if fqn not in seen_fqns:
                        subgraph.impact_radius_nodes.append(node)
                        seen_fqns.add(fqn)
                        if node.get("file_path"):
                            subgraph.all_affected_files.add(node["file_path"])

            # --- Direct callers/callees ---
            if self.index is not None:
                # Build minimal node dicts from the in-memory maps
                with self.index._lock:
                    direct_callers = list(self.index.callers.get(seed, set()))
                    direct_callees = list(self.index.callees.get(seed, set()))
                    meta_snap = self.index.fn_meta
                for caller_fqn in direct_callers:
                    m = meta_snap.get(caller_fqn, {})
                    c = {"fqn": caller_fqn, "file_path": m.get("file_path", ""), "depth": 1}
                    fqn_depths[caller_fqn] = min(fqn_depths.get(caller_fqn, 1), 1)
                    if caller_fqn not in seen_fqns:
                        subgraph.caller_nodes.append(c)
                        seen_fqns.add(caller_fqn)
                for callee_fqn in direct_callees:
                    m = meta_snap.get(callee_fqn, {})
                    c = {"fqn": callee_fqn, "file_path": m.get("file_path", ""), "depth": 1}
                    fqn_depths[callee_fqn] = min(fqn_depths.get(callee_fqn, 1), 1)
                    if callee_fqn not in seen_fqns:
                        subgraph.callee_nodes.append(c)
                        seen_fqns.add(callee_fqn)
            else:
                callers = get_callers(seed, self.graph)
                for c in callers.get("callers", []):
                    fqn = c.get("fqn", "")
                    if fqn:
                        fqn_depths[fqn] = min(fqn_depths.get(fqn, 1), 1)
                        if fqn not in seen_fqns:
                            subgraph.caller_nodes.append(c)
                            seen_fqns.add(fqn)

                callees = get_callees(seed, self.graph)
                for c in callees.get("callees", []):
                    fqn = c.get("fqn", "")
                    if fqn:
                        fqn_depths[fqn] = min(fqn_depths.get(fqn, 1), 1)
                        if fqn not in seen_fqns:
                            subgraph.callee_nodes.append(c)
                            seen_fqns.add(fqn)

            # Add seed's own file
            ctx = get_source_code(seed, self.graph, repo_root_override=str(self.repo_root))
            if ctx.get("found"):
                subgraph.source_code[seed] = ctx.get("source", "")
                if ctx.get("file_path"):
                    subgraph.all_affected_files.add(ctx["file_path"])

        # Fetch source code for all affected nodes
        for fqn in list(seen_fqns):
            if fqn not in subgraph.source_code:
                if fqn_depths.get(fqn, 0) <= 3:
                    src = get_source_code(fqn, self.graph, repo_root_override=str(self.repo_root))
                    if src.get("found"):
                        subgraph.source_code[fqn] = src.get("source", "")

        return subgraph

    # ------------------------------------------------------------------
    # Phase 3: Graph-Constrained Planning + Validation Gate
    # ------------------------------------------------------------------

    def _plan_with_validation(self, prompt: str, subgraph: ChangeSubgraph, on_event=None) -> ChangePlan:
        """Get LLM change plan and validate it against blast radius."""
        plan = ChangePlan()
        plan.blast_radius_files = set(subgraph.all_affected_files)

        source_section = self._format_source_section(subgraph)
        blast_text = self._format_node_list(subgraph.blast_radius_nodes)
        impact_text = self._format_node_list(subgraph.impact_radius_nodes)

        # First LLM call: get initial plan
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{
                    "role": "user",
                    "content": PLANNING_PROMPT.format(
                        prompt=prompt,
                        blast_radius=blast_text,
                        impact_radius=impact_text,
                        source_code_section=source_section,
                    ),
                }],
                max_tokens=16384,
                temperature=0.6,
                top_p=0.95,
                presence_penalty=0.0,
                # Thinking enabled — planning requires reasoning about blast radius.
                # Prompt asks for JSON array [...] — no response_format to avoid wrapping.
                extra_body=self._extra(True),
            )
            raw = response.choices[0].message.content.strip()
            plan.raw_plan = raw
            plan_items = self._parse_plan_json(raw)
            #If parsing produced nothing, return immediately
            if not plan_items:
                logger.warning("Phase 3: _parse_plan_json returned empty -auto-planning all blast-radius files")
                plan_items = [{"file": f, "action": "modify", "reason" : "auto-fallback : JSON parse failed"} for f in plan.blast_radius_files]
        except Exception as e:
            logger.error("Planning LLM call failed: %s", e)
            # Fallback: plan all affected files for modification
            plan_items = [{"file": f, "action": "modify", "reason": "fallback"} for f in plan.blast_radius_files]

        # Extract planned files
        for item in plan_items:
            if not isinstance(item, dict):
                logger.warning("Skipping invalid plan item (not a dict): %s", item)
                continue
            fp = item.get("file", "") or item.get("path", "") or item.get("filename", "")
            action=item.get("action", "") or item.get("type", "modify")
            if not fp:
                continue
            plan.planned_files.add(fp)
            plan.justifications[fp] = item.get("reason", "")
            plan.actions[fp] = action

        # VALIDATION GATE: retry loop up to MAX_VALIDATION_ROUNDS
        MAX_VALIDATION_ROUNDS = 3
        for round_num in range(MAX_VALIDATION_ROUNDS):
            plan.missing_files = plan.blast_radius_files - plan.planned_files
            if not plan.missing_files:
                break
            if on_event:
                on_event("phase_3_recovery", f"LLM Missed {(plan.missing_files)}blast-radius file(s). Graph engine intercepting and forcing recovery (Round {round_num + 1})")
            logger.info(
                "Validation round %d: %d files still missing, forcing coverage",
                round_num + 1, len(plan.missing_files)
            )
            plan = self._force_coverage(prompt, plan, subgraph)

        plan.missing_files = plan.blast_radius_files - plan.planned_files

        # Convergence fallback: if LLM still didn't cover all files, force-add them
        if plan.missing_files:
            logger.warning(
                "Convergence fallback: force-adding %d uncovered files as modify: %s",
                len(plan.missing_files), plan.missing_files,
            )
            for fp in list(plan.missing_files):
                plan.planned_files.add(fp)
                plan.justifications[fp] = "force-added by convergence fallback"
                plan.actions[fp] = "modify"
            plan.missing_files = set()

        plan.is_validated = len(plan.missing_files) == 0
        return plan

    def _force_coverage(self, prompt: str, plan: ChangePlan, subgraph: ChangeSubgraph) -> ChangePlan:
        """Force LLM to address files the graph says are affected but plan missed."""
        if self._on_phase:
            self._on_phase("self_correction", {
                "missing_files": list(plan.missing_files),
                "trigger": "LLM plan did not cover all blast-radius files",
                "recovering": f"{len(plan.missing_files)} file(s) via graph lookup",
            })

        # Build a file_path → [fqn] map from graph node metadata
        file_to_fqns: dict[str, list[str]] = {}
        all_nodes = (subgraph.blast_radius_nodes + subgraph.caller_nodes
                     + subgraph.impact_radius_nodes + subgraph.callee_nodes)
        for node in all_nodes:
            fp = node.get("file_path", "")
            fqn = node.get("fqn") or node.get("name", "")
            if fp and fqn:
                file_to_fqns.setdefault(fp, []).append(fqn)

        missing_source: dict[str, str] = {}
        for mf in plan.missing_files:
            for fqn in file_to_fqns.get(mf, []):
                if fqn in subgraph.source_code:
                    missing_source[fqn] = subgraph.source_code[fqn]

        _parts, _total = [], 0
        for fqn, src in missing_source.items():
            chunk = f"### {fqn}\n```python\n{src[:2000]}\n```"
            if _total + len(chunk) > 20_000:
                break
            _parts.append(chunk)
            _total += len(chunk)
        missing_src_text = "\n\n".join(_parts)

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "user", "content": PLANNING_PROMPT.format(
                        prompt=prompt,
                        blast_radius=self._format_node_list(subgraph.blast_radius_nodes),
                        impact_radius=self._format_node_list(subgraph.impact_radius_nodes),
                        source_code_section=self._format_source_section(subgraph),
                    )},
                    {"role": "assistant", "content": plan.raw_plan},
                    {"role": "user", "content": VALIDATION_PROMPT.format(
                        missing_files=", ".join(plan.missing_files),
                        missing_source=missing_src_text,
                    )},
                ],
                max_tokens=16384,
                temperature=0.6,
                top_p=0.95,
                presence_penalty=0.0,
                # Thinking ON — force-coverage needs careful reasoning about missing files.
                extra_body=self._extra(True),
            )
            raw = response.choices[0].message.content.strip()
            plan_items = self._parse_plan_json(raw)

            plan.planned_files.clear()
            plan.justifications.clear()
            plan.actions.clear()
            for item in plan_items:
                if not isinstance(item, dict):
                    logger.warning("Skipping invalid force-coverage plan item (not a dict): %s", item)
                    continue
                fp = item.get("file", "")
                if not fp:
                    continue
                plan.planned_files.add(fp)
                plan.justifications[fp] = item.get("reason", "")
                plan.actions[fp] = item.get("action", "modify")

            plan.missing_files = plan.blast_radius_files - plan.planned_files
        except Exception as e:
            logger.error("Validation force-coverage failed: %s", e)

        return plan

    # ------------------------------------------------------------------
    # Phase 4: Surgical Editing
    # ------------------------------------------------------------------

    def _generate_edits(
        self, prompt: str, plan: ChangePlan, subgraph: ChangeSubgraph,
    ) -> tuple[list[EditBlock], str]:
        """Generate SEARCH/REPLACE blocks for all planned modifications."""
        modify_files = {
            f for f, action in plan.actions.items()
            if action != "no_change" and f in plan.planned_files
        }

        plan_text = "\n".join(
            f"- {f}: {plan.justifications.get(f, 'modify')}" for f in modify_files
        )

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{
                    "role": "user",
                    "content": EDIT_PROMPT.format(
                        prompt=prompt,
                        plan=plan_text,
                        source_code_section=self._format_source_section(subgraph),
                    ),
                }],
                max_tokens=32768,   # model supports up to 81920 for complex tasks
                temperature=0.6,
                top_p=0.95,
                presence_penalty=0.0,
                extra_body=self._extra(True),  # thinking ON — edit generation is the hardest step
            )
            answer = response.choices[0].message.content.strip()

            # If response was cut off, request continuation
            if response.choices[0].finish_reason == "length":
                logger.warning("Edit generation truncated — requesting continuation")
                try:
                    continuation = self.client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[
                            {"role": "user", "content": EDIT_PROMPT.format(
                                prompt=prompt,
                                plan=plan_text,
                                source_code_section=self._format_source_section(subgraph),
                            )},
                            {"role": "assistant", "content": answer},
                            {"role": "user", "content":
                             "Continue the SEARCH/REPLACE blocks from exactly "
                             "where you left off. Do not repeat blocks already written."},
                        ],
                        max_tokens=32768,
                        temperature=0.6,
                        top_p=0.95,
                        presence_penalty=0.0,
                        extra_body=self._extra(True),
                    )
                    answer += "\n" + continuation.choices[0].message.content.strip()
                except Exception as cont_err:
                    logger.warning("Continuation request failed: %s", cont_err)

            answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
            answer = re.sub(r"```[a-zA-Z]*\n(.*?)```", r"\1", answer, flags=re.DOTALL).strip()
            edits = parse_edit_blocks(answer)
            return edits, answer
        except Exception as e:
            logger.error("Edit generation failed: %s", e)
            return [], f"Edit generation failed: {e}"

    # ------------------------------------------------------------------
    # Phase 5: Apply + Verify + Re-Analysis
    # ------------------------------------------------------------------

    def _apply_and_verify(
        self,
        edits: list[EditBlock],
        prompt: str,
        subgraph: ChangeSubgraph,
        max_retries: int = 2,
        on_event=None,
    ) -> tuple[Optional[ApplyResult], Optional[RunResult], Optional[dict]]:
        """Apply edits directly to repo_root (already sandboxed), run tests, retry on failure."""
        current_edits = edits

        # Snapshot current call edges before any edits
        pre_edit_calls = set()
        try:
            rows = self.graph.query(
                "MATCH (a:Function)-[:CALLS]->(b:Function) RETURN a.fqn, b.fqn"
            ).result_set
            pre_edit_calls = {(r[0], r[1]) for r in rows}
        except Exception:
            pass

        # Build the targeted test command from SWEbench FAIL_TO_PASS specs.
        # Falls back to the configured TEST_COMMAND if no specs are provided.
        if self.swebench_tests:
            import shlex
            _swe_test_cmd = [
                "bash", "-c",
                "pip install -e . -q --no-build-isolation 2>/dev/null; "
                "pytest " + " ".join(shlex.quote(t) for t in self.swebench_tests)
                + " --tb=short -x --no-header -q"
            ]
            _swe_timeout = 300  # pip install alone can take 60s on large packages
            logger.info("Phase 5: using %d FAIL_TO_PASS tests", len(self.swebench_tests))
        else:
            _swe_test_cmd = None
            _swe_timeout = 120

        for attempt in range(1, max_retries + 1):
            # Apply edits
            apply_result = apply_edits(current_edits, self.repo_root)
            if not apply_result.all_succeeded:
                logger.warning("Attempt %d: %d edits failed to apply", attempt, apply_result.failed_edits)
                if attempt == max_retries:
                    return apply_result, None, None

            # Run tests — targeted if SWEbench specs available, generic otherwise
            test_result = run_tests(self.repo_root, test_command=_swe_test_cmd, timeout=_swe_timeout)
            self._notify("phase_5_test", {
                "attempt": attempt,
                "passed": test_result.passed,
                "failed": test_result.failed,
            })

            if test_result.all_passed:
                # Success! Do post-edit graph re-analysis
                post_analysis = self._post_edit_graph_analysis(self.repo_root, subgraph, pre_edit_calls)
                return apply_result, test_result, post_analysis

            # Tests failed — retry with error feedback
            if attempt < max_retries:
                logger.info("Tests failed, retrying (attempt %d/%d)", attempt, max_retries)
                if on_event:
                    on_event(
                        "phase_5_retry", 
                        f"Sandbox tests failed! Graph engine captured pytest output and is forcing LLM to self-correct (Attempt {attempt})..."
                    )
                current_edits = self._retry_edits(prompt, subgraph, test_result)
                if not current_edits:
                    return apply_result, test_result, None

        return apply_result, test_result, None

    def _retry_edits(
        self, prompt: str, subgraph: ChangeSubgraph, test_result: RunResult,
    ) -> list[EditBlock]:
        """Ask LLM to fix edits based on test failures."""
        error_text = test_result.stdout[-2000:] if test_result.stdout else test_result.stderr[-2000:]

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{
                    "role": "user",
                    "content": RETRY_PROMPT.format(
                        test_errors=error_text,
                        source_code_section=self._format_source_section(subgraph),
                    ),
                }],
                max_tokens=32768,
                temperature=0.6,
                top_p=0.95,
                presence_penalty=0.0,
                extra_body=self._extra(True),  # thinking ON for self-correction
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            raw = re.sub(r"```[a-zA-Z]*\n(.*?)```", r"\1", raw, flags=re.DOTALL).strip()
            return parse_edit_blocks(raw)
        except Exception as e:
            logger.error("Retry edit generation failed: %s", e)
            return []

    def _post_edit_graph_analysis(
        self, sandbox: Path, subgraph: ChangeSubgraph, pre_edit_calls: set,
    ) -> dict:
        """Re-ingest changed files into the graph and diff call edges."""
        from ingest import reingest_files

        changed_rel_paths = list(subgraph.all_affected_files)

        try:
            # Step 1: Actually write parsed changes back to FalkorDB
            reingest_report = reingest_files(changed_rel_paths, self.graph, sandbox)
            logger.info("Post-edit re-ingestion: %s", reingest_report)
            if self.index is not None:
                self.index.rebuild(self.graph)
        except Exception as e:
            logger.error("Post-edit reingest failed: %s", e)
            return {"status": "failed", "error": str(e)}

        try:
            # Step 2: Now query for new/removed edges (graph is fresh)
            post_edit_calls = set()
            rows = self.graph.query(
                "MATCH (a:Function)-[:CALLS]->(b:Function) RETURN a.fqn, b.fqn"
            ).result_set
            post_edit_calls = {(r[0], r[1]) for r in rows}

            new_edges = post_edit_calls - pre_edit_calls
            removed_edges = pre_edit_calls - post_edit_calls

            return {
                "files_reingested": reingest_report.get("files_reingested", 0),
                "new_call_edges": [{"from": e[0], "to": e[1]} for e in new_edges],
                "removed_call_edges": [{"from": e[0], "to": e[1]} for e in removed_edges],
                "new_edge_count": len(new_edges),
                "removed_edge_count": len(removed_edges),
                "status": "analyzed",
            }
        except Exception as e:
            logger.error("Post-edit graph diff failed: %s", e)
            return {"status": "failed", "error": str(e)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _notify(self, phase: str, data: Any) -> None:
        if self._on_phase:
            try:
                self._on_phase(phase, data)
            except Exception:
                pass

    def _format_source_section(
        self,
        subgraph: ChangeSubgraph,
        max_chars_per_fn: int = 8000,   # was 3000 — model has 262K token context
        max_total_chars: int = 150_000,  # was 40K — safe headroom below context limit
    ) -> str:
        parts = []
        total = 0
        for fqn, src in sorted(subgraph.source_code.items()):
            display = src if len(src) <= max_chars_per_fn else src[:max_chars_per_fn] + "\n# ... (truncated)"
            chunk = f"### {fqn}\n```python\n{display}\n```"
            if total + len(chunk) > max_total_chars:
                remaining = len(subgraph.source_code) - len(parts)
                parts.append(
                    f"# ... {remaining} more function(s) omitted to stay within context window"
                )
                break
            parts.append(chunk)
            total += len(chunk)
        return "\n\n".join(parts) if parts else "(no source code available)"

    def _format_node_list(self, nodes: list[dict]) -> str:
        if not nodes:
            return "(none)"
        return "\n".join(
            f"- {n.get('fqn', '?')} (file: {n.get('file_path', '?')}, depth: {n.get('depth', '?')})"
            for n in nodes
        )

    def _extract_json_array(self, raw: str) -> list:
        """Parse a JSON array from text, tolerant of reasoning blocks and markdown fences."""
        import re, json
        # Strip Qwen3 <think>...</think> reasoning blocks if present
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        
        # 1. Try to find JSON inside ALL markdown fences
        for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL):
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

        # 2. Try direct parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        # 3. Robust balanced array extraction (try all [ and ] combinations)
        start_indices = [i for i, c in enumerate(text) if c == '[']
        end_indices = [i for i, c in enumerate(text) if c == ']']
        
        for start_idx in start_indices:
            for end_idx in reversed(end_indices):
                if end_idx > start_idx:
                    try:
                        parsed = json.loads(text[start_idx:end_idx+1])
                        if isinstance(parsed, list):
                            return parsed
                    except json.JSONDecodeError:
                        continue
        
        # 4. If all fails, log the raw text so we can debug
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                #Case A : {"items":[]} wrapper
                for v in parsed.values():
                    if isinstance(v, list):
                        return v
                # Single dict that is not an array wrapper — not a valid plan array
        except json.JSONDecodeError:
            pass
        logger.warning("Failed to extract JSON array. Cleaned text snippet: %s", text[:500])
        return []

    def _parse_plan_json(self, raw: str) -> list[dict]:
        """Extract JSON array from LLM response, handling <think> blocks and markdown fences."""
        result = self._extract_json_array(raw)
        # Unwrap [[{...}]] → [{...}] if the LLM double-wrapped the array
        if result and isinstance(result[0], list):
            result = result[0]
        if result:
            return result
        logger.warning("Could not parse plan JSON from LLM output")
        return []
