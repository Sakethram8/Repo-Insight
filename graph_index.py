# graph_index.py
"""
In-memory adjacency index built from FalkorDB at startup.

Eliminates per-query round-trips for multi-hop traversals (blast radius,
impact radius) and semantic search (embeddings cached here — no 81MB Redis
pull per query).

FalkorDB remains the source of truth — this is a read cache that must be
rebuilt after every reingest_files call.

Usage:
    from graph_index import GraphIndex
    idx = GraphIndex.build(graph)        # load from FalkorDB once
    hits = idx.blast_radius("api.connect")
    idx.rebuild(graph)                   # after re-ingest
"""

import json
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field

import falkordb

from config import BLAST_RADIUS_MAX_DEPTH, IMPACT_RADIUS_MAX_DEPTH

logger = logging.getLogger(__name__)


@dataclass
class GraphIndex:
    # call graph — keyed by FQN
    callees:    dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    callers:    dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    # data-flow: module_name → [(reader_fqn, read_name), ...]
    reads:      dict[str, list[tuple[str, str]]] = field(default_factory=lambda: defaultdict(list))
    # inheritance
    bases:      dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    subclasses: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    # metadata
    fn_meta:    dict[str, dict] = field(default_factory=dict)
    module_fns: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    # Fix 4: embedding cache — avoids 81MB Redis pull on every semantic_search
    embeddings:   dict[str, tuple[float, ...]] = field(default_factory=dict)
    summaries:    dict[str, str] = field(default_factory=dict)
    in_degree:    dict[str, int] = field(default_factory=dict)
    fingerprints: dict[str, str] = field(default_factory=dict)
    # Fix 6: thread safety — lock held only during the atomic swap, not during load
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def build(cls, graph: falkordb.Graph) -> "GraphIndex":
        idx = cls()
        idx.rebuild(graph)
        return idx

    def rebuild(self, graph: falkordb.Graph) -> None:
        # Build all new structures into locals — no lock needed during the
        # slow FalkorDB reads (rebuild is typically called from one thread).
        new_callees    = defaultdict(set)
        new_callers    = defaultdict(set)
        new_reads      = defaultdict(list)
        new_bases      = defaultdict(set)
        new_subclasses = defaultdict(set)
        new_fn_meta    = {}
        new_module_fns = defaultdict(list)
        new_embeddings:   dict[str, tuple[float, ...]] = {}
        new_summaries:    dict[str, str] = {}
        new_in_degree:    dict[str, int] = {}
        new_fingerprints: dict[str, str] = {}

        try:
            for row in graph.query(
                "MATCH (a:Function)-[:CALLS]->(b:Function) RETURN a.fqn, b.fqn"
            ).result_set:
                a, b = row[0], row[1]
                new_callees[a].add(b)
                new_callers[b].add(a)
        except Exception as e:
            logger.warning("GraphIndex: failed to load CALLS edges: %s", e)

        try:
            for row in graph.query(
                "MATCH (f:Function)-[r:READS]->(m:Module) RETURN f.fqn, m.name, r.name"
            ).result_set:
                fn_fqn, mod_name, read_name = row[0], row[1], row[2]
                new_reads[mod_name].append((fn_fqn, read_name))
        except Exception as e:
            logger.warning("GraphIndex: failed to load READS edges: %s", e)

        try:
            for row in graph.query(
                "MATCH (c:Class)-[:INHERITS_FROM]->(b:Class) RETURN c.fqn, b.fqn"
            ).result_set:
                c, b = row[0], row[1]
                new_bases[c].add(b)
                new_subclasses[b].add(c)
        except Exception as e:
            logger.warning("GraphIndex: failed to load INHERITS_FROM edges: %s", e)

        try:
            for row in graph.query(
                "MATCH (f:Function) WHERE f.name <> '<module>' "
                "RETURN f.fqn, f.file_path, f.start_line, f.module_name, f.name, f.class_name"
            ).result_set:
                fqn, file_path, start_line, module_name, name, class_name = row
                new_fn_meta[fqn] = {
                    "file_path": file_path,
                    "start_line": start_line,
                    "module_name": module_name,
                    "name": name,
                    "class_name": class_name,
                }
                if module_name:
                    new_module_fns[module_name].append(fqn)
        except Exception as e:
            logger.warning("GraphIndex: failed to load function metadata: %s", e)

        # Fix 4: load embeddings, summaries, and in-degree into memory so
        # semantic_search never needs to pull them from Redis again.
        try:
            for row in graph.query(
                "MATCH (f:Function) WHERE f.embedding IS NOT NULL "
                "RETURN f.fqn, f.embedding, f.summary, f.fingerprint"
            ).result_set:
                fqn, emb_json, summary, fingerprint = row[0], row[1], row[2], row[3]
                try:
                    new_embeddings[fqn] = tuple(json.loads(emb_json))
                except Exception:
                    pass
                new_summaries[fqn] = summary or ""
                if fingerprint:
                    new_fingerprints[fqn] = fingerprint
        except Exception as e:
            logger.warning("GraphIndex: failed to load embeddings: %s", e)

        try:
            for row in graph.query(
                "MATCH (f:Function) WHERE f.fingerprint IS NOT NULL AND f.embedding IS NULL "
                "RETURN f.fqn, f.fingerprint"
            ).result_set:
                fqn, fingerprint = row[0], row[1]
                if fingerprint:
                    new_fingerprints[fqn] = fingerprint
        except Exception as e:
            logger.warning("GraphIndex: failed to load fingerprints: %s", e)

        try:
            for row in graph.query(
                "MATCH ()-[:CALLS]->(f:Function) RETURN f.fqn, count(*)"
            ).result_set:
                new_in_degree[row[0]] = row[1]
        except Exception as e:
            logger.warning("GraphIndex: failed to load in-degree: %s", e)

        # Fix 6: atomic swap — acquire lock only for the brief replacement so
        # concurrent reads see either the fully old or fully new state.
        with self._lock:
            self.callees      = new_callees
            self.callers      = new_callers
            self.reads        = new_reads
            self.bases        = new_bases
            self.subclasses   = new_subclasses
            self.fn_meta      = new_fn_meta
            self.module_fns   = new_module_fns
            self.embeddings   = new_embeddings
            self.summaries    = new_summaries
            self.in_degree    = new_in_degree
            self.fingerprints = new_fingerprints

        logger.info(
            "GraphIndex loaded: %d functions, %d caller-edges, %d reads-edges, %d embeddings",
            len(self.fn_meta),
            sum(len(v) for v in self.callers.values()),
            sum(len(v) for v in self.reads.values()),
            len(self.embeddings),
        )

    # ------------------------------------------------------------------ #
    # Traversals  (snapshot under lock, traverse lock-free)
    # ------------------------------------------------------------------ #

    def blast_radius(
        self, fqn: str, max_depth: int = BLAST_RADIUS_MAX_DEPTH
    ) -> list[dict]:
        """BFS upstream: all functions that transitively call *fqn*."""
        with self._lock:
            callers_snap = dict(self.callers)
        return self._bfs(fqn, callers_snap, max_depth)

    def impact_radius(
        self, fqn: str, max_depth: int = IMPACT_RADIUS_MAX_DEPTH
    ) -> list[dict]:
        """BFS downstream: all functions transitively called by *fqn*."""
        with self._lock:
            callees_snap = dict(self.callees)
        return self._bfs(fqn, callees_snap, max_depth)

    def _bfs(self, start: str, adj: dict, max_depth: int) -> list[dict]:
        visited: dict[str, int] = {}
        predecessor: dict[str, str] = {}  # node → who discovered it
        queue: list[tuple[str, int]] = [(start, 0)]
        while queue:
            current, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            for neighbour in adj.get(current, set()):
                if neighbour not in visited:
                    visited[neighbour] = depth + 1
                    predecessor[neighbour] = current
                    queue.append((neighbour, depth + 1))
        return self._format_hits(visited, predecessor, start)

    def _format_hits(
        self, visited: dict[str, int], predecessor: dict[str, str] | None = None, start: str = ""
    ) -> list[dict]:
        if predecessor is None:
            predecessor = {}
        with self._lock:
            meta_snap = self.fn_meta
        result = []
        for hit_fqn, dist in sorted(visited.items(), key=lambda x: x[1]):
            meta = meta_snap.get(hit_fqn, {})

            # Reconstruct call chain from hit back to seed
            path_parts: list[str] = []
            node = hit_fqn
            while node and node != start and node in predecessor:
                path_parts.append(node.split(".")[-1])
                node = predecessor[node]
            if start:
                path_parts.append(start.split(".")[-1])
            path_str = " → ".join(reversed(path_parts)) if path_parts else ""

            # KGCompass-inspired exponential decay: β=0.6 per hop.
            # 1-hop = 0.60, 2-hop = 0.36, 3-hop = 0.22, 4-hop = 0.13.
            # Keeps distant-but-semantically-similar functions from dominating.
            relevance = round(0.6 ** dist, 4)

            result.append({
                "fqn": hit_fqn,
                "file_path": meta.get("file_path", ""),
                "distance": dist,
                "relevance": relevance,
                "path": path_str,
            })
        return result

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    def summary(self) -> dict:
        with self._lock:
            return {
                "functions": len(self.fn_meta),
                "modules": len(self.module_fns),
                "call_edges": sum(len(v) for v in self.callees.values()),
                "reads_edges": sum(len(v) for v in self.reads.values()),
                "inherits_edges": sum(len(v) for v in self.bases.values()),
                "embeddings_cached": len(self.embeddings),
                "fingerprints_cached": len(self.fingerprints),
            }
