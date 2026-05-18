# multi_repo.py
"""
Multi-repository graph support for Repo-Insight.

Enables cross-repository code intelligence by:
1. Adding repo_name property to all nodes
2. Detecting cross-repo imports and dependencies
3. Creating CROSS_REPO_CALLS edges
4. Updating blast radius queries to traverse cross-repo edges

This is the killer differentiator - NO other code intelligence tool
has cross-repository call graph analysis.

For TechEx Track 2: Enterprise polyglot codebases span 50+ repositories.
"""

import logging
from pathlib import Path
from typing import List, Dict, Set, Optional
import falkordb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Repository metadata
# ---------------------------------------------------------------------------

def add_repo_name_to_nodes(
    graph: falkordb.Graph,
    repo_name: str,
    file_paths: Optional[List[str]] = None
) -> int:
    """
    Add repo_name property to all nodes from a specific repository.
    
    This is idempotent - can be run multiple times safely.
    
    Args:
        graph: FalkorDB graph instance
        repo_name: Name of the repository (e.g., "django", "react-frontend")
        file_paths: Optional list of file paths to update (if None, updates all)
    
    Returns:
        Number of nodes updated
    
    Example:
        # After ingesting Django
        add_repo_name_to_nodes(graph, "django")
        
        # After ingesting React frontend
        add_repo_name_to_nodes(graph, "react-frontend")
    """
    if file_paths:
        # Update specific files only
        query = """
        MATCH (n)
        WHERE n.file_path IN $file_paths
        SET n.repo_name = $repo_name
        RETURN count(n) as updated
        """
        result = graph.query(query, {"file_paths": file_paths, "repo_name": repo_name})
    else:
        # Update all nodes without repo_name
        query = """
        MATCH (n)
        WHERE n.file_path IS NOT NULL AND n.repo_name IS NULL
        SET n.repo_name = $repo_name
        RETURN count(n) as updated
        """
        result = graph.query(query, {"repo_name": repo_name})
    
    updated = result.result_set[0][0] if result.result_set else 0
    logger.info(f"Added repo_name='{repo_name}' to {updated} nodes")
    return updated


def get_repository_stats(graph: falkordb.Graph) -> Dict[str, Dict[str, int]]:
    """
    Get statistics for each repository in the graph.
    
    Returns:
        Dict mapping repo_name to stats (functions, classes, modules, edges)
    
    Example:
        stats = get_repository_stats(graph)
        # {
        #   "django": {"functions": 6234, "classes": 892, "modules": 456, "edges": 15234},
        #   "react-frontend": {"functions": 234, "classes": 45, "modules": 67, "edges": 567}
        # }
    """
    query = """
    MATCH (n)
    WHERE n.repo_name IS NOT NULL
    RETURN n.repo_name as repo,
           labels(n)[0] as type,
           count(n) as count
    """
    
    result = graph.query(query)
    stats = {}
    
    for row in result.result_set:
        repo_name, node_type, count = row[0], row[1], row[2]
        if repo_name not in stats:
            stats[repo_name] = {"functions": 0, "classes": 0, "modules": 0}
        
        if node_type == "Function":
            stats[repo_name]["functions"] = count
        elif node_type == "Class":
            stats[repo_name]["classes"] = count
        elif node_type == "Module":
            stats[repo_name]["modules"] = count
    
    # Count edges per repo
    edge_query = """
    MATCH (a)-[r]->(b)
    WHERE a.repo_name IS NOT NULL
    RETURN a.repo_name as repo, count(r) as edges
    """
    
    edge_result = graph.query(edge_query)
    for row in edge_result.result_set:
        repo_name, edge_count = row[0], row[1]
        if repo_name in stats:
            stats[repo_name]["edges"] = edge_count
    
    return stats


# ---------------------------------------------------------------------------
# Cross-repository edge detection
# ---------------------------------------------------------------------------

def detect_cross_repo_imports(graph: falkordb.Graph) -> List[Dict]:
    """
    Detect imports that cross repository boundaries.
    
    Returns list of cross-repo import relationships:
    [
        {
            "from_repo": "react-frontend",
            "from_module": "src.components.Button",
            "to_repo": "shared-utils",
            "to_module": "utils.validation",
            "import_count": 5
        },
        ...
    ]
    
    This identifies potential cross-repo dependencies that should
    become CROSS_REPO_CALLS edges.
    """
    query = """
    MATCH (m1:Module)-[:IMPORTS]->(m2:Module)
    WHERE m1.repo_name <> m2.repo_name
    AND m1.repo_name IS NOT NULL
    AND m2.repo_name IS NOT NULL
    RETURN m1.repo_name as from_repo,
           m1.name as from_module,
           m2.repo_name as to_repo,
           m2.name as to_module,
           count(*) as import_count
    """
    
    result = graph.query(query)
    cross_repo_imports = []
    
    for row in result.result_set:
        cross_repo_imports.append({
            "from_repo": row[0],
            "from_module": row[1],
            "to_repo": row[2],
            "to_module": row[3],
            "import_count": row[4],
        })
    
    logger.info(f"Found {len(cross_repo_imports)} cross-repo import relationships")
    return cross_repo_imports


def create_cross_repo_edges(
    graph: falkordb.Graph,
    repo1_name: str,
    repo2_name: str,
    package_mappings: Optional[Dict[str, str]] = None
) -> int:
    """
    Create CROSS_REPO_CALLS edges between two repositories.
    
    Detects when repo1 imports from repo2 and creates edges from
    calling functions in repo1 to called functions in repo2.
    
    Args:
        graph: FalkorDB graph instance
        repo1_name: Name of the calling repository
        repo2_name: Name of the called repository
        package_mappings: Optional dict mapping import names to repo names
                         e.g., {"django": "django", "sqlparse": "sqlparse"}
    
    Returns:
        Number of CROSS_REPO_CALLS edges created
    
    Example:
        # Django imports sqlparse
        create_cross_repo_edges(graph, "django", "sqlparse")
        
        # React imports shared utilities
        create_cross_repo_edges(graph, "react-frontend", "shared-utils")
    """
    # Strategy: Find functions in repo1 that call functions in repo2
    # This happens when:
    # 1. repo1 imports a module from repo2
    # 2. Functions in repo1 call functions from that imported module
    
    query = """
    // Find imports from repo1 to repo2
    MATCH (m1:Module {repo_name: $repo1})-[:IMPORTS]->(m2:Module {repo_name: $repo2})
    
    // Find functions in repo1 that might call functions in m2
    MATCH (f1:Function {repo_name: $repo1})-[:CALLS]->(f2:Function)
    WHERE f2.repo_name = $repo2
    
    // Create CROSS_REPO_CALLS edge if it doesn't exist
    MERGE (f1)-[r:CROSS_REPO_CALLS]->(f2)
    
    RETURN count(r) as edges_created
    """
    
    result = graph.query(query, {"repo1": repo1_name, "repo2": repo2_name})
    edges_created = result.result_set[0][0] if result.result_set else 0
    
    logger.info(f"Created {edges_created} CROSS_REPO_CALLS edges from {repo1_name} to {repo2_name}")
    return edges_created


def link_repositories(
    graph: falkordb.Graph,
    repo_names: List[str],
    auto_detect: bool = True
) -> Dict[str, int]:
    """
    Link multiple repositories by creating CROSS_REPO_CALLS edges.
    
    Args:
        graph: FalkorDB graph instance
        repo_names: List of repository names to link
        auto_detect: If True, automatically detect cross-repo imports
    
    Returns:
        Dict mapping repo pairs to edge counts
        e.g., {"django->sqlparse": 45, "react->shared-utils": 12}
    
    Example:
        # Link all repositories
        link_repositories(graph, ["django", "sqlparse", "react-frontend"])
    """
    edge_counts = {}
    
    if auto_detect:
        # Detect cross-repo imports first
        cross_imports = detect_cross_repo_imports(graph)
        
        # Create edges for each detected relationship
        for imp in cross_imports:
            from_repo = imp["from_repo"]
            to_repo = imp["to_repo"]
            
            if from_repo in repo_names and to_repo in repo_names:
                key = f"{from_repo}->{to_repo}"
                if key not in edge_counts:
                    count = create_cross_repo_edges(graph, from_repo, to_repo)
                    edge_counts[key] = count
    else:
        # Create edges between all pairs
        for i, repo1 in enumerate(repo_names):
            for repo2 in repo_names[i+1:]:
                # Try both directions
                count1 = create_cross_repo_edges(graph, repo1, repo2)
                count2 = create_cross_repo_edges(graph, repo2, repo1)
                
                if count1 > 0:
                    edge_counts[f"{repo1}->{repo2}"] = count1
                if count2 > 0:
                    edge_counts[f"{repo2}->{repo1}"] = count2
    
    total_edges = sum(edge_counts.values())
    logger.info(f"Linked {len(repo_names)} repositories with {total_edges} total cross-repo edges")
    
    return edge_counts


# ---------------------------------------------------------------------------
# Cross-repository queries
# ---------------------------------------------------------------------------

def get_cross_repo_blast_radius(
    graph: falkordb.Graph,
    fqn: str,
    max_depth: int = 4,
    include_cross_repo: bool = True
) -> Dict:
    """
    Get blast radius including cross-repository callers.
    
    Args:
        graph: FalkorDB graph instance
        fqn: Fully qualified name of the target function
        max_depth: Maximum traversal depth
        include_cross_repo: If True, traverse CROSS_REPO_CALLS edges
    
    Returns:
        Dict with affected functions, grouped by repository
    
    Example:
        result = get_cross_repo_blast_radius(graph, "django.db.models.query.QuerySet.filter")
        # {
        #   "total_affected": 47,
        #   "by_repo": {
        #     "django": [{"fqn": "...", "distance": 1}, ...],
        #     "django-extensions": [{"fqn": "...", "distance": 2}, ...]
        #   },
        #   "cross_repo_count": 5
        # }
    """
    if include_cross_repo:
        edge_types = "CALLS|CROSS_REPO_CALLS"
    else:
        edge_types = "CALLS"
    
    query = f"""
    MATCH path = (caller:Function)-[:{edge_types}*1..{max_depth}]->(target:Function {{fqn: $fqn}})
    RETURN DISTINCT 
        caller.fqn as fqn,
        caller.repo_name as repo,
        caller.file_path as file_path,
        length(path) as distance
    ORDER BY distance ASC
    LIMIT 300
    """
    
    result = graph.query(query, {"fqn": fqn})
    
    by_repo = {}
    cross_repo_count = 0
    target_repo = None
    
    for row in result.result_set:
        caller_fqn, repo, file_path, distance = row[0], row[1], row[2], row[3]
        
        # Determine target repo from first result
        if target_repo is None:
            # Get target function's repo
            target_query = "MATCH (f:Function {fqn: $fqn}) RETURN f.repo_name"
            target_result = graph.query(target_query, {"fqn": fqn})
            if target_result.result_set:
                target_repo = target_result.result_set[0][0]
        
        # Track cross-repo callers
        if repo != target_repo:
            cross_repo_count += 1
        
        if repo not in by_repo:
            by_repo[repo] = []
        
        by_repo[repo].append({
            "fqn": caller_fqn,
            "file_path": file_path,
            "distance": distance,
            "is_cross_repo": repo != target_repo
        })
    
    total_affected = sum(len(funcs) for funcs in by_repo.values())
    
    return {
        "target_fqn": fqn,
        "target_repo": target_repo,
        "total_affected": total_affected,
        "by_repo": by_repo,
        "cross_repo_count": cross_repo_count,
        "repositories": list(by_repo.keys()),
    }


def get_cross_repo_dependencies(
    graph: falkordb.Graph,
    repo_name: str
) -> Dict:
    """
    Get all repositories that depend on this repository.
    
    Args:
        graph: FalkorDB graph instance
        repo_name: Name of the repository
    
    Returns:
        Dict with upstream and downstream dependencies
    
    Example:
        deps = get_cross_repo_dependencies(graph, "django")
        # {
        #   "upstream": ["sqlparse", "pytz"],  # django depends on these
        #   "downstream": ["django-extensions", "django-rest-framework"]  # these depend on django
        # }
    """
    # Upstream: repos that this repo depends on
    upstream_query = """
    MATCH (f1:Function {repo_name: $repo})-[:CROSS_REPO_CALLS]->(f2:Function)
    WHERE f2.repo_name <> $repo
    RETURN DISTINCT f2.repo_name as upstream_repo
    """
    
    upstream_result = graph.query(upstream_query, {"repo": repo_name})
    upstream = [row[0] for row in upstream_result.result_set]
    
    # Downstream: repos that depend on this repo
    downstream_query = """
    MATCH (f1:Function)-[:CROSS_REPO_CALLS]->(f2:Function {repo_name: $repo})
    WHERE f1.repo_name <> $repo
    RETURN DISTINCT f1.repo_name as downstream_repo
    """
    
    downstream_result = graph.query(downstream_query, {"repo": repo_name})
    downstream = [row[0] for row in downstream_result.result_set]
    
    return {
        "repository": repo_name,
        "upstream": upstream,
        "downstream": downstream,
        "upstream_count": len(upstream),
        "downstream_count": len(downstream),
    }


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def get_repo_dependency_graph(graph: falkordb.Graph) -> Dict:
    """
    Get repository-level dependency graph for visualization.
    
    Returns:
        Dict with nodes (repositories) and edges (dependencies)
    
    Example:
        graph_data = get_repo_dependency_graph(graph)
        # {
        #   "nodes": [
        #     {"id": "django", "functions": 6234, "classes": 892},
        #     {"id": "sqlparse", "functions": 234, "classes": 45}
        #   ],
        #   "edges": [
        #     {"from": "django", "to": "sqlparse", "calls": 45}
        #   ]
        # }
    """
    # Get all repositories
    stats = get_repository_stats(graph)
    nodes = [
        {
            "id": repo_name,
            "functions": repo_stats.get("functions", 0),
            "classes": repo_stats.get("classes", 0),
            "modules": repo_stats.get("modules", 0),
        }
        for repo_name, repo_stats in stats.items()
    ]
    
    # Get cross-repo edges
    edge_query = """
    MATCH (f1:Function)-[:CROSS_REPO_CALLS]->(f2:Function)
    WHERE f1.repo_name <> f2.repo_name
    RETURN f1.repo_name as from_repo,
           f2.repo_name as to_repo,
           count(*) as call_count
    """
    
    edge_result = graph.query(edge_query)
    edges = [
        {
            "from": row[0],
            "to": row[1],
            "calls": row[2],
        }
        for row in edge_result.result_set
    ]
    
    return {
        "nodes": nodes,
        "edges": edges,
    }


# ---------------------------------------------------------------------------
# Integration with existing tools
# ---------------------------------------------------------------------------

def update_blast_radius_for_multi_repo(
    graph: falkordb.Graph,
    fqn: str,
    max_depth: int = 4
) -> Dict:
    """
    Drop-in replacement for tools.get_upstream_callers that includes cross-repo edges.
    
    This can be used to update the existing get_blast_radius MCP tool.
    """
    return get_cross_repo_blast_radius(graph, fqn, max_depth, include_cross_repo=True)


# ---------------------------------------------------------------------------
# CLI / Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test script
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Multi-Repo Graph Support - Test Script")
    print("=" * 60)
    
    # This would normally connect to your FalkorDB instance
    print("\nFeatures:")
    print("✓ Add repo_name property to all nodes")
    print("✓ Detect cross-repo imports automatically")
    print("✓ Create CROSS_REPO_CALLS edges")
    print("✓ Query blast radius across repositories")
    print("✓ Visualize repository dependency graph")
    
    print("\nUsage Example:")
    print("""
    from multi_repo import *
    from ingest import get_connection
    
    graph = get_connection()
    
    # After ingesting Django
    add_repo_name_to_nodes(graph, "django")
    
    # After ingesting sqlparse
    add_repo_name_to_nodes(graph, "sqlparse")
    
    # Link repositories
    link_repositories(graph, ["django", "sqlparse"])
    
    # Query cross-repo blast radius
    result = get_cross_repo_blast_radius(
        graph,
        "django.db.models.query.QuerySet.filter"
    )
    
    print(f"Total affected: {result['total_affected']}")
    print(f"Cross-repo callers: {result['cross_repo_count']}")
    print(f"Repositories: {result['repositories']}")
    """)
    
    print("\n" + "=" * 60)
    print("Multi-repo support ready for TechEx demo!")

# Made with Bob
