#!/usr/bin/env python3
"""
GitHub Action script to analyze PR impact using Repo-Insight
Generates blast radius analysis for changed functions
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from falkordb import FalkorDB
from ingest import run_ingestion
from graph_index import GraphIndex
from multi_repo import get_cross_repo_blast_radius


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze PR impact')
    parser.add_argument('--pr-number', type=int, required=True)
    parser.add_argument('--changed-files', type=str, required=True)
    parser.add_argument('--base-ref', type=str, required=True)
    parser.add_argument('--head-ref', type=str, required=True)
    parser.add_argument('--output', type=str, default='pr_analysis.json')
    return parser.parse_args()


def get_changed_functions(changed_files: List[str], repo_path: str) -> List[Dict]:
    """
    Parse changed files and extract modified functions.
    Uses git diff to identify changed line ranges.
    """
    import subprocess
    
    changed_functions = []
    
    for file_path in changed_files:
        if not file_path.endswith(('.py', '.js', '.jsx', '.ts', '.tsx')):
            continue
        
        try:
            # Get diff for this file
            result = subprocess.run(
                ['git', 'diff', '--unified=0', 'HEAD^', 'HEAD', file_path],
                capture_output=True,
                text=True,
                cwd=repo_path
            )
            
            if result.returncode != 0:
                continue
            
            # Parse diff to get changed line numbers
            changed_lines = set()
            for line in result.stdout.split('\n'):
                if line.startswith('@@'):
                    # Extract line numbers from @@ -a,b +c,d @@
                    parts = line.split('@@')[1].strip().split()
                    if len(parts) >= 2:
                        new_range = parts[1].lstrip('+')
                        if ',' in new_range:
                            start, count = map(int, new_range.split(','))
                            changed_lines.update(range(start, start + count))
                        else:
                            changed_lines.add(int(new_range))
            
            changed_functions.append({
                'file_path': file_path,
                'changed_lines': sorted(changed_lines)
            })
            
        except Exception as e:
            print(f"Warning: Failed to analyze {file_path}: {e}")
            continue
    
    return changed_functions


def analyze_blast_radius(
    graph,
    index: GraphIndex,
    changed_functions: List[Dict]
) -> Dict:
    """
    Calculate blast radius for all changed functions.
    """
    analysis = {
        'total_changed_files': len(changed_functions),
        'total_affected_functions': 0,
        'high_impact_changes': [],
        'cross_repo_impacts': [],
        'function_details': []
    }
    
    for change in changed_functions:
        file_path = change['file_path']
        
        # Query functions in this file
        try:
            result = graph.query(
                "MATCH (f:Function) WHERE f.file_path = $fp RETURN f.fqn, f.start_line",
                {"fp": file_path}
            )
            
            for row in result.result_set:
                fqn, start_line = row[0], row[1]
                
                # Check if this function was modified
                if start_line in change['changed_lines']:
                    # Calculate blast radius
                    affected = set()
                    queue = [fqn]
                    visited = set()
                    
                    while queue and len(affected) < 100:  # Limit to prevent explosion
                        current = queue.pop(0)
                        if current in visited:
                            continue
                        visited.add(current)
                        
                        # Get callers
                        callers = index.callers.get(current, set())
                        for caller in callers:
                            if caller not in affected:
                                affected.add(caller)
                                queue.append(caller)
                    
                    function_analysis = {
                        'fqn': fqn,
                        'file_path': file_path,
                        'start_line': start_line,
                        'blast_radius_size': len(affected),
                        'affected_functions': list(affected)[:20],  # Top 20
                        'is_high_impact': len(affected) > 10
                    }
                    
                    analysis['function_details'].append(function_analysis)
                    analysis['total_affected_functions'] += len(affected)
                    
                    if len(affected) > 10:
                        analysis['high_impact_changes'].append(function_analysis)
        
        except Exception as e:
            print(f"Warning: Failed to analyze {file_path}: {e}")
            continue
    
    return analysis


def analyze_pr(args) -> Dict:
    """Main PR analysis logic."""
    
    print(f"🔍 Analyzing PR #{args.pr_number}")
    print(f"📝 Changed files: {args.changed_files}")
    
    # Parse changed files
    changed_files = args.changed_files.split()
    print(f"📊 Total changed files: {len(changed_files)}")
    
    # Connect to FalkorDB
    db = FalkorDB(
        host=os.getenv("FALKORDB_HOST", "localhost"),
        port=int(os.getenv("FALKORDB_PORT", 6379))
    )
    graph = db.select_graph("repo_insight_pr")
    
    # Ingest current codebase
    print("📥 Ingesting codebase...")
    repo_path = os.getcwd()
    stats = run_ingestion(repo_path)
    print(f"✅ Ingested {stats['functions']} functions, {stats['call_edges']} call edges")
    
    # Build index
    print("🔨 Building graph index...")
    index = GraphIndex.build(graph)
    print("✅ Index built")
    
    # Get changed functions
    print("🔎 Identifying changed functions...")
    changed_functions = get_changed_functions(changed_files, repo_path)
    print(f"✅ Found {len(changed_functions)} changed files")
    
    # Analyze blast radius
    print("💥 Calculating blast radius...")
    analysis = analyze_blast_radius(graph, index, changed_functions)
    print(f"✅ Analysis complete")
    print(f"   • Total affected functions: {analysis['total_affected_functions']}")
    print(f"   • High-impact changes: {len(analysis['high_impact_changes'])}")
    
    # Add metadata
    analysis['pr_number'] = args.pr_number
    analysis['base_ref'] = args.base_ref
    analysis['head_ref'] = args.head_ref
    analysis['changed_files'] = changed_files
    
    # Calculate cost if available
    try:
        from cost_dashboard import get_tracker
        tracker = get_tracker()
        summary = tracker.get_summary()
        analysis['cost_summary'] = {
            'total_cost': summary.total_cost,
            'total_calls': summary.total_calls,
            'total_tokens': summary.total_input_tokens + summary.total_output_tokens
        }
    except Exception:
        analysis['cost_summary'] = None
    
    return analysis


def main():
    args = parse_args()
    
    try:
        analysis = analyze_pr(args)
        
        # Write output
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\n✅ Analysis saved to {args.output}")
        
        # Exit with error if high-impact changes detected
        if len(analysis['high_impact_changes']) > 0:
            print(f"\n⚠️  Warning: {len(analysis['high_impact_changes'])} high-impact changes detected")
            sys.exit(0)  # Don't fail the build, just warn
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

# Made with Bob
