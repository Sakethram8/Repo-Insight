#!/usr/bin/env python3
"""
Check for breaking changes based on blast radius analysis
"""

import argparse
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Check for breaking changes')
    parser.add_argument('--analysis', type=str, required=True)
    parser.add_argument('--threshold', type=int, default=10,
                       help='Blast radius threshold for critical changes')
    return parser.parse_args()


def check_breaking_changes(analysis: dict, threshold: int) -> dict:
    """
    Analyze if changes are breaking based on blast radius.
    
    Returns:
        dict with 'has_critical', 'affected_count', 'critical_functions'
    """
    critical_functions = []
    total_affected = 0
    
    for func in analysis.get('function_details', []):
        blast_radius = func.get('blast_radius_size', 0)
        total_affected += blast_radius
        
        if blast_radius >= threshold:
            critical_functions.append({
                'fqn': func['fqn'],
                'file_path': func['file_path'],
                'blast_radius': blast_radius,
                'affected_functions': func.get('affected_functions', [])
            })
    
    return {
        'has_critical': len(critical_functions) > 0,
        'affected_count': total_affected,
        'critical_count': len(critical_functions),
        'critical_functions': critical_functions
    }


def main():
    args = parse_args()
    
    # Load analysis
    with open(args.analysis, 'r') as f:
        analysis = json.load(f)
    
    # Check for breaking changes
    result = check_breaking_changes(analysis, args.threshold)
    
    # Output for GitHub Actions
    print(f"::set-output name=has_critical::{str(result['has_critical']).lower()}")
    print(f"::set-output name=affected_count::{result['affected_count']}")
    print(f"::set-output name=critical_count::{result['critical_count']}")
    
    # Print summary
    if result['has_critical']:
        print(f"\n⚠️  CRITICAL: {result['critical_count']} function(s) with blast radius >= {args.threshold}")
        print(f"   Total affected functions: {result['affected_count']}")
        print("\nCritical functions:")
        for func in result['critical_functions']:
            print(f"   • {func['fqn']} (blast radius: {func['blast_radius']})")
    else:
        print(f"\n✅ No critical changes detected (threshold: {args.threshold})")
        print(f"   Total affected functions: {result['affected_count']}")
    
    # Don't fail the build, just warn
    sys.exit(0)


if __name__ == '__main__':
    main()

# Made with Bob
