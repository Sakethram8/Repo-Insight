#!/usr/bin/env python3
"""
Generate formatted PR comment from analysis results
"""

import argparse
import json
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Generate PR comment')
    parser.add_argument('--analysis', type=str, required=True)
    parser.add_argument('--output', type=str, default='pr_comment.md')
    return parser.parse_args()


def format_blast_radius_badge(size: int) -> str:
    """Generate colored badge for blast radius size."""
    if size == 0:
        return "🟢 None"
    elif size < 5:
        return f"🟢 Low ({size})"
    elif size < 15:
        return f"🟡 Medium ({size})"
    else:
        return f"🔴 High ({size})"


def generate_comment(analysis: dict) -> str:
    """Generate markdown comment from analysis."""
    
    lines = []
    
    # Header
    lines.append("## 🔍 Repo-Insight Analysis")
    lines.append("")
    lines.append(f"**PR #{analysis['pr_number']}** • Analysis completed at {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Summary metrics
    lines.append("### 📊 Impact Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Changed Files | {analysis['total_changed_files']} |")
    lines.append(f"| Modified Functions | {len(analysis['function_details'])} |")
    lines.append(f"| Affected Functions (Blast Radius) | {analysis['total_affected_functions']} |")
    lines.append(f"| High-Impact Changes | {len(analysis['high_impact_changes'])} |")
    
    if analysis.get('cost_summary'):
        cost = analysis['cost_summary']
        lines.append(f"| Analysis Cost | ${cost['total_cost']:.4f} |")
        lines.append(f"| API Calls | {cost['total_calls']} |")
        lines.append(f"| Tokens Used | {cost['total_tokens']:,} |")
    
    lines.append("")
    
    # High-impact changes warning
    if analysis['high_impact_changes']:
        lines.append("### ⚠️ High-Impact Changes Detected")
        lines.append("")
        lines.append(f"**{len(analysis['high_impact_changes'])} function(s)** have a blast radius > 10 functions.")
        lines.append("These changes may affect many parts of the codebase. Please review carefully.")
        lines.append("")
        
        for change in analysis['high_impact_changes'][:5]:  # Top 5
            lines.append(f"#### `{change['fqn']}`")
            lines.append("")
            lines.append(f"- **File:** `{change['file_path']}`")
            lines.append(f"- **Line:** {change['start_line']}")
            lines.append(f"- **Blast Radius:** {format_blast_radius_badge(change['blast_radius_size'])}")
            lines.append("")
            
            if change['affected_functions']:
                lines.append("<details>")
                lines.append("<summary>Show affected functions (top 20)</summary>")
                lines.append("")
                for func in change['affected_functions'][:20]:
                    lines.append(f"- `{func}`")
                lines.append("")
                lines.append("</details>")
                lines.append("")
    
    # All changed functions
    if analysis['function_details']:
        lines.append("### 📝 Changed Functions")
        lines.append("")
        lines.append("| Function | File | Blast Radius |")
        lines.append("|----------|------|--------------|")
        
        for func in analysis['function_details'][:20]:  # Top 20
            badge = format_blast_radius_badge(func['blast_radius_size'])
            lines.append(f"| `{func['fqn']}` | `{func['file_path']}` | {badge} |")
        
        if len(analysis['function_details']) > 20:
            lines.append(f"| ... and {len(analysis['function_details']) - 20} more | | |")
        
        lines.append("")
    
    # Cross-repo impacts (if available)
    if analysis.get('cross_repo_impacts'):
        lines.append("### 🔗 Cross-Repository Impacts")
        lines.append("")
        lines.append("Changes in this PR affect other repositories:")
        lines.append("")
        
        for impact in analysis['cross_repo_impacts'][:10]:
            lines.append(f"- **{impact['source_repo']}** → **{impact['target_repo']}**")
            lines.append(f"  - `{impact['source_function']}` calls `{impact['target_function']}`")
        
        lines.append("")
    
    # Recommendations
    lines.append("### 💡 Recommendations")
    lines.append("")
    
    if len(analysis['high_impact_changes']) > 0:
        lines.append("- ⚠️ **High-impact changes detected.** Consider:")
        lines.append("  - Adding comprehensive tests for affected functions")
        lines.append("  - Reviewing with senior team members")
        lines.append("  - Deploying to staging environment first")
        lines.append("  - Monitoring error rates after deployment")
    elif analysis['total_affected_functions'] > 0:
        lines.append("- ✅ **Moderate impact.** Standard review process recommended.")
        lines.append("  - Ensure tests cover the blast radius")
        lines.append("  - Check for potential side effects")
    else:
        lines.append("- ✅ **Low impact.** Changes are well-isolated.")
        lines.append("  - Standard testing should be sufficient")
    
    lines.append("")
    
    # Footer
    lines.append("---")
    lines.append("")
    lines.append("<sub>")
    lines.append("🤖 Generated by [Repo-Insight](https://github.com/yourusername/Repo-Insight) • ")
    lines.append("Powered by Google Gemini & FalkorDB • ")
    lines.append("[TechEx Hackathon 2024](https://lablab.ai/ai-hackathons/techex-intelligent-enterprise-solutions-hackathon)")
    lines.append("</sub>")
    
    return "\n".join(lines)


def main():
    args = parse_args()
    
    # Load analysis
    with open(args.analysis, 'r') as f:
        analysis = json.load(f)
    
    # Generate comment
    comment = generate_comment(analysis)
    
    # Write output
    with open(args.output, 'w') as f:
        f.write(comment)
    
    print(f"✅ PR comment generated: {args.output}")
    print(f"   • Changed files: {analysis['total_changed_files']}")
    print(f"   • Affected functions: {analysis['total_affected_functions']}")
    print(f"   • High-impact changes: {len(analysis['high_impact_changes'])}")


if __name__ == '__main__':
    main()

# Made with Bob
