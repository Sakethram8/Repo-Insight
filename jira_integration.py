#!/usr/bin/env python3
"""
Jira Integration for Repo-Insight - Feature 6
Connects code analysis with enterprise issue tracking

Provides:
- Automatic issue linking from PR analysis
- Status updates based on blast radius
- Cost tracking per Jira issue
- Sprint velocity insights
- Smart field updates (priority, labels, etc.)
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Try to import Jira SDK
try:
    from jira import JIRA
    JIRA_AVAILABLE = True
except ImportError:
    JIRA_AVAILABLE = False
    logger.warning("jira package not installed. Jira features disabled.")
    logger.warning("Install with: pip install jira")


class JiraIntegration:
    """
    Jira integration for Repo-Insight.
    
    Usage:
        jira = JiraIntegration()
        
        # Link PR analysis to Jira issue
        jira.update_issue_from_pr(
            issue_key="PROJ-123",
            pr_number=456,
            analysis={...}
        )
        
        # Track costs per issue
        jira.add_cost_comment(
            issue_key="PROJ-123",
            cost=0.0045,
            operation="pr_analysis"
        )
    """
    
    def __init__(
        self,
        server: Optional[str] = None,
        email: Optional[str] = None,
        api_token: Optional[str] = None
    ):
        """
        Initialize Jira connection.
        
        Args:
            server: Jira server URL (e.g., https://yourcompany.atlassian.net)
            email: Jira user email
            api_token: Jira API token
        
        If not provided, reads from environment:
            JIRA_SERVER, JIRA_EMAIL, JIRA_API_TOKEN
        """
        if not JIRA_AVAILABLE:
            self.client = None
            return
        
        self.server = server or os.getenv("JIRA_SERVER")
        self.email = email or os.getenv("JIRA_EMAIL")
        self.api_token = api_token or os.getenv("JIRA_API_TOKEN")
        
        if not all([self.server, self.email, self.api_token]):
            logger.warning("Jira credentials not configured. Integration disabled.")
            self.client = None
            return
        
        try:
            self.client = JIRA(
                server=self.server,
                basic_auth=(self.email, self.api_token)
            )
            logger.info(f"Connected to Jira: {self.server}")
        except Exception as e:
            logger.error(f"Failed to connect to Jira: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if Jira integration is available."""
        return self.client is not None
    
    def get_issue(self, issue_key: str):
        """Get Jira issue by key."""
        if not self.is_available():
            return None
        
        try:
            return self.client.issue(issue_key)
        except Exception as e:
            logger.error(f"Failed to get issue {issue_key}: {e}")
            return None
    
    def update_issue_from_pr(
        self,
        issue_key: str,
        pr_number: int,
        analysis: Dict,
        repo_url: Optional[str] = None
    ) -> bool:
        """
        Update Jira issue with PR analysis results.
        
        Args:
            issue_key: Jira issue key (e.g., "PROJ-123")
            pr_number: GitHub PR number
            analysis: PR analysis dict from analyze_pr.py
            repo_url: GitHub repository URL
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.warning("Jira not available, skipping update")
            return False
        
        try:
            issue = self.get_issue(issue_key)
            if not issue:
                return False
            
            # Build comment with analysis summary
            comment = self._format_pr_analysis_comment(
                pr_number, analysis, repo_url
            )
            
            # Add comment to issue
            self.client.add_comment(issue, comment)
            logger.info(f"Added PR analysis comment to {issue_key}")
            
            # Update labels based on impact
            self._update_labels_from_analysis(issue, analysis)
            
            # Update priority if high-impact
            if len(analysis.get('high_impact_changes', [])) > 0:
                self._update_priority_if_needed(issue, "High")
            
            # Add custom field for blast radius (if configured)
            self._update_custom_fields(issue, analysis)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update issue {issue_key}: {e}")
            return False
    
    def _format_pr_analysis_comment(
        self,
        pr_number: int,
        analysis: Dict,
        repo_url: Optional[str]
    ) -> str:
        """Format PR analysis as Jira comment."""
        
        lines = []
        lines.append(f"h3. 🔍 Repo-Insight PR Analysis")
        lines.append("")
        
        if repo_url:
            pr_url = f"{repo_url}/pull/{pr_number}"
            lines.append(f"*PR:* [#{pr_number}|{pr_url}]")
        else:
            lines.append(f"*PR:* #{pr_number}")
        
        lines.append(f"*Analyzed:* {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append("")
        
        # Impact summary
        lines.append("h4. Impact Summary")
        lines.append("")
        lines.append(f"* Changed Files: {analysis['total_changed_files']}")
        lines.append(f"* Modified Functions: {len(analysis['function_details'])}")
        lines.append(f"* Affected Functions: {analysis['total_affected_functions']}")
        lines.append(f"* High-Impact Changes: {len(analysis['high_impact_changes'])}")
        
        if analysis.get('cost_summary'):
            cost = analysis['cost_summary']
            lines.append(f"* Analysis Cost: ${cost['total_cost']:.4f}")
        
        lines.append("")
        
        # High-impact changes
        if analysis['high_impact_changes']:
            lines.append("h4. ⚠️ High-Impact Changes")
            lines.append("")
            for change in analysis['high_impact_changes'][:3]:
                lines.append(f"* {{{{monospace}}}}{change['fqn']}{{{{monospace}}}}")
                lines.append(f"** File: {change['file_path']}")
                lines.append(f"** Blast Radius: {change['blast_radius_size']} functions")
            lines.append("")
        
        # Recommendations
        lines.append("h4. Recommendations")
        lines.append("")
        if len(analysis['high_impact_changes']) > 0:
            lines.append("* (!) High-impact changes detected")
            lines.append("* Add comprehensive tests")
            lines.append("* Review with senior team members")
            lines.append("* Deploy to staging first")
        else:
            lines.append("* (/) Standard review process recommended")
        
        return "\n".join(lines)
    
    def _update_labels_from_analysis(self, issue, analysis: Dict):
        """Add labels based on analysis results."""
        try:
            labels = set(issue.fields.labels or [])
            
            # Add impact labels
            if len(analysis['high_impact_changes']) > 0:
                labels.add("high-impact")
                labels.add("needs-review")
            elif analysis['total_affected_functions'] > 0:
                labels.add("moderate-impact")
            else:
                labels.add("low-impact")
            
            # Add cross-repo label if applicable
            if analysis.get('cross_repo_impacts'):
                labels.add("cross-repo")
            
            # Update issue
            issue.update(fields={"labels": list(labels)})
            logger.info(f"Updated labels for {issue.key}")
            
        except Exception as e:
            logger.warning(f"Failed to update labels: {e}")
    
    def _update_priority_if_needed(self, issue, priority: str):
        """Update issue priority if not already high."""
        try:
            current_priority = issue.fields.priority.name if issue.fields.priority else "Medium"
            
            # Only escalate, never de-escalate
            priority_order = ["Lowest", "Low", "Medium", "High", "Highest"]
            if priority_order.index(priority) > priority_order.index(current_priority):
                issue.update(fields={"priority": {"name": priority}})
                logger.info(f"Updated priority for {issue.key} to {priority}")
        
        except Exception as e:
            logger.warning(f"Failed to update priority: {e}")
    
    def _update_custom_fields(self, issue, analysis: Dict):
        """Update custom fields with analysis data."""
        try:
            # This requires custom fields to be configured in Jira
            # Example: "customfield_10100" for blast radius
            
            # Get custom field IDs from environment
            blast_radius_field = os.getenv("JIRA_BLAST_RADIUS_FIELD")
            
            if blast_radius_field:
                total_blast = analysis['total_affected_functions']
                issue.update(fields={blast_radius_field: total_blast})
                logger.info(f"Updated blast radius field for {issue.key}")
        
        except Exception as e:
            logger.warning(f"Failed to update custom fields: {e}")
    
    def add_cost_comment(
        self,
        issue_key: str,
        cost: float,
        operation: str,
        details: Optional[Dict] = None
    ) -> bool:
        """
        Add cost tracking comment to Jira issue.
        
        Args:
            issue_key: Jira issue key
            cost: Cost in USD
            operation: Operation type (e.g., "pr_analysis", "behavior_labels")
            details: Additional cost details
        
        Returns:
            True if successful
        """
        if not self.is_available():
            return False
        
        try:
            issue = self.get_issue(issue_key)
            if not issue:
                return False
            
            comment = f"💰 *Repo-Insight Cost Tracking*\n\n"
            comment += f"* Operation: {operation}\n"
            comment += f"* Cost: ${cost:.4f}\n"
            comment += f"* Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
            
            if details:
                comment += f"\n*Details:*\n"
                for key, value in details.items():
                    comment += f"* {key}: {value}\n"
            
            self.client.add_comment(issue, comment)
            logger.info(f"Added cost comment to {issue_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add cost comment: {e}")
            return False
    
    def get_sprint_velocity(self, board_id: int, sprint_id: Optional[int] = None) -> Dict:
        """
        Calculate sprint velocity with Repo-Insight metrics.
        
        Args:
            board_id: Jira board ID
            sprint_id: Sprint ID (None for active sprint)
        
        Returns:
            Dict with velocity metrics
        """
        if not self.is_available():
            return {}
        
        try:
            # Get sprint
            if sprint_id:
                sprint = self.client.sprint(sprint_id)
            else:
                sprints = self.client.sprints(board_id, state='active')
                if not sprints:
                    return {}
                sprint = sprints[0]
            
            # Get issues in sprint
            issues = self.client.search_issues(
                f'sprint = {sprint.id}',
                maxResults=1000
            )
            
            # Calculate metrics
            total_issues = len(issues)
            completed_issues = len([i for i in issues if i.fields.status.name == 'Done'])
            
            # Calculate blast radius metrics from labels
            high_impact = len([i for i in issues if 'high-impact' in (i.fields.labels or [])])
            low_impact = len([i for i in issues if 'low-impact' in (i.fields.labels or [])])
            
            return {
                'sprint_name': sprint.name,
                'total_issues': total_issues,
                'completed_issues': completed_issues,
                'completion_rate': completed_issues / total_issues if total_issues > 0 else 0,
                'high_impact_issues': high_impact,
                'low_impact_issues': low_impact,
                'avg_impact': 'high' if high_impact > low_impact else 'low'
            }
            
        except Exception as e:
            logger.error(f"Failed to get sprint velocity: {e}")
            return {}
    
    def link_pr_to_issue(
        self,
        issue_key: str,
        pr_url: str,
        pr_title: str
    ) -> bool:
        """
        Create web link from Jira issue to GitHub PR.
        
        Args:
            issue_key: Jira issue key
            pr_url: GitHub PR URL
            pr_title: PR title
        
        Returns:
            True if successful
        """
        if not self.is_available():
            return False
        
        try:
            issue = self.get_issue(issue_key)
            if not issue:
                return False
            
            # Add remote link
            self.client.add_simple_link(
                issue,
                {
                    "url": pr_url,
                    "title": f"PR: {pr_title}"
                }
            )
            
            logger.info(f"Linked PR to {issue_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to link PR: {e}")
            return False


# Convenience functions
_global_jira: Optional[JiraIntegration] = None

def get_jira() -> JiraIntegration:
    """Get or create global Jira integration instance."""
    global _global_jira
    if _global_jira is None:
        _global_jira = JiraIntegration()
    return _global_jira


def update_jira_from_pr(issue_key: str, pr_number: int, analysis: Dict) -> bool:
    """
    Convenience function to update Jira from PR analysis.
    
    Usage:
        from jira_integration import update_jira_from_pr
        
        update_jira_from_pr("PROJ-123", 456, analysis_dict)
    """
    jira = get_jira()
    return jira.update_issue_from_pr(issue_key, pr_number, analysis)


if __name__ == "__main__":
    # Demo usage
    print("Jira Integration Demo")
    print("=" * 60)
    
    jira = JiraIntegration()
    
    if jira.is_available():
        print("✅ Connected to Jira")
        
        # Example: Update issue from PR
        analysis = {
            'total_changed_files': 5,
            'total_affected_functions': 47,
            'high_impact_changes': [
                {
                    'fqn': 'api.core.authenticate',
                    'file_path': 'api/core.py',
                    'blast_radius_size': 23
                }
            ],
            'function_details': [],
            'cost_summary': {
                'total_cost': 0.0045,
                'total_calls': 12
            }
        }
        
        print("\nExample PR analysis update:")
        print(f"  Issue: PROJ-123")
        print(f"  PR: #456")
        print(f"  High-impact changes: {len(analysis['high_impact_changes'])}")
        
        # Uncomment to actually update:
        # jira.update_issue_from_pr("PROJ-123", 456, analysis)
        
    else:
        print("❌ Jira not configured")
        print("\nTo enable Jira integration, set:")
        print("  export JIRA_SERVER=https://yourcompany.atlassian.net")
        print("  export JIRA_EMAIL=your.email@company.com")
        print("  export JIRA_API_TOKEN=your_api_token")

# Made with Bob
