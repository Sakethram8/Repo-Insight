#!/usr/bin/env python3
"""
Cost Dashboard for TechEx Hackathon - Feature 4
Real-time tracking of Gemini API usage, token consumption, and costs

Provides:
- Live cost monitoring during code analysis
- Token usage breakdown by operation
- Cost optimization recommendations
- Historical usage trends
- Budget alerts and warnings
"""

import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# Gemini API Pricing (as of 2024)
# Source: https://ai.google.dev/pricing
GEMINI_PRICING = {
    "gemini-1.5-flash": {
        "input_per_1k": 0.00001875,   # $0.01875 per 1M tokens
        "output_per_1k": 0.000075,    # $0.075 per 1M tokens
        "name": "Gemini 1.5 Flash"
    },
    "gemini-1.5-pro": {
        "input_per_1k": 0.00125,      # $1.25 per 1M tokens
        "output_per_1k": 0.005,       # $5.00 per 1M tokens
        "name": "Gemini 1.5 Pro"
    },
    "gemini-pro": {
        "input_per_1k": 0.0005,       # $0.50 per 1M tokens
        "output_per_1k": 0.0015,      # $1.50 per 1M tokens
        "name": "Gemini Pro"
    }
}

@dataclass
class CostEntry:
    """Single API call cost entry"""
    timestamp: str
    operation: str          # "behavior_label", "rerank", "semantic_search"
    model: str             # "gemini-1.5-flash", "gemini-1.5-pro"
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    metadata: Optional[Dict] = None  # Additional context (file_path, function_name, etc.)

@dataclass
class CostSummary:
    """Aggregated cost statistics"""
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    cost_by_operation: Dict[str, float]
    cost_by_model: Dict[str, float]
    tokens_by_operation: Dict[str, int]
    average_cost_per_call: float
    most_expensive_operation: str
    recommendations: List[str]

class CostTracker:
    """
    Tracks Gemini API costs in real-time.
    
    Usage:
        tracker = CostTracker()
        
        # Track a behavior label generation
        tracker.track_call(
            operation="behavior_label",
            model="gemini-1.5-flash",
            input_tokens=150,
            output_tokens=20,
            metadata={"function": "api.connect"}
        )
        
        # Get current summary
        summary = tracker.get_summary()
        print(f"Total cost: ${summary.total_cost:.4f}")
        
        # Save for persistence
        tracker.save()
    """
    
    def __init__(self, storage_path: str = ".cost_tracking.json"):
        self.storage_path = Path(storage_path)
        self.entries: List[CostEntry] = []
        self.load()
    
    def track_call(
        self,
        operation: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[Dict] = None
    ) -> CostEntry:
        """Track a single API call and calculate costs"""
        
        if model not in GEMINI_PRICING:
            logger.warning(f"Unknown model: {model}, using gemini-1.5-flash pricing")
            model = "gemini-1.5-flash"
        
        pricing = GEMINI_PRICING[model]
        
        # Calculate costs
        input_cost = (input_tokens / 1000) * pricing["input_per_1k"]
        output_cost = (output_tokens / 1000) * pricing["output_per_1k"]
        total_cost = input_cost + output_cost
        
        entry = CostEntry(
            timestamp=datetime.utcnow().isoformat(),
            operation=operation,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            metadata=metadata or {}
        )
        
        self.entries.append(entry)
        
        # Auto-save every 10 entries
        if len(self.entries) % 10 == 0:
            self.save()
        
        return entry
    
    def get_summary(self, since: Optional[datetime] = None) -> CostSummary:
        """Generate cost summary statistics"""
        
        # Filter entries by time if specified
        entries = self.entries
        if since:
            entries = [
                e for e in entries
                if datetime.fromisoformat(e.timestamp) >= since
            ]
        
        if not entries:
            return CostSummary(
                total_calls=0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_cost=0.0,
                cost_by_operation={},
                cost_by_model={},
                tokens_by_operation={},
                average_cost_per_call=0.0,
                most_expensive_operation="",
                recommendations=[]
            )
        
        # Aggregate statistics
        total_calls = len(entries)
        total_input_tokens = sum(e.input_tokens for e in entries)
        total_output_tokens = sum(e.output_tokens for e in entries)
        total_cost = sum(e.total_cost for e in entries)
        
        # Cost by operation
        cost_by_operation = defaultdict(float)
        tokens_by_operation = defaultdict(int)
        for e in entries:
            cost_by_operation[e.operation] += e.total_cost
            tokens_by_operation[e.operation] += e.input_tokens + e.output_tokens
        
        # Cost by model
        cost_by_model = defaultdict(float)
        for e in entries:
            cost_by_model[e.model] += e.total_cost
        
        # Most expensive operation
        most_expensive = max(cost_by_operation.items(), key=lambda x: x[1])[0] if cost_by_operation else ""
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            entries, cost_by_operation, cost_by_model, total_cost
        )
        
        return CostSummary(
            total_calls=total_calls,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_cost=total_cost,
            cost_by_operation=dict(cost_by_operation),
            cost_by_model=dict(cost_by_model),
            tokens_by_operation=dict(tokens_by_operation),
            average_cost_per_call=total_cost / total_calls,
            most_expensive_operation=most_expensive,
            recommendations=recommendations
        )
    
    def _generate_recommendations(
        self,
        entries: List[CostEntry],
        cost_by_operation: Dict[str, float],
        cost_by_model: Dict[str, float],
        total_cost: float
    ) -> List[str]:
        """Generate cost optimization recommendations"""
        
        recommendations = []
        
        # Check if using expensive models unnecessarily
        pro_usage = cost_by_model.get("gemini-1.5-pro", 0)
        if pro_usage > total_cost * 0.5:
            recommendations.append(
                "⚠️ Over 50% of costs from Gemini Pro. Consider using Flash for simpler tasks."
            )
        
        # Check for high-frequency operations
        operation_counts = defaultdict(int)
        for e in entries:
            operation_counts[e.operation] += 1
        
        for op, count in operation_counts.items():
            if count > 100 and cost_by_operation[op] > total_cost * 0.3:
                recommendations.append(
                    f"💡 '{op}' called {count} times (${cost_by_operation[op]:.4f}). "
                    f"Consider caching results."
                )
        
        # Check for large token usage
        avg_tokens = sum(e.input_tokens + e.output_tokens for e in entries) / len(entries)
        if avg_tokens > 1000:
            recommendations.append(
                f"📊 Average {avg_tokens:.0f} tokens per call. "
                f"Consider using fingerprints to reduce context size."
            )
        
        # Budget warnings
        if total_cost > 10.0:
            recommendations.append(
                f"🚨 Total cost ${total_cost:.2f} exceeds $10. Review usage patterns."
            )
        elif total_cost > 5.0:
            recommendations.append(
                f"⚠️ Total cost ${total_cost:.2f} approaching $10 budget threshold."
            )
        
        # Positive feedback
        if not recommendations:
            recommendations.append(
                "✅ Cost usage is optimal. No recommendations at this time."
            )
        
        return recommendations
    
    def get_hourly_breakdown(self, hours: int = 24) -> Dict[str, float]:
        """Get cost breakdown by hour for the last N hours"""
        
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=hours)
        
        hourly_costs = defaultdict(float)
        
        for entry in self.entries:
            timestamp = datetime.fromisoformat(entry.timestamp)
            if timestamp >= cutoff:
                hour_key = timestamp.strftime("%Y-%m-%d %H:00")
                hourly_costs[hour_key] += entry.total_cost
        
        return dict(sorted(hourly_costs.items()))
    
    def get_top_expensive_calls(self, limit: int = 10) -> List[CostEntry]:
        """Get the most expensive API calls"""
        return sorted(self.entries, key=lambda e: e.total_cost, reverse=True)[:limit]
    
    def save(self):
        """Persist cost data to disk"""
        try:
            data = {
                "version": "1.0",
                "last_updated": datetime.utcnow().isoformat(),
                "entries": [asdict(e) for e in self.entries]
            }
            self.storage_path.write_text(json.dumps(data, indent=2))
            logger.debug(f"Saved {len(self.entries)} cost entries to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save cost data: {e}")
    
    def load(self):
        """Load cost data from disk"""
        if not self.storage_path.exists():
            return
        
        try:
            data = json.loads(self.storage_path.read_text())
            self.entries = [CostEntry(**e) for e in data.get("entries", [])]
            logger.info(f"Loaded {len(self.entries)} cost entries from {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to load cost data: {e}")
            self.entries = []
    
    def reset(self):
        """Clear all cost data"""
        self.entries = []
        if self.storage_path.exists():
            self.storage_path.unlink()
        logger.info("Cost tracking data reset")
    
    def export_csv(self, output_path: str):
        """Export cost data to CSV for analysis"""
        import csv
        
        with open(output_path, 'w', newline='') as f:
            if not self.entries:
                return
            
            fieldnames = ['timestamp', 'operation', 'model', 'input_tokens', 
                         'output_tokens', 'total_cost']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in self.entries:
                writer.writerow({
                    'timestamp': entry.timestamp,
                    'operation': entry.operation,
                    'model': entry.model,
                    'input_tokens': entry.input_tokens,
                    'output_tokens': entry.output_tokens,
                    'total_cost': entry.total_cost
                })
        
        logger.info(f"Exported {len(self.entries)} entries to {output_path}")

# Global tracker instance
_global_tracker: Optional[CostTracker] = None

def get_tracker() -> CostTracker:
    """Get or create the global cost tracker"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
    return _global_tracker

def track_gemini_call(
    operation: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    **metadata
) -> CostEntry:
    """
    Convenience function to track a Gemini API call.
    
    Usage:
        from cost_dashboard import track_gemini_call
        
        track_gemini_call(
            operation="behavior_label",
            model="gemini-1.5-flash",
            input_tokens=150,
            output_tokens=20,
            function="api.connect"
        )
    """
    tracker = get_tracker()
    return tracker.track_call(operation, model, input_tokens, output_tokens, metadata)

def print_cost_summary():
    """Print a formatted cost summary to console"""
    tracker = get_tracker()
    summary = tracker.get_summary()
    
    print("\n" + "=" * 80)
    print("GEMINI API COST SUMMARY")
    print("=" * 80)
    print(f"\n📊 Overall Statistics:")
    print(f"   Total API Calls:     {summary.total_calls:,}")
    print(f"   Total Input Tokens:  {summary.total_input_tokens:,}")
    print(f"   Total Output Tokens: {summary.total_output_tokens:,}")
    print(f"   Total Cost:          ${summary.total_cost:.4f}")
    print(f"   Avg Cost per Call:   ${summary.average_cost_per_call:.6f}")
    
    if summary.cost_by_operation:
        print(f"\n💰 Cost by Operation:")
        for op, cost in sorted(summary.cost_by_operation.items(), key=lambda x: x[1], reverse=True):
            pct = (cost / summary.total_cost * 100) if summary.total_cost > 0 else 0
            print(f"   {op:20s} ${cost:8.4f} ({pct:5.1f}%)")
    
    if summary.cost_by_model:
        print(f"\n🤖 Cost by Model:")
        for model, cost in sorted(summary.cost_by_model.items(), key=lambda x: x[1], reverse=True):
            pct = (cost / summary.total_cost * 100) if summary.total_cost > 0 else 0
            model_name = GEMINI_PRICING.get(model, {}).get("name", model)
            print(f"   {model_name:20s} ${cost:8.4f} ({pct:5.1f}%)")
    
    if summary.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in summary.recommendations:
            print(f"   {rec}")
    
    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    # Demo usage
    tracker = CostTracker()
    
    # Simulate some API calls
    print("Simulating Gemini API calls...")
    
    for i in range(5):
        tracker.track_call(
            operation="behavior_label",
            model="gemini-1.5-flash",
            input_tokens=150,
            output_tokens=20,
            metadata={"function": f"api.function_{i}"}
        )
    
    for i in range(2):
        tracker.track_call(
            operation="rerank",
            model="gemini-1.5-pro",
            input_tokens=500,
            output_tokens=50,
            metadata={"query": f"search_{i}"}
        )
    
    # Print summary
    print_cost_summary()
    
    # Show top expensive calls
    print("Top 3 Most Expensive Calls:")
    for i, entry in enumerate(tracker.get_top_expensive_calls(3), 1):
        print(f"{i}. {entry.operation} ({entry.model}): ${entry.total_cost:.6f}")
    
    # Export to CSV
    tracker.export_csv("cost_report.csv")
    print("\nExported to cost_report.csv")

# Made with Bob
