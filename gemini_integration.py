# gemini_integration.py
"""
Google Gemini integration for Repo-Insight.
Provides two architectural integrations (not just wrappers):

1. Behavior Label Generation (Tier 2 fingerprints) via Gemini Flash
2. Candidate Reranking for issue context via Gemini Pro

For TechEx Track 2: AI Agents with Google AI Studio
"""

import logging
import os
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Try to import Gemini SDK
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Gemini features disabled.")
    logger.warning("Install with: pip install google-generativeai")


def configure_gemini():
    """Configure Gemini API with key from environment."""
    if not GEMINI_AVAILABLE:
        return False
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set. Gemini features disabled.")
        return False
    
    genai.configure(api_key=api_key)
    return True


# ---------------------------------------------------------------------------
# Feature 1: Behavior Label Generation (Tier 2 Fingerprints)
# ---------------------------------------------------------------------------

def generate_behavior_label_gemini(skeleton: str, fqn: str) -> Optional[str]:
    """
    Use Gemini Flash for fast, cheap one-line behavior label generation.
    This is Tier 2 of the fingerprint system.
    
    Gemini Flash is optimized for:
    - Low latency (ideal for real-time label generation)
    - Cost efficiency (cheaper than Pro for simple tasks)
    - High throughput (can generate labels for many functions quickly)
    
    Args:
        skeleton: AST-stripped code skeleton (~38% of source tokens)
        fqn: Fully qualified name of the function
    
    Returns:
        One-line behavior description, or None if generation fails
    
    Example:
        Input skeleton:
        ```python
        def validate_user(username, password):
            if not username or not password:
                raise ValueError
            user = db.query(User).filter_by(username=username).first()
            if not user or not user.check_password(password):
                raise AuthenticationError
            return user
        ```
        
        Output: "Validates user credentials against database and returns user object"
    """
    if not configure_gemini():
        return None
    
    try:
        # Track cost for this operation
        from cost_dashboard import track_gemini_call
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""Given this Python function skeleton, generate a single-line behavior description in code-comment style.

Function: {fqn}

Skeleton:
```python
{skeleton}
```

Generate ONE concise line describing what this function does, like:
- "Validates user credentials and returns auth token"
- "Filters queryset by date range and status"
- "Parses JSON response and extracts error codes"
- "Calculates total price with tax and discounts"

Focus on the WHAT (behavior), not the HOW (implementation details).

Behavior:"""
        
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=50,  # One line only
                temperature=0.3,  # Low temperature for consistency
            )
        )
        
        # Track the API call cost
        input_tokens = len(prompt.split()) * 1.3  # Rough estimate
        output_tokens = len(response.text.split()) * 1.3
        track_gemini_call(
            operation="behavior_label",
            model="gemini-1.5-flash",
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            function=fqn
        )
        
        label = response.text.strip()
        
        # Clean up common prefixes
        for prefix in ["Behavior:", "Description:", "//", "#"]:
            if label.startswith(prefix):
                label = label[len(prefix):].strip()
        
        # Ensure it's a single line
        label = label.split('\n')[0].strip()
        
        logger.info(f"Generated behavior label for {fqn}: {label}")
        return label
        
    except Exception as e:
        logger.error(f"Failed to generate behavior label with Gemini: {e}")
        return None


def generate_behavior_labels_batch(
    skeletons: List[Dict[str, str]], 
    max_concurrent: int = 5
) -> Dict[str, str]:
    """
    Generate behavior labels for multiple functions in batch.
    
    Args:
        skeletons: List of dicts with 'fqn' and 'skeleton' keys
        max_concurrent: Maximum concurrent API calls
    
    Returns:
        Dict mapping fqn -> behavior_label
    
    Example:
        skeletons = [
            {'fqn': 'auth.User.login', 'skeleton': '...'},
            {'fqn': 'auth.User.logout', 'skeleton': '...'},
        ]
        labels = generate_behavior_labels_batch(skeletons)
        # {'auth.User.login': 'Authenticates user...', 'auth.User.logout': 'Logs out user...'}
    """
    if not configure_gemini():
        return {}
    
    labels = {}
    
    # Simple sequential processing for now
    # TODO: Add concurrent processing with asyncio for production
    for item in skeletons:
        fqn = item['fqn']
        skeleton = item['skeleton']
        
        label = generate_behavior_label_gemini(skeleton, fqn)
        if label:
            labels[fqn] = label
    
    return labels


# ---------------------------------------------------------------------------
# Feature 2: Candidate Reranking for Issue Context
# ---------------------------------------------------------------------------

def rerank_candidates_with_gemini(
    issue_text: str,
    candidates: List[Dict],
    top_k: int = 10
) -> List[Dict]:
    """
    Two-stage retrieval: graph hybrid search → Gemini rerank.
    
    Gemini Pro is used for:
    - Advanced reasoning about code-issue relationships
    - Understanding complex bug descriptions
    - Semantic matching beyond simple embeddings
    
    This is architecturally interesting because:
    1. Graph provides initial candidates (fast, deterministic)
    2. Gemini refines ranking (semantic, context-aware)
    3. Best of both worlds: speed + accuracy
    
    Args:
        issue_text: GitHub/Jira issue description
        candidates: Top 20 candidates from graph hybrid search
        top_k: Number of candidates to return after reranking
    
    Returns:
        Reranked list of top_k candidates with updated scores
    
    Example:
        Issue: "QuerySet filter crashes with Q objects"
        Graph returns 20 candidates based on embeddings + name tokens
        Gemini reranks based on understanding of Django ORM and Q objects
        Returns top 10 most relevant functions
    """
    if not configure_gemini():
        logger.warning("Gemini not available, returning original candidates")
        return candidates[:top_k]
    
    if len(candidates) <= top_k:
        return candidates
    
    try:
        # Track cost for this operation
        from cost_dashboard import track_gemini_call
        
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Take top 20 from graph for reranking
        top_20 = candidates[:20]
        
        # Build candidate list for prompt
        candidate_list = "\n".join([
            f"{i+1}. {c['fqn']} (graph_score: {c.get('score', 0):.3f})\n"
            f"   Summary: {c.get('summary', 'No summary available')}\n"
            f"   File: {c.get('file_path', 'Unknown')}"
            for i, c in enumerate(top_20)
        ])
        
        prompt = f"""You are a code analysis expert helping to localize bugs in a large codebase.

Given this issue description and 20 candidate functions from a code knowledge graph, rerank them by likelihood of being the bug location or relevant to fixing the bug.

Issue Description:
{issue_text}

Candidate Functions (from graph hybrid search):
{candidate_list}

Consider:
1. Semantic relevance to the issue description
2. Function names and their relationship to the problem
3. Module/file paths that suggest relevance
4. Summary descriptions that match the issue

Return ONLY the top {top_k} function numbers (1-20) in order of relevance, comma-separated.
Example: 3,7,1,15,9,2,11,4,18,5

Top {top_k} function numbers:"""
        
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=100,
                temperature=0.1,  # Very low temperature for consistent ranking
            )
        )
        
        # Track the API call cost
        input_tokens = len(prompt.split()) * 1.3
        output_tokens = len(response.text.split()) * 1.3
        track_gemini_call(
            operation="rerank_candidates",
            model="gemini-1.5-pro",
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            issue_length=len(issue_text),
            num_candidates=len(candidates)
        )
        
        # Parse response
        ranking_text = response.text.strip()
        logger.info(f"Gemini reranking response: {ranking_text}")
        
        # Extract numbers
        import re
        numbers = [int(n) for n in re.findall(r'\d+', ranking_text)]
        
        # Validate and reorder candidates
        reranked = []
        seen_indices = set()
        
        for num in numbers[:top_k]:
            idx = num - 1  # Convert to 0-based index
            if 0 <= idx < len(top_20) and idx not in seen_indices:
                candidate = top_20[idx].copy()
                candidate['gemini_rank'] = len(reranked) + 1
                candidate['original_rank'] = idx + 1
                reranked.append(candidate)
                seen_indices.add(idx)
        
        # Fill remaining slots with original order if needed
        for i, candidate in enumerate(top_20):
            if len(reranked) >= top_k:
                break
            if i not in seen_indices:
                candidate = candidate.copy()
                candidate['gemini_rank'] = len(reranked) + 1
                candidate['original_rank'] = i + 1
                reranked.append(candidate)
        
        logger.info(f"Reranked {len(reranked)} candidates with Gemini Pro")
        return reranked
        
    except Exception as e:
        logger.error(f"Failed to rerank with Gemini: {e}")
        logger.warning("Falling back to original graph ranking")
        return candidates[:top_k]


# ---------------------------------------------------------------------------
# Feature 3: Multi-Modal Code Understanding (Future)
# ---------------------------------------------------------------------------

def analyze_code_with_context(
    code: str,
    context: str,
    question: str
) -> Optional[str]:
    """
    Use Gemini's multi-modal capabilities for advanced code analysis.
    
    This is a future enhancement that could:
    - Analyze code + documentation together
    - Understand code + test cases
    - Process code + error logs
    
    Args:
        code: Source code to analyze
        context: Additional context (docs, tests, logs)
        question: Specific question to answer
    
    Returns:
        Analysis result or None if failed
    """
    if not configure_gemini():
        return None
    
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        prompt = f"""Analyze this code in the given context and answer the question.

Code:
```python
{code}
```

Context:
{context}

Question: {question}

Answer:"""
        
        response = model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Failed to analyze code with Gemini: {e}")
        return None


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def test_gemini_connection() -> bool:
    """Test if Gemini API is configured and working."""
    if not configure_gemini():
        return False
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Say 'OK' if you can read this.")
        return 'OK' in response.text or 'ok' in response.text.lower()
    except Exception as e:
        logger.error(f"Gemini connection test failed: {e}")
        return False


def get_gemini_model_info() -> Dict[str, any]:
    """Get information about available Gemini models."""
    if not configure_gemini():
        return {"available": False, "error": "Gemini not configured"}
    
    try:
        models = genai.list_models()
        return {
            "available": True,
            "models": [
                {
                    "name": m.name,
                    "display_name": m.display_name,
                    "description": m.description,
                }
                for m in models
                if 'gemini' in m.name.lower()
            ]
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Integration with Existing Tools
# ---------------------------------------------------------------------------

def enhance_fingerprinting_with_gemini(
    fqn: str,
    skeleton: str,
    existing_label: Optional[str] = None
) -> Optional[str]:
    """
    Enhance the fingerprinting system with Gemini-generated labels.
    
    This integrates with fingerprinting.py's store_behavior_labels tool.
    
    Args:
        fqn: Function fully qualified name
        skeleton: Code skeleton from Tier 1
        existing_label: Existing label if any
    
    Returns:
        New behavior label or existing label if generation fails
    """
    if existing_label and existing_label.strip():
        logger.info(f"Using existing label for {fqn}")
        return existing_label
    
    logger.info(f"Generating new label for {fqn} with Gemini Flash")
    new_label = generate_behavior_label_gemini(skeleton, fqn)
    
    return new_label if new_label else existing_label


if __name__ == "__main__":
    # Test script
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Gemini Integration...")
    print("-" * 60)
    
    # Test connection
    print("\n1. Testing Gemini connection...")
    if test_gemini_connection():
        print("✓ Gemini API is working!")
    else:
        print("✗ Gemini API connection failed")
        print("  Make sure GEMINI_API_KEY is set in your environment")
        exit(1)
    
    # Test behavior label generation
    print("\n2. Testing behavior label generation...")
    test_skeleton = """
def validate_user(username, password):
    if not username or not password:
        raise ValueError("Username and password required")
    user = db.query(User).filter_by(username=username).first()
    if not user or not user.check_password(password):
        raise AuthenticationError("Invalid credentials")
    return user
"""
    
    label = generate_behavior_label_gemini(test_skeleton, "auth.User.validate_user")
    if label:
        print(f"✓ Generated label: {label}")
    else:
        print("✗ Label generation failed")
    
    # Test candidate reranking
    print("\n3. Testing candidate reranking...")
    test_issue = "QuerySet filter crashes when using Q objects with complex conditions"
    test_candidates = [
        {"fqn": "django.db.models.query.QuerySet.filter", "score": 0.85, "summary": "Filters queryset"},
        {"fqn": "django.db.models.Q.__init__", "score": 0.75, "summary": "Q object constructor"},
        {"fqn": "django.db.models.sql.query.Query.build_filter", "score": 0.70, "summary": "Builds SQL filter"},
    ]
    
    reranked = rerank_candidates_with_gemini(test_issue, test_candidates, top_k=2)
    if reranked:
        print(f"✓ Reranked {len(reranked)} candidates")
        for i, c in enumerate(reranked, 1):
            print(f"  {i}. {c['fqn']} (original rank: {c.get('original_rank', '?')})")
    else:
        print("✗ Reranking failed")
    
    print("\n" + "-" * 60)
    print("Gemini integration test complete!")

# Made with Bob
