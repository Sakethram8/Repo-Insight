# config.py
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

FALKORDB_HOST = os.getenv("FALKORDB_HOST", "172.17.0.1")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
GRAPH_NAME = os.getenv("GRAPH_NAME", "repo_insight")

SGLANG_BASE_URL = os.getenv("SGLANG_BASE_URL", "http://localhost:30000/v1")
SGLANG_API_KEY = os.getenv("SGLANG_API_KEY", "EMPTY")        # SGLang default; required by openai client but unused
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3.6-35B-A3B")          # Override via env var for production (e.g. Qwen3-8B)

# Baseline model for "fair fight" A/B comparison
# This is the STRONGER model that runs WITHOUT the graph — proving structure > scale


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")   # Sentence-transformer model; runs locally
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))                # Output dimension for all-MiniLM-L6-v2

IMPACT_RADIUS_MAX_DEPTH = int(os.getenv("IMPACT_RADIUS_MAX_DEPTH", "3"))
BLAST_RADIUS_MAX_DEPTH = int(os.getenv("BLAST_RADIUS_MAX_DEPTH", "3"))
IMPACT_RADIUS_WARN_THRESHOLD = int(os.getenv("IMPACT_RADIUS_WARN_THRESHOLD", "5"))
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))
AGENT_TOOL_TIMEOUT_SECONDS = int(os.getenv("AGENT_TOOL_TIMEOUT_SECONDS", "30"))

FLUSH_GRAPH_ON_INGEST = os.getenv("FLUSH_GRAPH_ON_INGEST", "false").lower() in ("true", "1", "yes")

INGEST_CONCURRENCY = int(os.getenv("INGEST_CONCURRENCY", "20"))   # Max LLM threads for Node Summarization

SKIP_DIRS = ["__pycache__", ".git", ".venv", "venv", "node_modules", "dist", "build"]
TEST_COMMAND = os.getenv("TEST_COMMAND", "pytest tests/ -v --tb=short -q -m 'not integration'")
TOOL_OUTPUT_MAX_LENGTH = int(os.getenv("TOOL_OUTPUT_MAX_LENGTH", "8000"))
