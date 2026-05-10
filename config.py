# config.py (Final Bulletproof Version)
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- FalkorDB & Connectivity ---
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "172.17.0.1")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
GRAPH_NAME = os.getenv("GRAPH_NAME", "repo_insight")

SGLANG_BASE_URL = os.getenv("SGLANG_BASE_URL", "http://rocm:30000/v1")
SGLANG_API_KEY = os.getenv("SGLANG_API_KEY", "EMPTY")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3.6-35B-A3B")

# Baseline model for "fair fight" comparison (defaults to same server/model)
BASELINE_SGLANG_BASE_URL = os.getenv("BASELINE_SGLANG_BASE_URL", SGLANG_BASE_URL)
BASELINE_LLM_MODEL = os.getenv("BASELINE_LLM_MODEL", LLM_MODEL)

# --- Embeddings ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "woodx/Qwen3-Embedding-0.6B-SGLang")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# --- Engine & Change Logic (Required by change_engine.py) ---
IMPACT_RADIUS_MAX_DEPTH = int(os.getenv("IMPACT_RADIUS_MAX_DEPTH", "3"))
BLAST_RADIUS_MAX_DEPTH = int(os.getenv("BLAST_RADIUS_MAX_DEPTH", "4"))
IMPACT_RADIUS_WARN_THRESHOLD = int(os.getenv("IMPACT_RADIUS_WARN_THRESHOLD", "5"))
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))
AGENT_TOOL_TIMEOUT_SECONDS = int(os.getenv("AGENT_TOOL_TIMEOUT_SECONDS", "30"))

# --- Optimized Throughput Guards ---
# Protects host RAM during the Jedi precision pass
INGEST_CONCURRENCY = int(os.getenv("INGEST_CONCURRENCY", "8")) 
# REDUCED from 60 to 15 to ensure JSON remains under token limits
SUMMARIZATION_BATCH_SIZE = 15 

# Persist graph data across instances for same repo
FLUSH_GRAPH_ON_INGEST = os.getenv("FLUSH_GRAPH_ON_INGEST", "false").lower() in ("true", "1", "yes")

# --- Workflow Constants ---
SKIP_DIRS = ["__pycache__", ".git", ".venv", "venv", "node_modules", "dist", "build"]
TEST_COMMAND = os.getenv("TEST_COMMAND", "pytest tests/ -v --tb=short -q -m 'not integration'")
TOOL_OUTPUT_MAX_LENGTH = int(os.getenv("TOOL_OUTPUT_MAX_LENGTH", "8000"))
