# config.py
# config.py (Bulletproof Version)
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

FALKORDB_HOST = os.getenv("FALKORDB_HOST", "172.17.0.1")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))
GRAPH_NAME = os.getenv("GRAPH_NAME", "repo_insight")

SGLANG_BASE_URL = os.getenv("SGLANG_BASE_URL", "http://rocm:30000/v1") # Use container name
SGLANG_API_KEY = "EMPTY"
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3.6-35B-A3B")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

# Optimization Thresholds
# Reduced concurrency to 8 to protect host RAM while Jedi is running
INGEST_CONCURRENCY = int(os.getenv("INGEST_CONCURRENCY", "8")) 
# REDUCED from 60 to 15 to fix "Unterminated String" JSON errors
SUMMARIZATION_BATCH_SIZE = 15 

# Persist graph data across same-repo instances
FLUSH_GRAPH_ON_INGEST = os.getenv("FLUSH_GRAPH_ON_INGEST", "false").lower() in ("true", "1", "yes")

SKIP_DIRS = ["__pycache__", ".git", ".venv", "venv", "node_modules", "dist", "build"]
TOOL_OUTPUT_MAX_LENGTH = 8000
