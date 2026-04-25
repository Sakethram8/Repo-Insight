# config.py

FALKORDB_HOST = "localhost"
FALKORDB_PORT = 6379
GRAPH_NAME = "repo_insight"

SGLANG_BASE_URL = "http://localhost:11434/v1"
SGLANG_API_KEY = "EMPTY"           # SGLang default; required by openai client but unused
LLM_MODEL ="qwen2.5:1.5b"             # Must match the model name served by SGLang

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Sentence-transformer model; runs locally
EMBEDDING_DIM = 384                     # Output dimension for all-MiniLM-L6-v2

IMPACT_RADIUS_MAX_DEPTH = 2        # Maximum Cypher traversal depth for get_impact_radius
IMPACT_RADIUS_WARN_THRESHOLD = 5   # Warn user if impacted node count exceeds this
AGENT_MAX_ITERATIONS = 10          # Hard loop cap for ReAct agent; prevents infinite loops
AGENT_TOOL_TIMEOUT_SECONDS = 5     # Max seconds allowed per tool call before timeout error

FLUSH_GRAPH_ON_INGEST = True       # If True, DROP graph before each ingest run

INGEST_CONCURRENCY = 5             # Max LLM threads for Node Summarization (Set to 100 on MI300X)
