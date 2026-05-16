# embedder.py
"""
Embedding backend with three-tier fallback:

  Tier 1 (local)  — sentence-transformers, CPU-friendly, works offline.
                    Active when sentence_transformers is installed AND
                    EMBED_BASE_URL is not explicitly set.
  Tier 2 (remote) — HTTP POST to a running embedding server (SGLang, OpenAI,
                    or any /encode-compatible endpoint).
                    Active when EMBED_BASE_URL is explicitly set in the environment.
  Tier 3 (none)   — Returns empty vectors; semantic search gracefully returns [].
                    Active when both tiers above fail or are unavailable.

No call site needs to handle failure — the fallback is transparent.
"""

import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

# Tier 2: only active when EMBED_BASE_URL is explicitly set by the caller.
# The presence of the env-var is the signal, not its value.
_HTTP_EMBED_URL   = os.getenv("EMBED_BASE_URL")      # None → use local tier 1
_HTTP_EMBED_MODEL = os.getenv("EMBED_MODEL", "woodx/Qwen3-Embedding-0.6B-SGLang")
_BATCH_SIZE       = 128
_EMBED_TIMEOUT    = 120
_EMBED_RETRIES    = 3

# Tier 1: local sentence-transformers
_LOCAL_MODEL_NAME = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")
_local_model      = None   # lazy-loaded on first use


def _get_local_model():
    global _local_model
    if _local_model is not None:
        return _local_model
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading local embedding model: %s", _LOCAL_MODEL_NAME)
        _local_model = SentenceTransformer(_LOCAL_MODEL_NAME)
        logger.info("Local embedding model ready")
        return _local_model
    except Exception as e:
        logger.warning("Local embedding model unavailable: %s", e)
        return None


def _embed_local(texts: list[str]) -> list[list[float]]:
    model = _get_local_model()
    if model is None:
        return []
    try:
        vecs = model.encode(texts, convert_to_numpy=True,
                            batch_size=512, show_progress_bar=False)
        return vecs.tolist()
    except Exception as e:
        logger.warning("Local embedding failed: %s", e)
        return []


def _embed_http(texts: list[str]) -> list[list[float]]:
    import requests
    base_url = _HTTP_EMBED_URL
    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) + _BATCH_SIZE - 1) // _BATCH_SIZE

    for batch_idx, i in enumerate(range(0, len(texts), _BATCH_SIZE)):
        batch = texts[i : i + _BATCH_SIZE]
        last_err = None
        for attempt in range(1, _EMBED_RETRIES + 1):
            try:
                resp = requests.post(
                    f"{base_url}/encode",
                    json={"model": _HTTP_EMBED_MODEL, "text": batch},
                    timeout=_EMBED_TIMEOUT,
                )
                resp.raise_for_status()
                raw = resp.json()
                data = raw if isinstance(raw, list) else raw["data"]
                if not isinstance(raw, list):
                    data.sort(key=lambda x: x["index"])
                all_embeddings.extend(d["embedding"] for d in data)
                last_err = None
                break
            except Exception as e:
                last_err = e
                logger.warning("HTTP embedding batch %d/%d attempt %d/%d failed: %s",
                               batch_idx + 1, total_batches, attempt, _EMBED_RETRIES, e)
        if last_err is not None:
            logger.warning("HTTP embedding gave up after %d attempts — batch skipped",
                           _EMBED_RETRIES)
            return []   # Tier 3 fallback from within HTTP tier

    return all_embeddings


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings.  Never raises — returns [] per text on failure."""
    if not texts:
        return []

    # Tier 2: explicit HTTP server configured
    if _HTTP_EMBED_URL:
        result = _embed_http(texts)
        if result:
            return result
        logger.warning(
            "HTTP embedding server at %s unavailable — falling back to local model",
            _HTTP_EMBED_URL,
        )

    # Tier 1: local sentence-transformers
    result = _embed_local(texts)
    if result:
        return result

    # Tier 3: graceful empty fallback
    logger.warning(
        "No embedding backend available (set EMBED_BASE_URL or install "
        "sentence-transformers).  Semantic search will be disabled."
    )
    return [[] for _ in texts]


def embed_text(text: str) -> list[float]:
    """Embed a single string.  Returns [] if no backend is available."""
    result = embed_texts([text])
    return result[0] if result else []


def build_embedding_text(
    name: str,
    docstring: str | None,
    file_path: str,
    fingerprint: str | None = None,
) -> str:
    """Build the string that gets embedded for a function or class node.

    Priority: fingerprint → docstring → name + file context.
    The fingerprint contains signature + calls + reads + behavior label —
    richer structural signal than docstring alone, improving semantic search
    for undocumented functions.
    """
    if fingerprint and fingerprint.strip():
        text = fingerprint.strip()[:400]
        last_nl = text.rfind("\n")
        return text[:last_nl] if last_nl > 0 else text
    if docstring and docstring.strip():
        return f"{name}. {docstring.strip()[:300]}"
    return f"{name} in {file_path}"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vectors must be the same length, got {len(a)} and {len(b)}")
    if len(a) == 0:
        raise ValueError("Vectors must not be zero-length")
    vec_a = np.array(a, dtype=np.float32)
    vec_b = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return float(np.dot(vec_a, vec_b) / denom) if denom > 0 else 0.0
