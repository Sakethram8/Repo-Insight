# # embedder.py
# """
# Generate and cache embeddings for functions and classes.
# Embeddings are stored as node properties in FalkorDB as JSON-serialised float arrays.
# """

# import logging
# from sentence_transformers import SentenceTransformer
# from config import EMBEDDING_MODEL
# import json
# import numpy as np
# import torch

# logger = logging.getLogger(__name__)

# _model: SentenceTransformer | None = None

# def _pick_device() -> str:
#     # SGLang occupies the ROCm HSA context — concurrent GPU access deadlocks.
#     # Force CPU for all embedding work when running alongside SGLang.
#     import os
#     if os.getenv("SGLANG_RUNNING", "true").lower() == "true":
#         logger.info("SGLANG_RUNNING=true — forcing CPU for embeddings to avoid ROCm HSA contention")
#         return "cpu"
#     try:
#         import torch
#         if torch.cuda.is_available():
#             free, total = torch.cuda.mem_get_info(0)
#             free_gb = free / 1e9
#             if free_gb > 2.0:
#                 return "cuda"
#     except Exception:
#         pass
#     return "cpu"
# def _get_model() -> SentenceTransformer:
#     """Lazy-load the embedding model exactly once per process."""
#     global _model
#     if _model is None:
#         if torch.cuda.is_available():
#             free_vram = torch.cuda.mem_get_info()[0] / 1e9
#             device = "cuda" if free_vram > 2.0 else "cpu"
#             logger.info("Free VRAM: %.1fGB-using device: %s (ROCm)", free_vram, device)
#         else:
#             device="cpu"
#             logger.warning("ROCm/CUDA not available, using CPU for embeddings (slower)")
#         _model = SentenceTransformer(EMBEDDING_MODEL, device=device)
#     return _model


# def embed_text(text: str) -> list[float]:
#     """
#     Embed a single string. Returns a list of floats of length EMBEDDING_DIM.
#     """
#     model = _get_model()
#     embedding = model.encode(text, convert_to_numpy=True, batch_size=512, show_progress_bar=False)
#     return embedding.tolist()


# def embed_texts(texts: list[str]) -> list[list[float]]:
#     """
#     Embed a list of strings in batches.
#     Returns a list of lists of floats.
#     """
#     if not texts:
#         return []
#     model = _get_model()
#     embeddings = model.encode(texts, convert_to_numpy=True,batch_size=128,show_progress_bar=False)
#     return embeddings.tolist()


# def build_embedding_text(name: str, docstring: str | None, file_path: str) -> str:
#     """
#     Construct the string that will be embedded for a function or class node.
#     Format: "{name}. {docstring}. Defined in {file_path}"
#     If docstring is None, omit it: "{name}. Defined in {file_path}"
#     """
#     if docstring:
#         return f"{name}. {docstring}. Defined in {file_path}"
#     return f"{name}. Defined in {file_path}"


# def cosine_similarity(a: list[float], b: list[float]) -> float:
#     """
#     Compute cosine similarity between two equal-length float vectors using numpy.
#     Returns a float in [-1.0, 1.0].
#     Raises ValueError if vectors differ in length or are zero-length.
#     """
#     if len(a) != len(b):
#         raise ValueError(
#             f"Vectors must be the same length, got {len(a)} and {len(b)}"
#         )
#     if len(a) == 0:
#         raise ValueError("Vectors must not be zero-length")

#     vec_a = np.array(a,  dtype=np.float32)
#     vec_b = np.array(b,  dtype=np.float32)
#     denom =np.linalg.norm(vec_a)*np.linalg.norm(vec_b)
#     if denom >0:
#         return float(np.dot(vec_a, vec_b) / denom )
#     else:
#         return 0.0
# embedder.py
"""
Generate embeddings via the SGLang Qwen3-Embedding server (http://127.0.0.1:30001).
No local model is loaded — zero ROCm contention with the main SGLang server.
"""

import logging
import os
import numpy as np
import requests

logger = logging.getLogger(__name__)

EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://127.0.0.1:30001")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "woodx/Qwen3-Embedding-0.6B-SGLang")
_BATCH_SIZE    = 512   # max texts per HTTP request


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        resp = requests.post(
            f"{EMBED_BASE_URL}/v1/embeddings",
            json={"model": EMBED_MODEL, "input": batch},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        data.sort(key=lambda x: x["index"])
        all_embeddings.extend(d["embedding"] for d in data)
    logger.info("Embedded %d texts via SGLang embedding server", len(texts))
    return all_embeddings


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def build_embedding_text(name: str, docstring: str | None, file_path: str) -> str:
    if docstring:
        return f"{name}. {docstring}. Defined in {file_path}"
    return f"{name}. Defined in {file_path}"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vectors must be the same length, got {len(a)} and {len(b)}")
    if len(a) == 0:
        raise ValueError("Vectors must not be zero-length")
    vec_a = np.array(a, dtype=np.float32)
    vec_b = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return float(np.dot(vec_a, vec_b) / denom) if denom > 0 else 0.0