# embedder.py
"""
Generate and cache embeddings for functions and classes.
Embeddings are stored as node properties in FalkorDB as JSON-serialised float arrays.
"""

import logging
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL
import json
import numpy as np
import torch

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model exactly once per process."""
    global _model
    if _model is None:
        if torch.cuda.is_available():
            free_vram = torch.cuda.mem_get_info()[0] / 1e9
            device = "cuda" if free_vram > 2.0 else "cpu"
            logger.info("Free VRAM: %.1fGB-using device: %s (ROCm)", free_vram, device)
        else:
            device="cpu"
            logger.warning("ROCm/CUDA not available, using CPU for embeddings (slower)")
        _model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return _model


def embed_text(text: str) -> list[float]:
    """
    Embed a single string. Returns a list of floats of length EMBEDDING_DIM.
    """
    model = _get_model()
    embedding = model.encode(text, convert_to_numpy=True, batch_size=512, show_progress_bar=False)
    return embedding.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings in batches.
    Returns a list of lists of floats.
    """
    if not texts:
        return []
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()


def build_embedding_text(name: str, docstring: str | None, file_path: str) -> str:
    """
    Construct the string that will be embedded for a function or class node.
    Format: "{name}. {docstring}. Defined in {file_path}"
    If docstring is None, omit it: "{name}. Defined in {file_path}"
    """
    if docstring:
        return f"{name}. {docstring}. Defined in {file_path}"
    return f"{name}. Defined in {file_path}"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two equal-length float vectors using numpy.
    Returns a float in [-1.0, 1.0].
    Raises ValueError if vectors differ in length or are zero-length.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Vectors must be the same length, got {len(a)} and {len(b)}"
        )
    if len(a) == 0:
        raise ValueError("Vectors must not be zero-length")

    vec_a = np.array(a,  dtype=np.float32)
    vec_b = np.array(b,  dtype=np.float32)
    denom =np.linalg.norm(vec_a)*np.linalg.norm(vec_b)
    if denom >0:
        return float(np.dot(vec_a, vec_b) / denom )
    else:
        return 0.0
