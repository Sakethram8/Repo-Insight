# tests/test_embedder.py
"""
Unit tests for embedder.py.
Tests embedding generation, cosine similarity, and text building.
"""

import pytest
from unittest.mock import patch, MagicMock
from embedder import embed_text, embed_texts, cosine_similarity, build_embedding_text
from config import EMBEDDING_DIM


# ---------------------------------------------------------------------------
# embed_texts — HTTP path (Tier 2)
# ---------------------------------------------------------------------------

class TestEmbedTextsHttp:
    def _http_response(self, n: int) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "data": [{"index": i, "embedding": [0.1] * EMBEDDING_DIM} for i in range(n)]
        }
        return mock_resp

    def test_http_path_used_when_url_set(self):
        with (
            patch("embedder._HTTP_EMBED_URL", "http://test:30001"),
            patch("embedder._embed_http",
                  return_value=[[0.1] * EMBEDDING_DIM]) as mock_http,
        ):
            result = embed_texts(["hello"])
            mock_http.assert_called_once()
            assert len(result) == 1
            assert len(result[0]) == EMBEDDING_DIM

    def test_http_batch_returns_correct_count(self):
        vecs = [[0.1] * EMBEDDING_DIM] * 3
        with (
            patch("embedder._HTTP_EMBED_URL", "http://test:30001"),
            patch("embedder._embed_http", return_value=vecs),
        ):
            results = embed_texts(["a", "b", "c"])
            assert len(results) == 3

    def test_http_fallback_to_local_on_failure(self):
        local_vecs = [[0.2] * EMBEDDING_DIM]
        with (
            patch("embedder._HTTP_EMBED_URL", "http://test:30001"),
            patch("embedder._embed_http", return_value=[]),   # HTTP fails
            patch("embedder._embed_local", return_value=local_vecs) as mock_local,
        ):
            result = embed_texts(["hello"])
            mock_local.assert_called_once()
            assert result == local_vecs


# ---------------------------------------------------------------------------
# embed_texts — local path (Tier 1)
# ---------------------------------------------------------------------------

class TestEmbedTextsLocal:
    def test_local_path_used_when_no_url(self):
        vecs = [[0.3] * EMBEDDING_DIM]
        with (
            patch("embedder._HTTP_EMBED_URL", None),
            patch("embedder._embed_local", return_value=vecs) as mock_local,
        ):
            result = embed_texts(["hello"])
            mock_local.assert_called_once()
            assert result == vecs

    def test_empty_list_returns_empty(self):
        results = embed_texts([])
        assert results == []


# ---------------------------------------------------------------------------
# embed_texts — graceful fallback (Tier 3)
# ---------------------------------------------------------------------------

class TestEmbedTextsFallback:
    def test_returns_empty_vectors_when_no_backend(self):
        with (
            patch("embedder._HTTP_EMBED_URL", None),
            patch("embedder._embed_local", return_value=[]),  # no local model
        ):
            result = embed_texts(["hello", "world"])
            assert len(result) == 2
            assert all(v == [] for v in result)

    def test_embed_text_returns_empty_list_when_no_backend(self):
        with (
            patch("embedder._HTTP_EMBED_URL", None),
            patch("embedder._embed_local", return_value=[]),
        ):
            result = embed_text("hello")
            assert result == []


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_return_one(self):
        vec = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_return_negative_one(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_mismatched_lengths_raise_value_error(self):
        with pytest.raises(ValueError, match="same length"):
            cosine_similarity([1.0, 2.0], [1.0])

    def test_zero_length_raises_value_error(self):
        with pytest.raises(ValueError, match="zero-length"):
            cosine_similarity([], [])

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_similar_vectors_high_score(self):
        assert cosine_similarity([1.0, 2.0, 3.0], [1.1, 2.1, 3.1]) > 0.99


# ---------------------------------------------------------------------------
# build_embedding_text
# ---------------------------------------------------------------------------

class TestBuildEmbeddingText:
    def test_with_docstring(self):
        result = build_embedding_text("my_func", "Does stuff", "path/to/file.py")
        assert "my_func" in result
        assert "Does stuff" in result

    def test_without_docstring_uses_file_path(self):
        result = build_embedding_text("my_func", None, "path/to/file.py")
        assert "my_func" in result
        assert "path/to/file.py" in result

    def test_empty_docstring_treated_as_falsy(self):
        result = build_embedding_text("my_func", "", "path/to/file.py")
        assert "my_func" in result
        assert "path/to/file.py" in result

    def test_whitespace_docstring_treated_as_falsy(self):
        result = build_embedding_text("my_func", "   ", "path/to/file.py")
        assert "path/to/file.py" in result
