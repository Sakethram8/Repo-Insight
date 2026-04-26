# tests/test_embedder.py
"""
Unit tests for embedder.py.
Tests embedding generation, cosine similarity, and text building.
"""

import pytest
from embedder import embed_text, embed_texts, cosine_similarity, build_embedding_text
from config import EMBEDDING_DIM


class TestEmbedText:
    def test_returns_list_of_floats(self):
        result = embed_text("hello world")
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_returns_correct_dimension(self):
        result = embed_text("test embedding dimension")
        assert len(result) == EMBEDDING_DIM

    def test_empty_string_still_returns_embedding(self):
        result = embed_text("")
        assert isinstance(result, list)
        assert len(result) == EMBEDDING_DIM


class TestEmbedTexts:
    def test_batch_returns_correct_count(self):
        texts = ["hello", "world", "test"]
        results = embed_texts(texts)
        assert len(results) == 3

    def test_batch_dimensions_correct(self):
        texts = ["alpha", "beta"]
        results = embed_texts(texts)
        for emb in results:
            assert len(emb) == EMBEDDING_DIM

    def test_empty_list_returns_empty(self):
        results = embed_texts([])
        assert results == []

    def test_single_item_batch(self):
        results = embed_texts(["single"])
        assert len(results) == 1
        assert len(results[0]) == EMBEDDING_DIM


class TestCosineSimilarity:
    def test_identical_vectors_return_one(self):
        vec = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_return_negative_one(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_mismatched_lengths_raise_value_error(self):
        with pytest.raises(ValueError, match="same length"):
            cosine_similarity([1.0, 2.0], [1.0])

    def test_zero_length_raises_value_error(self):
        with pytest.raises(ValueError, match="zero-length"):
            cosine_similarity([], [])

    def test_zero_vector_returns_zero(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_similar_vectors_high_score(self):
        a = [1.0, 2.0, 3.0]
        b = [1.1, 2.1, 3.1]
        score = cosine_similarity(a, b)
        assert score > 0.99


class TestBuildEmbeddingText:
    def test_with_docstring(self):
        result = build_embedding_text("my_func", "Does stuff", "path/to/file.py")
        assert result == "my_func. Does stuff. Defined in path/to/file.py"

    def test_without_docstring(self):
        result = build_embedding_text("my_func", None, "path/to/file.py")
        assert result == "my_func. Defined in path/to/file.py"

    def test_empty_docstring_treated_as_falsy(self):
        result = build_embedding_text("my_func", "", "path/to/file.py")
        assert result == "my_func. Defined in path/to/file.py"
