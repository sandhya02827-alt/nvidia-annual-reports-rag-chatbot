
import pytest

from src.retrieve import retrieve_passages


class DummyEmbeddingModel:
    def encode(self, *args, **kwargs):
        raise AssertionError(
            "Embedding should not run for an empty question."
        )


def test_empty_question_returns_empty_list():
    result = retrieve_passages(
        "",
        DummyEmbeddingModel(),
        index=None,
        chunks=[],
    )
    assert result == []


def test_whitespace_question_returns_empty_list():
    result = retrieve_passages(
        "   ",
        DummyEmbeddingModel(),
        index=None,
        chunks=[],
    )
    assert result == []


def test_top_k_must_be_at_least_one():
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        retrieve_passages(
            "NVIDIA revenue",
            DummyEmbeddingModel(),
            index=None,
            chunks=[],
            top_k=0,
        )
