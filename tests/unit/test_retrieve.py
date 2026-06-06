from __future__ import annotations

import math

import pytest

from document_qa.chunk import chunk_document
from document_qa.providers import fake_embedder
from document_qa.retrieve import (
    cosine_similarity,
    overlap_score,
    retrieve_by_embedding,
    retrieve_by_overlap,
)

_DOC = (
    "The sky is blue during a clear day. "
    "Photosynthesis lets plants convert sunlight into chemical energy. "
    "The capital of France is Paris."
)


def test_cosine_similarity_known_values() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 1.0], [1.0, 0.0]) == pytest.approx(1 / math.sqrt(2))


def test_cosine_similarity_zero_vector() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_similarity_length_mismatch() -> None:
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 2.0])


def test_overlap_score_rewards_matching_terms() -> None:
    assert overlap_score("capital of France", "The capital of France is Paris.") > 0.0
    assert overlap_score("capital of France", "Plants use sunlight.") == 0.0


def test_overlap_ranks_answer_passage_first() -> None:
    chunks = chunk_document(_DOC, max_words=12, overlap=0)
    results = retrieve_by_overlap("What is the capital of France?", chunks, top_k=3)
    assert results, "expected at least one retrieved passage"
    assert "Paris" in results[0].chunk.text


def test_embedding_ranks_answer_passage_first() -> None:
    chunks = chunk_document(_DOC, max_words=12, overlap=0)
    results = retrieve_by_embedding(
        "How do plants make energy from sunlight?",
        chunks,
        fake_embedder(),
        top_k=3,
    )
    assert results
    assert "Photosynthesis" in results[0].chunk.text


def test_retrieve_excludes_zero_scores() -> None:
    chunks = chunk_document(_DOC, max_words=12, overlap=0)
    results = retrieve_by_overlap("Paris", chunks, top_k=10)
    assert all(r.score > 0.0 for r in results)
    assert len(results) < len(chunks)


def test_top_k_is_respected() -> None:
    chunks = chunk_document(_DOC, max_words=8, overlap=0)
    results = retrieve_by_overlap("the", chunks, top_k=1)
    assert len(results) <= 1


def test_invalid_top_k_raises() -> None:
    chunks = chunk_document(_DOC, max_words=8, overlap=0)
    with pytest.raises(ValueError):
        retrieve_by_overlap("paris", chunks, top_k=0)
