"""Score and rank chunks against a question.

Two scorers are provided, both returning a ranked top-k:

* :func:`retrieve_by_overlap` — a lightweight TF-style term-overlap score that
  needs no model at all.
* :func:`retrieve_by_embedding` — genuine cosine similarity over vectors from a
  pluggable :data:`~document_qa.providers.Embedder`.

Part of Document QA by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import math
import re
from collections import Counter

from pydantic import BaseModel, Field

from document_qa.chunk import Chunk
from document_qa.providers import Embedder

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class ScoredChunk(BaseModel):
    """A chunk paired with its relevance score for a question.

    Attributes:
        chunk: The retrieved :class:`~document_qa.chunk.Chunk`.
        score: Relevance score; higher is more relevant.
    """

    chunk: Chunk
    score: float = Field(ge=0.0)


def _tokenize(text: str) -> list[str]:
    """Lowercase a string into alphanumeric word tokens."""
    return _TOKEN_RE.findall(text.lower())


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute the cosine similarity between two equal-length vectors.

    Args:
        a: First vector.
        b: Second vector (same length as ``a``).

    Returns:
        Cosine similarity in ``[-1.0, 1.0]``; ``0.0`` if either vector has zero
        magnitude.

    Raises:
        ValueError: If the vectors have different lengths.
    """
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def overlap_score(question: str, passage: str) -> float:
    """Score a passage by TF-weighted term overlap with the question.

    For each distinct question term, the passage's term frequency is summed and
    normalised by the passage length, rewarding passages that mention the
    question's words often without unduly favouring long passages.

    Args:
        question: The user's question.
        passage: Candidate passage text.

    Returns:
        A non-negative relevance score; ``0.0`` when there is no overlap.
    """
    q_terms = set(_tokenize(question))
    if not q_terms:
        return 0.0
    p_tokens = _tokenize(passage)
    if not p_tokens:
        return 0.0
    counts = Counter(p_tokens)
    matched = sum(counts[term] for term in q_terms)
    return matched / len(p_tokens)


def retrieve_by_overlap(question: str, chunks: list[Chunk], top_k: int = 3) -> list[ScoredChunk]:
    """Rank chunks by term overlap and return the top ``top_k``.

    Args:
        question: The user's question.
        chunks: Candidate chunks (e.g. from :func:`document_qa.chunk.chunk_document`).
        top_k: Maximum number of results to return. Must be positive.

    Returns:
        The highest-scoring chunks, descending by score then chunk index, with
        zero-score chunks excluded. May be shorter than ``top_k``.

    Raises:
        ValueError: If ``top_k`` is not positive.
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    scored = [
        ScoredChunk(chunk=chunk, score=overlap_score(question, chunk.text)) for chunk in chunks
    ]
    return _top_k(scored, top_k)


def retrieve_by_embedding(
    question: str,
    chunks: list[Chunk],
    embedder: Embedder,
    top_k: int = 3,
) -> list[ScoredChunk]:
    """Rank chunks by embedding cosine similarity and return the top ``top_k``.

    Args:
        question: The user's question.
        chunks: Candidate chunks.
        embedder: A pluggable text-to-vector callable.
        top_k: Maximum number of results to return. Must be positive.

    Returns:
        The highest-scoring chunks, descending by score then chunk index, with
        zero-score chunks excluded. May be shorter than ``top_k``.

    Raises:
        ValueError: If ``top_k`` is not positive.
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    q_vec = embedder(question)
    scored = [
        ScoredChunk(
            chunk=chunk,
            score=max(0.0, cosine_similarity(q_vec, embedder(chunk.text))),
        )
        for chunk in chunks
    ]
    return _top_k(scored, top_k)


def _top_k(scored: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
    """Sort by score (desc) then chunk index (asc); drop zeros; cap at ``top_k``."""
    ranked = sorted(scored, key=lambda s: (-s.score, s.chunk.index))
    return [s for s in ranked if s.score > 0.0][:top_k]
