from __future__ import annotations

import math

import pytest

from document_qa.providers import echo_llm, fake_embedder, fake_llm


def test_fake_embedder_is_deterministic() -> None:
    embed = fake_embedder()
    assert embed("hello world") == embed("hello world")


def test_fake_embedder_unit_norm() -> None:
    embed = fake_embedder(dim=32)
    vec = embed("photosynthesis converts sunlight")
    norm = math.sqrt(sum(v * v for v in vec))
    assert norm == pytest.approx(1.0)
    assert len(vec) == 32


def test_fake_embedder_shared_words_are_closer() -> None:
    embed = fake_embedder()
    a = embed("the quick brown fox")
    b = embed("the quick brown dog")
    c = embed("completely unrelated vocabulary entirely")
    dot_ab = sum(x * y for x, y in zip(a, b, strict=True))
    dot_ac = sum(x * y for x, y in zip(a, c, strict=True))
    assert dot_ab > dot_ac


def test_fake_embedder_invalid_dim() -> None:
    with pytest.raises(ValueError):
        fake_embedder(dim=0)


def test_fake_llm_uses_context_and_cites() -> None:
    prompt = "CONTEXT:\n[passage 1] Paris is the capital of France.\n\nQUESTION:\nWhere?\n\nANSWER:"
    answer = fake_llm(prompt)
    assert "Paris is the capital of France." in answer
    assert "[passage 1]" in answer


def test_echo_llm_returns_prompt() -> None:
    assert echo_llm("anything") == "anything"
