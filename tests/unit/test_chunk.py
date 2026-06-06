from __future__ import annotations

import pytest

from document_qa.chunk import chunk_document, split_sentences


def test_split_sentences_basic() -> None:
    sentences = split_sentences("Hello world. How are you? I am fine!")
    assert sentences == ["Hello world.", "How are you?", "I am fine!"]


def test_chunking_covers_whole_text() -> None:
    text = " ".join(f"word{i}." for i in range(1, 51))
    chunks = chunk_document(text, max_words=10, overlap=3)
    # Every source word must appear in at least one chunk.
    produced = {tok for chunk in chunks for tok in chunk.text.replace(".", "").split()}
    expected = {f"word{i}" for i in range(1, 51)}
    assert expected <= produced


def test_chunk_word_count_respects_max() -> None:
    text = " ".join(f"alpha{i}" for i in range(100)) + "."
    chunks = chunk_document(text, max_words=20, overlap=5)
    assert chunks, "expected at least one chunk"
    assert all(chunk.word_count <= 20 for chunk in chunks)


def test_chunk_overlap_is_applied() -> None:
    # Two sentences that must land in separate chunks given the small budget.
    text = "one two three four five. six seven eight nine ten."
    chunks = chunk_document(text, max_words=6, overlap=2)
    assert len(chunks) >= 2
    prev_tail = chunks[0].text.split()[-2:]
    next_head = chunks[1].text.split()[:2]
    assert prev_tail == next_head


def test_chunk_indices_are_sequential() -> None:
    text = " ".join(f"w{i}." for i in range(30))
    chunks = chunk_document(text, max_words=8, overlap=2)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_empty_document_yields_no_chunks() -> None:
    assert chunk_document("   \n  ") == []


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError):
        chunk_document("hi.", max_words=0)
    with pytest.raises(ValueError):
        chunk_document("hi.", max_words=5, overlap=5)
    with pytest.raises(ValueError):
        chunk_document("hi.", max_words=5, overlap=-1)
