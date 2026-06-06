"""Split documents into overlapping passages (chunks).

Chunking is sentence-aware: text is first segmented into sentences, then
sentences are packed greedily into chunks of at most ``max_words`` words. A
configurable ``overlap`` (in words) is carried from the tail of one chunk into
the head of the next so that context straddling a boundary is not lost. Every
word of the source document appears in at least one chunk.

Part of Document QA by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

_SENTENCE_RE = re.compile(r".+?(?:[.!?]+(?=\s|$)|$)", re.DOTALL)
_WORD_RE = re.compile(r"\S+")


class Chunk(BaseModel):
    """A contiguous passage of a document.

    Attributes:
        index: Zero-based position of this chunk within the document.
        text: The chunk's text content.
        word_count: Number of whitespace-delimited words in ``text``.
    """

    index: int = Field(ge=0)
    text: str
    word_count: int = Field(ge=0)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on ``.``, ``!`` and ``?`` boundaries.

    Args:
        text: The raw document text.

    Returns:
        A list of non-empty, stripped sentence strings (in order).
    """
    return [m.group(0).strip() for m in _SENTENCE_RE.finditer(text) if m.group(0).strip()]


def chunk_document(text: str, max_words: int = 40, overlap: int = 10) -> list[Chunk]:
    """Split a document into overlapping, sentence-aware chunks.

    Sentences are packed greedily until adding the next would exceed
    ``max_words``; the new chunk then starts with the last ``overlap`` words of
    the previous chunk. A single sentence longer than ``max_words`` becomes its
    own chunk rather than being dropped, so coverage of the source is complete.

    Args:
        text: The document to split.
        max_words: Maximum number of words per chunk. Must be positive.
        overlap: Number of trailing words repeated at the start of the next
            chunk. Must be non-negative and strictly less than ``max_words``.

    Returns:
        A list of :class:`Chunk` objects in document order. Empty input yields
        an empty list.

    Raises:
        ValueError: If ``max_words`` or ``overlap`` are out of range.
    """
    if max_words <= 0:
        raise ValueError("max_words must be a positive integer")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= max_words:
        raise ValueError("overlap must be strictly less than max_words")

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current: list[str] = []  # words in the chunk being built

    def flush() -> list[str]:
        """Emit the current chunk and return its overlap tail."""
        if not current:
            return []
        chunks.append(Chunk(index=len(chunks), text=" ".join(current), word_count=len(current)))
        return current[-overlap:] if overlap else []

    for sentence in sentences:
        words = _WORD_RE.findall(sentence)
        if not words:
            continue
        if current and len(current) + len(words) > max_words:
            current = flush()
        current.extend(words)
        # A single oversized sentence: emit immediately so it is never lost.
        while len(current) > max_words:
            head, current = current[:max_words], current[max_words:]
            chunks.append(Chunk(index=len(chunks), text=" ".join(head), word_count=len(head)))
            if overlap:
                current = head[-overlap:] + current

    flush()
    return chunks
