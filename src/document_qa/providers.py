"""Pluggable model providers — a swappable LLM and embedder.

The whole library treats model access as two simple callables:

* an **LLM** is ``Callable[[str], str]`` (prompt in, answer out);
* an **embedder** is ``Callable[[str], list[float]]`` (text in, vector out).

That is all the rest of the code depends on, so you can plug in OpenAI,
a local model, or anything else without touching the core. This module also
ships deterministic, offline **fake** providers so the demo and the test
suite run with no network and no API keys.

Part of Document QA by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable

# Type aliases describing the pluggable contract.
LLM = Callable[[str], str]
Embedder = Callable[[str], list[float]]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase a string into alphanumeric word tokens."""
    return _TOKEN_RE.findall(text.lower())


def fake_embedder(dim: int = 64) -> Embedder:
    """Build a deterministic, offline embedder for tests and demos.

    Each token is hashed into the vector space (a tiny "hashing trick"
    bag-of-words), so texts that share words get similar vectors. The result
    is L2-normalised, which makes a dot product equal to cosine similarity.
    No randomness is involved, so the same text always yields the same vector.

    Args:
        dim: Dimensionality of the produced vectors. Must be positive.

    Returns:
        An :data:`Embedder` mapping a string to a unit-length ``list[float]``.

    Raises:
        ValueError: If ``dim`` is not a positive integer.
    """
    if dim <= 0:
        raise ValueError("dim must be a positive integer")

    def embed(text: str) -> list[float]:
        vec = [0.0] * dim
        for token in _tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign
        norm = math.sqrt(sum(value * value for value in vec))
        if norm == 0.0:
            return vec
        return [value / norm for value in vec]

    return embed


def fake_llm(prompt: str) -> str:
    """Answer a grounded prompt offline, with no model call.

    This canned LLM does not "reason"; it extracts the ``CONTEXT`` block from a
    prompt built by :class:`document_qa.qa.DocumentQA`, returns the first
    context sentence, and echoes the passage citations. That is enough to prove
    the full pipeline end to end while staying fully deterministic.

    Args:
        prompt: A grounded prompt containing a ``CONTEXT:`` section.

    Returns:
        A short answer string referencing the supplied context.
    """
    context = _extract_section(prompt, "CONTEXT:")
    citations = re.findall(r"\[passage \d+\]", context)
    body = _first_sentence(_strip_citation_markers(context)) or "No answer found in context."
    if citations:
        return f"{body} (source: {', '.join(dict.fromkeys(citations))})"
    return body


def echo_llm(prompt: str) -> str:
    """Return the prompt verbatim — handy for inspecting what was sent.

    Args:
        prompt: The prompt that would be sent to a model.

    Returns:
        The prompt unchanged.
    """
    return prompt


def _extract_section(prompt: str, header: str) -> str:
    """Return the text following ``header`` up to the next ALL-CAPS header."""
    start = prompt.find(header)
    if start == -1:
        return ""
    start += len(header)
    rest = prompt[start:]
    match = re.search(r"\n[A-Z][A-Z ]+:", rest)
    end = match.start() if match else len(rest)
    return rest[:end].strip()


def _strip_citation_markers(text: str) -> str:
    """Remove ``[passage N]`` markers from a context block."""
    return re.sub(r"\[passage \d+\]\s*", "", text).strip()


def _first_sentence(text: str) -> str:
    """Return the first sentence of ``text`` (period/question/exclamation)."""
    match = re.search(r".+?[.!?](?=\s|$)", text, re.DOTALL)
    if match:
        return match.group(0).strip()
    return text.strip().splitlines()[0].strip() if text.strip() else ""
