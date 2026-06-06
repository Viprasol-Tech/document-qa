"""End-to-end document question answering: chunk -> retrieve -> ground -> answer.

:class:`DocumentQA` ingests a document once (chunking it), then for each
question retrieves the most relevant passages, builds a *grounded* prompt that
cites those passages, and calls a pluggable LLM for the final answer. The
default LLM and embedder are the offline fakes from
:mod:`document_qa.providers`, so the class works with no network access.

Part of Document QA by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from document_qa.chunk import Chunk, chunk_document
from document_qa.providers import LLM, Embedder, fake_embedder, fake_llm
from document_qa.retrieve import ScoredChunk, retrieve_by_embedding, retrieve_by_overlap

_PROMPT_TEMPLATE = """\
You are a helpful assistant. Answer the QUESTION using ONLY the CONTEXT below.
Cite the passages you use by their [passage N] markers. If the answer is not in
the context, say you do not know.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


class Answer(BaseModel):
    """The result of answering a question against a document.

    Attributes:
        question: The original question.
        answer: The LLM's answer text.
        prompt: The exact grounded prompt sent to the LLM.
        passages: The retrieved, scored passages used as grounding.
    """

    model_config = ConfigDict(frozen=True)

    question: str
    answer: str
    prompt: str
    passages: list[ScoredChunk] = Field(default_factory=list)


def build_grounded_prompt(question: str, passages: list[ScoredChunk]) -> str:
    """Render a grounded prompt that cites each retrieved passage.

    Each passage is prefixed with a ``[passage N]`` marker (1-based) so the
    model — and the reader — can see exactly which evidence supports the answer.

    Args:
        question: The user's question.
        passages: Retrieved, scored passages to ground the answer in.

    Returns:
        A prompt string containing a numbered, cited ``CONTEXT`` block and the
        ``QUESTION``.
    """
    if passages:
        context = "\n\n".join(
            f"[passage {i}] {scored.chunk.text}" for i, scored in enumerate(passages, start=1)
        )
    else:
        context = "(no relevant passages found)"
    return _PROMPT_TEMPLATE.format(context=context, question=question)


class DocumentQA:
    """Answer questions about a single document using grounded retrieval.

    The document is chunked once at construction time. Each
    :meth:`answer` call retrieves the top passages, builds a grounded prompt,
    and delegates the final wording to the pluggable LLM.
    """

    def __init__(
        self,
        document: str,
        *,
        llm: LLM | None = None,
        embedder: Embedder | None = None,
        use_embeddings: bool = False,
        max_words: int = 40,
        overlap: int = 10,
        top_k: int = 3,
    ) -> None:
        """Ingest and chunk a document, wiring up the providers.

        Args:
            document: The full document text to answer questions about.
            llm: Prompt-to-answer callable. Defaults to the offline fake LLM.
            embedder: Text-to-vector callable. Defaults to the offline fake
                embedder; only used when ``use_embeddings`` is ``True``.
            use_embeddings: If ``True``, retrieve via embedding cosine
                similarity; otherwise use TF-style term overlap.
            max_words: Maximum words per chunk (see
                :func:`document_qa.chunk.chunk_document`).
            overlap: Word overlap between consecutive chunks.
            top_k: Number of passages to retrieve per question. Must be positive.

        Raises:
            ValueError: If ``top_k`` is not positive.
        """
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        self._llm: LLM = llm if llm is not None else fake_llm
        self._embedder: Embedder = embedder if embedder is not None else fake_embedder()
        self._use_embeddings = use_embeddings
        self._top_k = top_k
        self.chunks: list[Chunk] = chunk_document(document, max_words=max_words, overlap=overlap)

    def retrieve(self, question: str) -> list[ScoredChunk]:
        """Retrieve the top passages for a question.

        Args:
            question: The user's question.

        Returns:
            The retrieved, scored passages (possibly empty).
        """
        if self._use_embeddings:
            return retrieve_by_embedding(question, self.chunks, self._embedder, top_k=self._top_k)
        return retrieve_by_overlap(question, self.chunks, top_k=self._top_k)

    def answer(self, question: str) -> Answer:
        """Answer a question with a grounded, cited prompt.

        Args:
            question: The user's question.

        Returns:
            An :class:`Answer` bundling the answer text, the grounded prompt,
            and the passages used as evidence.
        """
        passages = self.retrieve(question)
        prompt = build_grounded_prompt(question, passages)
        return Answer(
            question=question,
            answer=self._llm(prompt),
            prompt=prompt,
            passages=passages,
        )
