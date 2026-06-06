from __future__ import annotations

from document_qa.providers import LLM
from document_qa.qa import DocumentQA, build_grounded_prompt

_DOC = (
    "The Eiffel Tower is located in Paris. "
    "It was completed in 1889 for the World's Fair. "
    "The tower is made of wrought iron and stands 330 metres tall."
)


def test_built_prompt_includes_retrieved_passage() -> None:
    qa = DocumentQA(_DOC, top_k=2)
    result = qa.answer("When was the Eiffel Tower completed?")
    assert result.passages, "expected retrieved passages"
    top = result.passages[0].chunk.text
    assert top in result.prompt
    assert "1889" in top


def test_prompt_contains_citation_markers() -> None:
    qa = DocumentQA(_DOC, top_k=2)
    result = qa.answer("Where is the Eiffel Tower located?")
    assert "[passage 1]" in result.prompt
    assert "CONTEXT:" in result.prompt
    assert "QUESTION:" in result.prompt


def test_answer_uses_pluggable_llm() -> None:
    captured: dict[str, str] = {}

    def my_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return "CUSTOM ANSWER"

    llm: LLM = my_llm
    qa = DocumentQA(_DOC, llm=llm, top_k=1)
    result = qa.answer("Where is the Eiffel Tower?")
    assert result.answer == "CUSTOM ANSWER"
    assert "CONTEXT:" in captured["prompt"]


def test_embedding_mode_answers() -> None:
    qa = DocumentQA(_DOC, use_embeddings=True, top_k=2)
    result = qa.answer("What is the tower made of?")
    assert result.passages
    assert "Paris" in result.answer or "iron" in result.answer or "1889" in result.answer


def test_build_grounded_prompt_no_passages() -> None:
    prompt = build_grounded_prompt("anything?", [])
    assert "no relevant passages found" in prompt
