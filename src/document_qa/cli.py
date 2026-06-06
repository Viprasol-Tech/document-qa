"""Command-line interface for Document QA.

``document-qa demo`` answers a question over a short built-in document fully
offline (using the fake providers), printing the retrieved passages and the
grounded answer — no API keys, no network.

Part of Document QA by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markup import escape

from document_qa import __version__
from document_qa.qa import DocumentQA

app = typer.Typer(add_completion=False, help="Document QA — by Viprasol Tech.")
console = Console()

_BUILTIN_DOCUMENT = """\
The Apollo program was a series of human spaceflight missions run by NASA.
Apollo 11 was the first mission to land humans on the Moon. It launched on
July 16, 1969, carrying Neil Armstrong, Buzz Aldrin, and Michael Collins.
Neil Armstrong became the first person to walk on the lunar surface on
July 20, 1969. The program ended in 1972 with the Apollo 17 mission, which
remains the most recent crewed landing on the Moon.
"""


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"document-qa [bold cyan]{__version__}[/] - by Viprasol Tech")


@app.command()
def demo(
    question: str = typer.Option(
        "Who was the first person to walk on the Moon?",
        help="Question to ask the built-in document.",
    ),
    use_embeddings: bool = typer.Option(
        False, help="Retrieve via fake-embedding cosine instead of term overlap."
    ),
    top_k: int = typer.Option(2, help="Number of passages to retrieve."),
) -> None:
    """Answer a question over a short built-in document, fully offline."""
    qa = DocumentQA(_BUILTIN_DOCUMENT, use_embeddings=use_embeddings, top_k=top_k)
    result = qa.answer(question)

    console.print(f"Chunks:   [bold]{len(qa.chunks)}[/]")
    console.print(f"Question: [bold]{result.question}[/]")
    console.print("Retrieved passages:")
    if result.passages:
        for i, scored in enumerate(result.passages, start=1):
            text = escape(" ".join(scored.chunk.text.split()))
            console.print(f"  passage {i} (score {scored.score:.3f}): {text}")
    else:
        console.print("  [yellow]none[/]")
    console.print(f"Answer:   [bold green]{escape(result.answer)}[/]")


if __name__ == "__main__":
    app()
