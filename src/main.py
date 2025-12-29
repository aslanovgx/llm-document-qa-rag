import typer
from pathlib import Path
from rich import print

from src.config import settings
from src.ingestion.loaders import load_text_from_path
from src.ingestion.chunker import chunk_text
from src.embeddings.hf_embedder import HFEmbedder
from src.vectorstore.schema import Chunk
from src.vectorstore.faiss_store import (
    build_faiss_index,
    save_index,
    save_metadata,
    load_index,
    load_metadata,
)

from src.rag.pipeline import RAGPipeline
from src.llm.hf_client import HFClient

app = typer.Typer(
    help="LLM Document Q&A (RAG) - CLI",
    no_args_is_help=True,
)

# ---------------- INGEST ----------------
@app.command()
def ingest(
    path: str = typer.Option(..., "--path", "-p", help="PDF/TXT file path"),
    chunk_size: int = typer.Option(800, help="chunk size (chars)"),
    overlap: int = typer.Option(120, help="chunk overlap (chars)"),
):
    """Load doc -> chunk -> embed -> build FAISS -> save index+metadata"""

    print(f"[bold]Loading:[/bold] {path}")
    text = load_text_from_path(path)

    print("[bold]Chunking...[/bold]")
    chunks_text = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    if not chunks_text:
        print("[red]No text extracted from file.[/red]")
        raise typer.Exit(code=1)

    chunks = [
        Chunk(id=i, text=t, source=Path(path).name)
        for i, t in enumerate(chunks_text)
    ]

    print(f"[green]Chunks:[/green] {len(chunks)}")

    print(f"[bold]Embedding model:[/bold] {settings.hf_model_name}")
    embedder = HFEmbedder(settings.hf_model_name)
    embeddings = embedder.embed([c.text for c in chunks])

    print("[bold]Building FAISS index...[/bold]")
    index = build_faiss_index(embeddings)

    save_index(index, settings.faiss_index_path)
    save_metadata(chunks, settings.metadata_path)

    print("[green][OK][/green] Ingestion completed.")
    print(f"[dim]Index:[/dim] {settings.faiss_index_path}")
    print(f"[dim]Metadata:[/dim] {settings.metadata_path}")

# ---------------- ASK ----------------
@app.command()
def ask(
    question: str = typer.Option(..., "--question", "-q", help="User question"),
):
    index_path = Path(settings.faiss_index_path)
    meta_path = Path(settings.metadata_path)

    if not index_path.exists() or not meta_path.exists():
        print("[red]Index/metadata not found. Run ingest first.[/red]")
        raise typer.Exit(code=1)

    index = load_index(settings.faiss_index_path)
    chunks = load_metadata(settings.metadata_path)

    embedder = HFEmbedder(settings.hf_model_name)
    llm = HFClient()  # local/free LLM

    rag = RAGPipeline(
        index=index,
        chunks=chunks,
        embedder=embedder,
        llm=llm,
        top_k=settings.top_k,
    )

    answer, retrieved = rag.answer(question)

    print("\n[bold green]Answer:[/bold green]")
    print(answer)

    print("\n[bold]Sources (top-k chunks) [score]:[/bold]")
    for i, (chunk, score) in enumerate(retrieved, 1):
        print(f"[cyan]#{i} | score={score:.3f} | source={chunk.source}[/cyan]")
        print(chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""))
        print("-" * 80)

if __name__ == "__main__":
    app()
