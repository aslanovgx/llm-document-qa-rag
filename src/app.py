from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

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

# ------------------------------------------------------------------------------
# App
# ------------------------------------------------------------------------------
app = FastAPI(
    title="LLM Document Q&A (RAG)",
    version="1.0.0",
    description="Document-based Question Answering using Retrieval-Augmented Generation"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev üçün OK
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    id: int
    source: str
    score: float
    text_preview: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


# ------------------------------------------------------------------------------
# Constants & Helpers
# ------------------------------------------------------------------------------
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def ensure_index_exists() -> None:
    if not Path(settings.faiss_index_path).exists():
        raise HTTPException(status_code=400, detail="FAISS index not found. Call /ingest first.")
    if not Path(settings.metadata_path).exists():
        raise HTTPException(status_code=400, detail="Metadata not found. Call /ingest first.")


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload PDF/TXT -> chunk -> embed -> FAISS -> save index + metadata
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is missing")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".pdf", ".txt"]:
        raise HTTPException(status_code=400, detail="Only .pdf or .txt files are supported")

    save_path = RAW_DIR / file.filename
    content = await file.read()
    save_path.write_bytes(content)

    # Load & chunk
    text = load_text_from_path(str(save_path))
    chunks_text = chunk_text(text, chunk_size=800, overlap=120)

    if not chunks_text:
        raise HTTPException(status_code=400, detail="No text extracted from file")

    chunks = [
        Chunk(id=i, text=t, source=file.filename)
        for i, t in enumerate(chunks_text)
    ]

    # Embedding
    embedder = HFEmbedder(settings.hf_model_name)
    embeddings = embedder.embed([c.text for c in chunks])

    # FAISS
    index = build_faiss_index(embeddings)
    save_index(index, settings.faiss_index_path)
    save_metadata(chunks, settings.metadata_path)

    return {
        "status": "ok",
        "file": file.filename,
        "chunks": len(chunks),
    }


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    """
    Ask a question over ingested documents
    """
    ensure_index_exists()

    index = load_index(settings.faiss_index_path)
    chunks = load_metadata(settings.metadata_path)

    embedder = HFEmbedder(settings.hf_model_name)
    llm = HFClient()  # local, free LLM

    rag = RAGPipeline(
        index=index,
        chunks=chunks,
        embedder=embedder,
        llm=llm,
        top_k=settings.top_k,
    )

    answer, retrieved = rag.answer(payload.question)

    sources: List[SourceItem] = []
    for chunk, score in retrieved:
        sources.append(
            SourceItem(
                id=int(chunk.id),
                source=chunk.source,
                score=float(score),
                text_preview=chunk.text[:240] + ("..." if len(chunk.text) > 240 else ""),
            )
        )

    return AskResponse(answer=answer, sources=sources)
