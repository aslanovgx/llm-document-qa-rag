from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import secrets
from dataclasses import dataclass

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import settings
from src.ingestion.loaders import load_text_from_path
from src.ingestion.chunker import chunk_text
from src.embeddings.hf_embedder import HFEmbedder
from src.vectorstore.schema import Chunk
from src.vectorstore.faiss_store import build_faiss_index
from src.rag.pipeline import RAGPipeline
from src.llm.hf_client import HFClient


# ------------------------------------------------------------------------------
# App
# ------------------------------------------------------------------------------
app = FastAPI(
    title="LLM Document Q&A (RAG)",
    version="1.0.0",
    description="Ephemeral (in-memory) RAG: ingest a PDF/TXT, ask questions, no persistence."
)

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
    doc_id: str
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
# In-memory session store (NO DB, NO persistence)
# ------------------------------------------------------------------------------
@dataclass
class SessionState:
    index: Any  # faiss.Index
    chunks: List[Chunk]
    created_at: float


SESSIONS: Dict[str, SessionState] = {}
SESSION_TTL_SECONDS = 30 * 60  # 30 dəqiqə
MAX_SESSIONS = 50


def cleanup_sessions() -> None:
    now = time.time()

    expired = [
        k for k, v in SESSIONS.items()
        if (now - v.created_at) > SESSION_TTL_SECONDS
    ]
    for k in expired:
        SESSIONS.pop(k, None)

    # sadə limit (DoS-a qarşı)
    if len(SESSIONS) > MAX_SESSIONS:
        oldest = sorted(SESSIONS.items(), key=lambda kv: kv[1].created_at)[: len(SESSIONS) - MAX_SESSIONS]
        for k, _ in oldest:
            SESSIONS.pop(k, None)


# ------------------------------------------------------------------------------
# Temp directory (only for reading PDFs via loader; deleted immediately)
# ------------------------------------------------------------------------------
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload PDF/TXT -> chunk -> embed -> build FAISS -> store in RAM -> return doc_id
    """
    cleanup_sessions()

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is missing")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".pdf", ".txt"]:
        raise HTTPException(status_code=400, detail="Only .pdf or .txt files are supported")

    # Temp write because loader accepts a file path
    tmp_path = RAW_DIR / f"tmp_{secrets.token_hex(8)}{suffix}"
    tmp_path.write_bytes(await file.read())

    try:
        text = load_text_from_path(str(tmp_path))
    finally:
        # delete temp file (no persistence)
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    chunks_text = chunk_text(text, chunk_size=800, overlap=120)
    if not chunks_text:
        raise HTTPException(status_code=400, detail="No text extracted from file")

    chunks = [Chunk(id=i, text=t, source=file.filename) for i, t in enumerate(chunks_text)]

    embedder = HFEmbedder(settings.hf_model_name)
    embeddings = embedder.embed([c.text for c in chunks])

    index = build_faiss_index(embeddings)

    doc_id = secrets.token_hex(16)
    SESSIONS[doc_id] = SessionState(index=index, chunks=chunks, created_at=time.time())

    return {"status": "ok", "doc_id": doc_id, "file": file.filename, "chunks": len(chunks)}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    """
    Ask a question over the ingested document (in-memory only).
    """
    cleanup_sessions()

    state = SESSIONS.get(payload.doc_id)
    if not state:
        raise HTTPException(status_code=400, detail="Session expired or invalid doc_id. Please ingest again.")

    embedder = HFEmbedder(settings.hf_model_name)
    llm = HFClient()  # local/free LLM

    rag = RAGPipeline(
        index=state.index,
        chunks=state.chunks,
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


@app.post("/reset")
def reset(doc_id: str) -> Dict[str, str]:
    """
    Optional: delete a session explicitly (e.g., user clicks "Reset").
    """
    SESSIONS.pop(doc_id, None)
    return {"status": "ok"}
