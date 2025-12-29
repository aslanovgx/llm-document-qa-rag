from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import json
import numpy as np
import faiss

from .schema import Chunk

def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D array (n, d)")

    # FAISS üçün təhlükəsiz format: contiguous + float32
    x = np.ascontiguousarray(embeddings, dtype=np.float32)

    d = x.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(x)  # type: ignore[arg-type]
    return index

def save_index(index: faiss.Index, index_path: str) -> None:
    ensure_parent_dir(index_path)
    faiss.write_index(index, index_path)

def load_index(index_path: str) -> faiss.Index:
    return faiss.read_index(index_path)

def save_metadata(chunks: List[Chunk], metadata_path: str) -> None:
    ensure_parent_dir(metadata_path)
    payload: List[Dict[str, Any]] = [
        {"id": c.id, "text": c.text, "source": c.source} for c in chunks
    ]
    Path(metadata_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def load_metadata(metadata_path: str) -> List[Chunk]:
    raw = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return [Chunk(**item) for item in raw]
