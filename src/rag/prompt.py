from __future__ import annotations
from typing import List, Tuple, Union
from src.vectorstore.schema import Chunk

ChunkWithScore = Tuple[Chunk, float]

def build_context(items: Union[List[Chunk], List[ChunkWithScore]]) -> str:
    parts = []
    for i, item in enumerate(items, 1):
        if isinstance(item, tuple):
            c, score = item
            parts.append(f"[Chunk {i} | score={score:.3f} | source={c.source}]\n{c.text}")
        else:
            c = item
            parts.append(f"[Chunk {i} | source={c.source}]\n{c.text}")
    return "\n\n".join(parts)

def build_prompt(question: str, context: str) -> str:
    return f"""Answer using ONLY the context.
If the answer is not in the context, say: I don't know based on the provided document.

Context:
{context}

Question: {question}

Rules:
- First line must start with: Definition:
- Then add: Steps: with 2 bullets
- Add a final line: Sources: (Chunk X, Chunk Y)

Answer:
"""
