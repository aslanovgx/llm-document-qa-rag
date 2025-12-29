from __future__ import annotations

from typing import List, Tuple, Protocol
import faiss

from src.embeddings.hf_embedder import HFEmbedder
from src.vectorstore.schema import Chunk
from src.rag.retriever import retrieve_top_k
from src.rag.prompt import build_context, build_prompt
from src.rag.definition_extract import extract_rag_definition
from src.rag.reranker import CEReranker


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


class RAGPipeline:
    def __init__(
        self,
        index: faiss.Index,
        chunks: List[Chunk],
        embedder: HFEmbedder,
        llm: LLMClient,
        top_k: int = 4,
        faiss_top_k: int = 12,
        min_faiss_score: float = 0.20,  # ↓ “documentdə yoxdur” üçün guard
    ):
        self.index = index
        self.chunks = chunks
        self.embedder = embedder
        self.llm = llm
        self.top_k = top_k
        self.faiss_top_k = faiss_top_k
        self.min_faiss_score = min_faiss_score
        self.reranker = CEReranker()

    def _is_rag_question(self, question: str) -> bool:
        q = question.lower()
        keywords = [
            "rag",
            "retrieval-augmented",
            "retrieval augmented",
            "vector search",
            "faiss",
            "embedding",
            "chunk",
            "retrieval",
        ]
        return any(k in q for k in keywords)

    def answer(self, question: str) -> Tuple[str, List[Tuple[Chunk, float]]]:
        q_emb = self.embedder.embed([question])

        retrieved: List[Tuple[Chunk, float]] = retrieve_top_k(
            self.index, q_emb, self.chunks, top_k=self.faiss_top_k
        )

        # ---------- Guard 1: heç nə tapılmayıb ----------
        if not retrieved:
            return "I don't know based on the provided document.", []

        # ---------- Guard 2: sənəddə uyğunluq zəifdir (question documentdə yoxdur) ----------
        # NOTE: bu check FAISS score üstündədir (rerank score deyil)
        best_faiss_score = float(retrieved[0][1])
        if best_faiss_score < self.min_faiss_score:
            # yenə də “sources” göstərmək istəsən, top_k qədər qaytara bilərsən
            return "I don't know based on the provided document.", retrieved[: self.top_k]

        # ---------- Rerank ----------
        reranked: List[Tuple[Chunk, float]] = self.reranker.rerank(
            question, retrieved, top_k=self.top_k
        )

        context = build_context(reranked)
        prompt = build_prompt(question, context)

        # ---------- RAG definition fallback yalnız RAG sualları üçün ----------
        definition = extract_rag_definition(context)
        if definition and self._is_rag_question(question):
            sources = ", ".join([f"Chunk {c.id}" for (c, _s) in reranked])
            final_answer = (
                f"Definition: {definition}\n"
                f"Steps:\n"
                f"- Retrieve relevant chunks via vector search (FAISS)\n"
                f"- Use retrieved chunks as context for generation\n"
                f"Sources: ({sources})"
            )
            return final_answer, reranked

        # ---------- Normal LLM answer (document context ilə) ----------
        final_answer = self.llm.generate(prompt).strip()
        return final_answer, reranked