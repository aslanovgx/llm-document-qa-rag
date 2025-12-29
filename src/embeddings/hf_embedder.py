from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class HFEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Returns: (n, d) float32 numpy array
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Type checker üçün explicit cast
        embeddings = np.asarray(embeddings, dtype="float32")
        return embeddings