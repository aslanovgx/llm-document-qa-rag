from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    hf_model_name: str = os.getenv("HF_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    faiss_index_path: str = os.getenv("FAISS_INDEX_PATH", "data/processed/faiss.index")
    metadata_path: str = os.getenv("METADATA_PATH", "data/processed/chunks.json")
    top_k: int = int(os.getenv("TOP_K", "4"))

settings = Settings()