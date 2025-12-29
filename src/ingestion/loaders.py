from pathlib import Path
from pypdf import PdfReader
import re

def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_text_from_path(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    suffix = p.suffix.lower()

    if suffix == ".txt":
        return normalize_text(
            p.read_text(encoding="utf-8", errors="ignore")
        )

    if suffix == ".pdf":
        reader = PdfReader(str(p))
        texts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            texts.append(t)

        raw_text = "\n".join(texts)
        return normalize_text(raw_text)

    raise ValueError(f"Unsupported file type: {suffix}. Use .pdf or .txt")
