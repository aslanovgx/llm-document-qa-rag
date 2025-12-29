from pypdf import PdfReader
import re
import io

def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    suffix = filename.lower().split(".")[-1]

    # -------- TXT --------
    if suffix == "txt":
        text = file_bytes.decode("utf-8", errors="ignore")
        return normalize_text(text)

    # -------- PDF --------
    if suffix == "pdf":
        pdf_stream = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_stream)

        texts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            texts.append(t)

        raw_text = "\n".join(texts)
        return normalize_text(raw_text)

    raise ValueError("Unsupported file type. Only .pdf or .txt")
