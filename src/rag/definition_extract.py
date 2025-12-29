import re
from typing import Optional

def extract_rag_definition(text: str) -> Optional[str]:
    # "RAG is a technique that ..." cümləsini tuturuq
    pattern = r"(Retrieval-Augmented Generation\s*\(RAG\)\s*is\s*a\s*technique\s*that[^.]*\.)"
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None
