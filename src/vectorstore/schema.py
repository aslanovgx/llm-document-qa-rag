from dataclasses import dataclass

@dataclass
class Chunk:
    id: int
    text: str
    source: str
