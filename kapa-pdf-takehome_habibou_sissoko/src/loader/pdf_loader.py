from pathlib import Path
from typing import List

from .types import LoadedPDF


class DirectoryPDFLoader:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> List[LoadedPDF]:
        docs = []
        for p in self.path.glob("*.pdf"):
            with open(p, "rb") as f:
                docs.append(LoadedPDF(name=p.name, raw_bytes=f.read()))
        return docs
