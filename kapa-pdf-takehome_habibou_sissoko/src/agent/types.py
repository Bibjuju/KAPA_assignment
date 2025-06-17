from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..chunker.types import Chunk


@dataclass
class Document:
    """
    A PDF file and its derived data that travel through the pipeline.
    """

    name: str
    raw_bytes: bytes
    markdown: Optional[str] = None
    chunks: List[Chunk] = None
