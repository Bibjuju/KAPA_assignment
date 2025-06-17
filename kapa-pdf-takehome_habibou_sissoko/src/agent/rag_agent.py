from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI

from ..chunker.base import BaseChunker
from ..converter.base import PDFtoMarkdown
from ..loader.pdf_loader import DirectoryPDFLoader
from ..vector_store.in_memory import InMemoryVectorStore
from .types import Document

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini-2025-04-14")
TOP_K = int(os.getenv("TOP_K", "3"))

logger = logging.getLogger(__name__)


class RAGAgent:
    """End-to-end RAG pipeline wired together with pluggable components."""

    def __init__(
        self,
        loader: DirectoryPDFLoader,
        converter: PDFtoMarkdown,
        chunker: BaseChunker,
        store: InMemoryVectorStore,
        model: str = DEFAULT_MODEL,
        top_k: int = TOP_K,
    ):
        self.loader = loader
        self.converter = converter
        self.chunker = chunker
        self.store = store
        self.model = model
        self.top_k = top_k
        self.client = OpenAI()

        self._docs: Dict[str, Document] = {}  # internal storage

    # --------------------------------------------------------------------- #
    # Properties
    # --------------------------------------------------------------------- #
    @property
    def docs(self) -> Dict[str, Document]:
        """All indexed documents keyed by file name (read-only)."""
        return self._docs

    # --------------------------------------------------------------------- #
    # Indexing
    # --------------------------------------------------------------------- #
    def index(self) -> None:
        """
        Load PDFs → Markdown → chunks → vector store.
        Re-runs are additive; call `self.store.reset()` outside if needed.
        """
        for doc in self.loader.load():
            if doc.name in self._docs:  # skip duplicates
                continue

            markdown = self.converter.convert(doc)
            chunks = self.chunker.split(markdown)

            # Write each chunk
            self.store.add_texts([c.content for c in chunks])

            # bookkeeping
            self._docs[doc.name] = Document(
                name=doc.name,
                raw_bytes=doc.raw_bytes,
                markdown=markdown,
                chunks=chunks,
            )

    # --------------------------------------------------------------------- #
    # Question-answer
    # --------------------------------------------------------------------- #
    def answer(self, query: str) -> Tuple[str, List[Tuple[str, float]]]:
        matches = self.store.search(query, k=self.top_k)
        context = "\n\n---\n\n".join(c for c, _ in matches) or "N/A"

        prompt = (
            "You are a helpful assistant. The context below consists of excerpts "
            "from technical documents. Answer **strictly** using that context. "
            "If the context does not contain the answer, reply that you don’t know.\n\n"
            f"### Context\n{context}\n\n### Question\n{query}\n\n### Answer:"
        )

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return completion.choices[0].message.content.strip(), matches
