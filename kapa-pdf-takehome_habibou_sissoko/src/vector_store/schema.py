"""
In-memory LanceDB hybrid store
──────────────────────────────
• Dense vectors  : all-mpnet-base-v2  (HuggingFace, 768-d)
• Sparse index   : BM25 on ``text`` column
• Hybrid query   : LanceDB blends BM25 + cosine in one call
• Reset support  : drops the table so UI can re-index
"""

from __future__ import annotations

from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

model = get_registry().get("openai").create(name="text-embedding-3-small")


class Document(LanceModel):
    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()
