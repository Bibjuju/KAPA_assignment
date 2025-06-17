"""Project-wide immutable constants.

The mapping dictionaries below are used extensively throughout the
conversion pipeline.  They are **not** expected to change at runtime,
so we keep them all in a single module to avoid accidental mutation
and to simplify the import graph.
"""

from typing import Dict, Set

__all__ = ["ID2NAME", "CLASSES_TO_OCR"]

ID2NAME: Dict[int, str] = {
    0: "title",
    1: "plain text",
    2: "abandon",
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",
    9: "formula_caption",
}

# The *semantic* classes whose crops should be sent through OCR.
CLASSES_TO_OCR: Set[str] = {
    "title",
    "plain text",
    "figure",
    "figure_caption",
    "table_caption",
    "table_footnote",
    "isolate_formula",
    "formula_caption",
}