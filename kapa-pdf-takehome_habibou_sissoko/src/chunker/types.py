from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class MarkdownSection:
    """
    Represents a heading and the following content in Markdown.
    """

    title: str
    title_url: str
    content: str
    level: int
    sub_sections: List["MarkdownSection"] = field(default_factory=list)


@dataclass
class Chunk:
    """
    Represents a chunk of text.

    Attributes
    ----------
    content : str
        The chunked text including a header, e.g. “# Introduction > Getting Started”.
    original_content : str
        The original text without the header.
    original_title : str
        The most specific Markdown header, e.g. “### Docker Installation”.
    root_title : str
        The highest-level root Markdown header, e.g. “# Introduction”.
    """

    content: str
    original_content: str
    original_title: str
    root_title: str
