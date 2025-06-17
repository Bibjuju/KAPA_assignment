import logging
import re
from typing import List, Optional, Tuple

import mistune
from langchain.text_splitter import RecursiveCharacterTextSplitter
from mistune.core import BlockState
from mistune.renderers.markdown import MarkdownRenderer

from .base import BaseChunker
from .types import Chunk, MarkdownSection
from .utils import sanitize_title

logger = logging.getLogger(__name__)


class MarkdownSectionChunker(BaseChunker):
    """
    Chunker that chunks based on markdown sections
    and optionally with a character limit per chunk.
    It makes use of the inherent hierarchy of Markdown.
    """

    def split(self, content: str) -> List[Chunk]:
        """
        Divide a markdown into chunks
        - The chunks will not be larger than 'self.max_chunk_size'
        - if specified the title_prefix will be prepended to top level headings
        """
        markdown_sections = self._parse_markdown_into_sections(content)
        chunks = []

        for section in markdown_sections:
            chunks.extend(
                self._chunk_markdown_section(section, self.max_chunk_size, None)
            )

        return chunks

    def _render_ast_token_to_markdown(self, ast_token: dict) -> str:
        """
        Render an AST token to a markdown string
        """
        renderer = MarkdownRenderer()
        return renderer([ast_token], state=BlockState())

    def _parse_markdown_into_sections(self, content: str) -> List[MarkdownSection]:
        """
        Divide a markdown string into hierarchical sections.
        A section is content that is grouped together, optionally under a heading.
        """
        sections = []

        markdown = mistune.create_markdown(renderer=None)
        content_ast = markdown(content)

        # Fold the content tokens into the headings
        for token in content_ast:
            if token["type"] == "heading":
                # Create a new section
                heading = self._render_ast_token_to_markdown(token)
                title, title_url = self._maybe_extract_title_url_from_heading(heading)

                new_section = MarkdownSection(
                    title=title,
                    title_url=title_url,  # TO DO, maybe I can take this out
                    content="",
                    sub_sections=[],
                    level=token["attrs"]["level"],
                )
                sections.append(new_section)

            else:
                # If it is not a heading the content is added to the last section on the stack
                token_markdown = self._render_ast_token_to_markdown(token).strip()

                if len(sections) > 0:
                    sections[-1].content += f"\n{token_markdown}"
                    sections[-1].content = sections[-1].content.strip()
                else:
                    # if there is text before the first markdown heading, we treat it as its own section without a title
                    new_section = MarkdownSection(
                        title="",
                        title_url=None,
                        content=token_markdown,
                        sub_sections=[],
                        level=1,
                    )
                    sections.append(new_section)

        # Collapse the stack to the highest level of sections
        index = len(sections) - 1

        while index > 0:
            if sections[index].level > sections[index - 1].level:
                sections[index - 1].sub_sections.append(sections[index])
                del sections[index]
                index = len(sections) - 1
            else:
                index -= 1
        return sections

    @staticmethod
    def _maybe_extract_title_url_from_heading(heading: str) -> Tuple[str, str | None]:
        """
        Extracts the title and title_url (if it exists) from a markdown heading
        """

        def is_heading_a_link(s):
            """
            Checks if a string starts with a `[` and ends with a `)` by disregarding
            leading hashes (`#`) and leading / trailing whitespaces
            """
            starts_with_square_bracket = r"^\s*#*\s*\["
            ends_with_bracket = r"\)\s*$"

            return bool(re.match(starts_with_square_bracket, s)) and bool(
                re.search(ends_with_bracket, s)
            )

        if not is_heading_a_link(heading):
            return sanitize_title(heading), None

        title = heading
        title_url = None

        # Regular expression pattern for a Markdown link in the string
        pattern = r"\[(.*?)\]\((.*?)\)"

        # Find the match
        match = re.search(pattern, heading)
        if match:
            # return title name, title url
            title = match.group(1)
            title_url = match.group(2)
        return sanitize_title(title), title_url

    def _increment_markdown_heading_levels(self, markdown_text: str) -> str:
        """
        Increase all headings in a markdown text by 1 level.
        """
        markdown = mistune.create_markdown(renderer=None)
        content_ast = markdown(markdown_text)

        for token in content_ast:
            if token["type"] == "heading":
                token["attrs"]["level"] += 1

        return "\n".join(
            [self._render_ast_token_to_markdown(token).strip() for token in content_ast]
        )

    def _create_chunk_heading(self, heading_prefix: Optional[str], title: str) -> str:
        """
        Create a heading based on a title and prefix for the chunk
        """
        if heading_prefix and len(title) > 0:
            return heading_prefix + " > " + title
        elif heading_prefix:
            return heading_prefix
        elif len(title) > 0:
            return title
        else:
            return ""

    def _create_root_heading(self, title: str) -> str:
        """
        Create a single higher level heading for the chunk (i.e., always the first heading of the chunk)
        """
        if len(title) > 0:
            return title
        return ""

    def _split_text(self, text: str, max_chars: int) -> List[str]:
        """
        Recursively split a string into smaller chunks
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=0,
            length_function=len,
        ).split_text(text)

    def _format_split_title(self, heading: str, split: int) -> str:
        """
        Format a title with an additional split indicator
        """
        if heading:
            return f"{heading} Part {split}"
        return ""

    def _format_section_with_heading(
        self, heading: str, content: str, split: Optional[int]
    ) -> str:
        """
        Format a section with heading and optional split indicator
        """
        if heading and split:
            return f"# {self._format_split_title(heading, split)}\n{content}"
        elif heading:
            return f"# {heading}\n{content}"
        return content

    def _chunk_markdown_section(
        self,
        section: MarkdownSection,
        max_chars: Optional[int],
        heading_prefix: Optional[str],
        heading_parent: Optional[str] = None,  # Add heading_parent parameter
    ) -> List[Chunk]:
        """
        Recursively divide a 'MarkdownSectionChunker' into chunks. If parent and child sections
        are smaller than the 'max_chars' limit they will be merged.
        """
        heading = self._create_chunk_heading(heading_prefix, section.title)

        # If heading_parent is None, it means we are at the first heading of the document
        if heading_parent is None:
            heading_parent = self._create_root_heading(section.title)

        result: list[Chunk] = []

        content = section.content

        # Split the content if it is too long
        if max_chars and len(content) > max_chars:
            # Split the document into multiple documents of `max_chars` length
            splits = self._split_text(content, max_chars)

            for i in range(len(splits)):
                split = splits[i]
                split_index_label = (
                    i + 1
                )  # 1-based index for labeling chunks (ie: Title #1, Title #2, etc.)
                new_chunk = Chunk(
                    content=self._format_section_with_heading(
                        heading, split, split_index_label
                    ),
                    original_content=split,
                    original_title=self._format_split_title(
                        section.title, split_index_label
                    ),
                    root_title=heading_parent,
                )
                result.append(new_chunk)
        else:
            new_chunk = Chunk(
                content=self._format_section_with_heading(heading, content, None),
                original_content=content,
                original_title=self._create_root_heading(section.title),
                root_title=heading_parent,
            )
            result.append(new_chunk)

        # Create documents from children
        children: list[Chunk] = []
        for sub_section in section.sub_sections:
            tmp = self._chunk_markdown_section(
                sub_section, max_chars, heading, heading_parent
            )

            children.extend(tmp)

        # At the rest of the children to the result
        result.extend(children)
        return result
