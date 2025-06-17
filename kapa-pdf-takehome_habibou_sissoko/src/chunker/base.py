from abc import abstractmethod
from typing import List

from .types import Chunk


class BaseChunker:
    """
    Base class for chunkers
    """

    def __init__(
        self,
        max_chunk_size: int = 2000,
    ):
        self.max_chunk_size = max_chunk_size

    @abstractmethod
    def split(self, content: str) -> List[Chunk]:
        """
        Chunk a string into a list of chunks
        """
        raise NotImplementedError
