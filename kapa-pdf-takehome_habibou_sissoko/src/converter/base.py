from abc import abstractmethod

from ..loader.types import LoadedPDF


class PDFtoMarkdown:
    """
    Base class for chunkers
    """

    @abstractmethod
    def convert(self, doc: LoadedPDF) -> str:
        """
        Convert a loaded PDF file to markdown
        """
        raise NotImplementedError
