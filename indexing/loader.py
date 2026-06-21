from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader

from config import KNOWLEDGE_SRC_DIR


class DocumentLoader:
    """Loads PDF documents from a local directory."""

    def __init__(self, src_dir: str = KNOWLEDGE_SRC_DIR):
        self.src_dir = src_dir

    def load(self) -> list:
        reader = SimpleDirectoryReader(
            input_dir=self.src_dir,
            file_extractor={".pdf": PDFReader()},
        )
        documents = reader.load_data()
        print(f"Loaded {len(documents)} documents.")
        return documents
