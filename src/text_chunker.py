"""Module for chunking text and code documents."""

from typing import Dict, Any, List
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


class Chunker:
    """Splits documents into smaller chunks based on language structure.

    Attributes:
        md_splitter (RecursiveCharacterTextSplitter): Splitter for Markdown.
        txt_splitter (RecursiveCharacterTextSplitter): Splitter for plain text.
        py_splitter (RecursiveCharacterTextSplitter): Splitter for Python code.
    """

    def __init__(self, max_chunk_size: int = 800) -> None:
        """Initializes language-specific text splitters.

        Args:
            max_chunk_size (int, optional): Maximum characters per chunk.
                Defaults to 800.
        """
        calcul_overlap = int(min(200, max_chunk_size // 10))
        splitter_kwargs = {
            "chunk_size": max_chunk_size,
            "chunk_overlap": calcul_overlap,
            "add_start_index": True
        }
        self.md_splitter = RecursiveCharacterTextSplitter.from_language(
           language=Language.MARKDOWN,
           keep_separator=True,
           **splitter_kwargs
        )
        self.txt_splitter = RecursiveCharacterTextSplitter(
           separators=["\n\n## ", "\n\n", "\n", " ", ""],
           keep_separator=True,
           **splitter_kwargs
        )
        self.py_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            **splitter_kwargs
        )

    def chunk_files(self, content: str, files_path: str
                    ) -> List[Dict[str, Any]]:
        """Splits file content into indexed chunks.

        Args:
            content (str): The raw text of the file.
            files_path (str): The path to the file to determine language.

        Returns:
            List[Dict[str, Any]]: A list of chunk dictionaries with indices.
        """
        if files_path.endswith(".py"):
            docs = self.py_splitter.create_documents([content])
        elif files_path.endswith(".md"):
            docs = self.md_splitter.create_documents([content])
        elif files_path.endswith(".txt"):
            docs = self.txt_splitter.create_documents([content])
        else:
            return []

        stored_chunk: List[Dict[str, Any]] = []

        for doc in docs:
            chunk_txt = str(doc.page_content)
            start_index = int(doc.metadata.get("start_index", -1))
            if start_index != -1:
                end_index = start_index + len(chunk_txt)
                stored_chunk.append({
                    "file_path": files_path,
                    "text": chunk_txt,
                    "first_character_index": start_index,
                    "last_character_index": end_index
                })
        return stored_chunk
