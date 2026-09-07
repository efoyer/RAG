"""Module responsible for indexing and chunking source files."""

import json
from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm
from .text_chunker import Chunker


class Indexer:
    """Handles reading, chunking, and saving the document corpus.

    Attributes:
        raw_dir (Path): Path to the directory containing raw files.
        chunker (Chunker): Text chunking tool (Markdown, Python, etc.).
        all_chunks (List[Dict[str, Any]]): Global list storing the segments.
        save_path (Path): Dynamic path for the JSON save file.
    """

    def __init__(self, raw_dir: str = "data/raw/vllm-0.10.1",
                 max_chunk_size: int = 2000) -> None:
        """Initializes the indexer with chunking parameters.

        Args:
            raw_dir (str, optional): Source directory.
                Defaults to "data/raw/vllm-0.10.1".
            max_chunk_size (int, optional): Maximum allowed size per chunk.
                Defaults to 2000.
        """
        self.raw_dir = Path(raw_dir)
        self.chunker = Chunker(max_chunk_size)
        self.all_chunks: List[Dict[str, Any]] = []
        self.save_path = Path(f"data/processed/corpus_{max_chunk_size}_.json")

    def build_index(self) -> List[Dict[str, Any]]:
        """Builds the index by chunking files or loads the existing cache.

        Returns:
            List[Dict[str, Any]]: The list of generated or cached chunks.
        """
        if not self.raw_dir.exists():
            print("Warning : No files found !")
            return []

        if self.save_path.exists():
            with open(self.save_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                cache_res: list[dict[str, Any]] = []
                if isinstance(raw_data, list):
                    for item in raw_data:
                        if isinstance(item, dict):
                            cache_res.append(item)
                return cache_res

        files = (list(self.raw_dir.rglob("*.py")) +
                 list(self.raw_dir.rglob("*.md")) +
                 list(self.raw_dir.rglob("*.txt")))

        for file in tqdm(files, desc="Chunk files..."):
            try:
                with open(file, 'r', encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    file_path = str(file)
                    chunks = self.chunker.chunk_files(content, file_path)
                    self.all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error reading {file}: {e}")

        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump(self.all_chunks, f, indent=4)
        except Exception as e:
            print(f"Error saving index: {e}")

        return self.all_chunks
