from pathlib import Path
from chunk import Chunker
from tqdm import tqdm


class Indexer:
    def __init__(self, raw_dir: str = "data/raw/vllm-0.10.1",
                 max_chunk_size: int = 2000):
        self.raw_dir = Path(raw_dir)
        self.chunker = Chunker(max_chunk_size)
        self.all_chunks = []

    def build_index(self) -> list[dict]:
        if not self.raw_dir.exists():
            print("Warning : No files found !")
            return []
        files = (list(self.raw_dir.rglob("*.py")) +
                 list(self.raw_dir.rglob("*.md")))
        for file in tqdm(files, desc="Chunk files..."):
            try:
                with open(file, 'r', encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    file_path = str(file)
                    chunks = self.chunker.chunk_files(content, file_path)
                    self.all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error: {e}")
        return self.all_chunks
