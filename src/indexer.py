from pathlib import Path
from .chunk import Chunker
from tqdm import tqdm
import json


class Indexer:
    def __init__(self, raw_dir: str = "data/raw/vllm-0.10.1",
                 max_chunk_size: int = 2000):
        self.raw_dir = Path(raw_dir)
        self.chunker = Chunker(max_chunk_size)
        self.all_chunks = []
        self.save_path = Path(f"data/processed/corpus_{max_chunk_size}_.json")

    def build_index(self) -> list[dict]:
        if not self.raw_dir.exists():
            print("Warning : No files found !")
            return []
        if self.save_path.exists():
            with open(self.save_path, "r") as f:
                return json.load(f)

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
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump(self.all_chunks, f, indent=4)
        except Exception as e:
            print(f"Error: {e}")

        return self.all_chunks
