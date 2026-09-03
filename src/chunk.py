from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


class Chunker:
    def __init__(self, max_chunk_size: int = 2000):
        calcul_overlap = int(min(200, max_chunk_size // 10))
        splitter_kwargs = {
            "chunk_size": max_chunk_size,
            "chunk_overlap": calcul_overlap
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

    def chunk_files(self, content: str, files_path: str):
        if files_path.endswith(".py"):
            txt_chunk = self.py_splitter.split_text(content)
        elif files_path.endswith((".md")):
            txt_chunk = self.md_splitter.split_text(content)
        elif files_path.endswith((".txt")):
            txt_chunk = self.txt_splitter.split_text(content)
        else:
            return []

        stored_chunk = []
        current_search_index = 0

        for chunk_txt in txt_chunk:
            start_index = content.find(chunk_txt, current_search_index)
            if start_index == -1:
                start_index = content.find(chunk_txt)
            if start_index != -1:
                end_index = start_index + len(chunk_txt)
                current_search_index = start_index + 1
                stored_chunk.append({
                    "file_path": files_path,
                    "text": chunk_txt,
                    "first_character_index": start_index,
                    "last_character_index": end_index
                })
        return stored_chunk
