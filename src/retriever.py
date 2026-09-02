import BM25
from chunk import Chunker


class Retriever:
    def __init__(self, chunks: Chunker):
        self.chunks = chunks
        corpus = [chunk["text"] for chunk in self.chunks]
        self.retriever = BM25.index(corpus, language="english")

    def search(self, query: str | list[str], top_k: int = 10):
        is_str = isinstance(query, str)
        queries = [query] if is_str else query
        gross_res = self.retriever.search(queries, top_k)
        self.chunk_map = {chunk["text"]: chunk for chunk in self.chunks}
        final_res = []
        for tmp_res in gross_res:
            tmp_l = []
            for res in tmp_res:
                txt_found = res["document"]

                if txt_found in self.chunk_map:
                    tmp_l.append(self.chunk_map[txt_found])
            final_res.append(tmp_l)
        return final_res[0] if is_str else final_res
