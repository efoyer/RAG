import bm25s
import Stemmer
from chunk import Chunker


class Retriever:
    def __init__(self, chunks: Chunker):
        self.chunks = chunks
        corpus_txt = [chunk["text"] for chunk in self.chunks]
        self.stemmer = Stemmer.Stemmer("english")
        corpus_tokens = bm25s.tokenize(
            corpus_txt, stopwords="en", stemmer=self.stemmer
            )
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)

    def search(self, query: str | list[str], top_k: int = 10):
        is_str = isinstance(query, str)
        queries = [query] if is_str else query
        safe_top_k = max(1, min(top_k, len(self.chunks))) if self.chunks else 0
        if safe_top_k == 0 or not any(q.strip() for q in queries):
            empty = [[] for _ in queries]
            return empty[0] if is_str else empty

        query_tokens = bm25s.tokenize(
            queries, stopwords="en", stemmer=self.stemmer
        )
        results, _scores = self.retriever.retrieve(query_tokens, k=safe_top_k)
        final_res = []
        for res in results:
            final_res.append([self.chunks[index] for index in res])

        return final_res[0] if is_str else final_res
