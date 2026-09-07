"""Module for retrieving relevant document chunks using BM25."""

import bm25s
import Stemmer
from typing import Any, List, Dict, Union


class Retriever:
    """Handles lexical search using the BM25 algorithm.

    Attributes:
        chunks (List[Dict[str, Any]]): The loaded corpus of document chunks.
        stemmer (Stemmer.Stemmer): English stemmer for tokenization.
        retriever (bm25s.BM25): The initialized BM25 search index.
    """

    def __init__(self, chunks: List[Dict[str, Any]]) -> None:
        """Initializes the retriever and builds the BM25 index.

        Args:
            chunks (List[Dict[str, Any]]): The pre-processed text chunks.
        """
        self.chunks = chunks

        corpus_txt = [chunk["text"] for chunk in self.chunks]
        self.stemmer = Stemmer.Stemmer("english")
        corpus_tokens = bm25s.tokenize(
            corpus_txt, stopwords="en", stemmer=self.stemmer
        )
        self.retriever = bm25s.BM25(k1=1.5, b=0.80)
        self.retriever.index(corpus_tokens)

    def search(
        self, query: Union[str, List[str]], top_k: int = 10
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """Searches the index for the most relevant chunks.

        Args:
            query (Union[str, List[str]]): A single query string
            or list of queries.

            top_k (int, optional): The number of results to return.
            Defaults to 10.

        Returns:
            Union[List[Dict], List[List[Dict]]]: The retrieved chunks.
        """
        if isinstance(query, str):
            queries: list[str] = [query]
            is_str = True
        else:
            queries = query
            is_str = False

        safe_top_k = max(1, min(top_k, len(self.chunks))) if self.chunks else 0
        if safe_top_k == 0 or not any(q.strip() for q in queries):
            empty: list[list[dict[str, Any]]] = [[] for _ in queries]
            return empty[0] if is_str else empty

        query_tokens = bm25s.tokenize(
            queries, stopwords="en", stemmer=self.stemmer
        )
        results, _ = self.retriever.retrieve(query_tokens, k=safe_top_k)

        final_res: List[List[Dict[str, Any]]] = []
        for res in results:
            final_res.append([self.chunks[index] for index in res])

        return final_res[0] if is_str else final_res
