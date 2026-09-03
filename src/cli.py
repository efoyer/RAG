import sys
import json
from pathlib import Path
from typing import Any
from .indexer import Indexer
from .retriever import Retriever
from tqdm import tqdm
import uuid
from .data_models import MinimalSource, MinimalSearchResults
from .data_models import StudentSearchResults, MinimalAnswer
from .data_models import StudentSearchResultsAndAnswer
from .ai import AiGenerator


class RAGCLI:
    def index(self, max_chunk_size: int = 800) -> None:
        indexer = Indexer(max_chunk_size=max_chunk_size)
        indexer.build_index()

    def search(self, query: str, k: int = 3) -> None:
        if not query or not query.strip():
            print("Error: Empty request")
            sys.exit(1)
        safe_k = self._validate_k(k)
        indexer = Indexer(max_chunk_size=800)
        corpus = indexer.build_index()
        retriever = Retriever(chunks=corpus)
        gross_res = retriever.search(query, safe_k)

        to_msr = self._format_single_result(query=query, res_bruts=gross_res)

        final_res = StudentSearchResults(
            search_results=[to_msr],
            k=safe_k
        )
        print(final_res.model_dump_json(indent=4))

    def search_dataset(self, dataset_path: str,
                       save_directory: str, k: int = 3) -> None:
        path_data = Path(dataset_path)
        data_name = path_data.name
        folder_save = Path(save_directory)
        path_save = folder_save / data_name
        safe_k = self._validate_k(k)
        indexer = Indexer(max_chunk_size=1000)
        corpus = indexer.build_index()
        retriever = Retriever(chunks=corpus)

        try:
            with open(path_data, "r") as f:
                json_file = json.load(f)

            question_list = json_file.get("rag_questions", [])
            all_result = []
            for item in tqdm(question_list, desc="Searching dataset..."):
                question_txt = item.get("question")
                q_id = item.get("question_id", str(uuid.uuid4()))
                gross_res = retriever.search(question_txt, safe_k)
                to_msr = self._format_single_result(query=question_txt,
                                                    res_bruts=gross_res,
                                                    q_id=q_id)
                all_result.append(to_msr)

            final_res = StudentSearchResults(
                search_results=all_result,
                k=safe_k
            )
            folder_save.mkdir(parents=True, exist_ok=True)
            with open(path_save, "w", encoding="utf-8") as f:
                f.write(final_res.model_dump_json(indent=4))

        except Exception as e:
            print(e)

    def answer(self, query: str, k: int = 3) -> None:
        if not query or not query.strip():
            print("Error: Empty request")
            sys.exit(1)
        safe_k = self._validate_k(k)
        indexer = Indexer(max_chunk_size=800)
        corpus = indexer.build_index()
        retriever = Retriever(chunks=corpus)
        gross_res = retriever.search(query, safe_k)

        to_msr = self._format_single_result(query=query, res_bruts=gross_res)

        llm = AiGenerator()
        final_answer = llm.generate(query, gross_res)

        to_ma = MinimalAnswer(
            **to_msr.model_dump(),
            answer=final_answer
        )
        final_answer = StudentSearchResultsAndAnswer(
            search_results=[to_ma],
            k=safe_k
        )
        print(final_answer.model_dump_json(indent=4))

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        ssr_path = Path(student_search_results_path)
        if not ssr_path.is_file():
            print(f"{student_search_results_path} is not a file !")
            sys.exit(1)

        folder_save = Path(save_directory)
        content = ssr_path.read_text(encoding="utf-8")
        try:
            search_data = StudentSearchResults.model_validate_json(content)
        except Exception as e:
            print(f"Error parsing json: {e}")
            sys.exit(1)

        llm = AiGenerator()
        lst_res = []

        file_cache: dict[str, str] = {}

        for item in tqdm(search_data.search_results,
                         desc="Generating answers..."):
            question_txt = item.question

            gross_res = []
            for source in item.retrieved_sources:
                file_path = Path(source.file_path)
                if file_path.exists():
                    path_str = str(file_path)

                    if path_str not in file_cache:
                        file_cache[path_str] = file_path.read_text(
                            encoding="utf-8",
                            errors="ignore"
                        )

                    full_text = file_cache[path_str]
                    fragment = full_text[
                        source.first_character_index:
                        source.last_character_index
                    ]

                    gross_res.append({
                        "file_path": source.file_path,
                        "text": fragment
                    })

            raw_answer = llm.generate(question_txt, gross_res)
            to_ma = MinimalAnswer(
                question_id=item.question_id,
                question=question_txt,
                retrieved_sources=item.retrieved_sources,
                answer=raw_answer
            )
            lst_res.append(to_ma)

        final_answer = StudentSearchResultsAndAnswer(
            search_results=lst_res,
            k=search_data.k
        )

        folder_save.mkdir(parents=True, exist_ok=True)
        path_save = folder_save / ssr_path.name

        with open(path_save, "w", encoding="utf-8") as f:
            f.write(final_answer.model_dump_json(indent=4))

        print(f"{path_save}")

    def _validate_k(self, k: Any) -> int:
        try:
            k_int = int(k)
            if k_int <= 0:
                raise ValueError
            return k_int
        except ValueError:
            print("Error: 'k' must be positive")
        except TypeError:
            print("Error: 'k' must be a integer")
        sys.exit(1)

    def _validate_input_file(self, file_path: str) -> Path:
        path = Path(file_path)
        if not path.is_file():
            print("Error: The file does not exist.")
            sys.exit(1)
        return path

    def _format_single_result(self, query: str, res_bruts: list[dict],
                              q_id: str | None = None) -> MinimalSearchResults:
        sources_pydantic = [
            MinimalSource(
                file_path=res["file_path"],
                first_character_index=int(res["first_character_index"]),
                last_character_index=int(res["last_character_index"])
            ) for res in res_bruts
        ]

        return MinimalSearchResults(
            question_id=q_id if q_id else str(uuid.uuid4()),
            question=query,
            retrieved_sources=sources_pydantic
        )
