"""Module for evaluating the RAG pipeline performance (Recall and MRR)."""

from pathlib import Path
from typing import Dict, List, Tuple
from pydantic import ValidationError

from .data_models import (
    AnsweredQuestion,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    MinimalSource
)


def get_iou(chunk_a: MinimalSource, chunk_b: MinimalSource) -> float:
    """Calculates the Intersection over Union (IoU) between two text chunks.

    Args:
        chunk_a (MinimalSource): The first text segment.
        chunk_b (MinimalSource): The second text segment.

    Returns:
        float: The overlap ratio (between 0.0 and 1.0).
    """
    if chunk_a.file_path != chunk_b.file_path:
        return 0.0

    start_max = max(chunk_a.first_character_index,
                    chunk_b.first_character_index)
    end_min = min(chunk_a.last_character_index,
                  chunk_b.last_character_index)

    intersection = max(0, end_min - start_max)
    if intersection == 0:
        return 0.0

    start_min = min(chunk_a.first_character_index,
                    chunk_b.first_character_index)
    end_max = max(chunk_a.last_character_index,
                  chunk_b.last_character_index)
    union = end_max - start_min

    return float(intersection / union) if union > 0 else 0.0


def score_query(predictions: List[MinimalSource],
                truths: List[MinimalSource]) -> Tuple[float, float]:
    """Evaluates a single query by calculating its Recall and MRR.

    Args:
        predictions (List[MinimalSource]): The sources found by the model.
        truths (List[MinimalSource]): The reference sources (Ground Truth).

    Returns:
        Tuple[float, float]: A tuple containing (Recall, MRR).
    """
    if not truths:
        return 0.0, 0.0

    found_count = sum(
        1 for truth in truths
        if any(get_iou(pred, truth) >= 0.05 for pred in predictions)
    )
    recall = float(found_count / len(truths))

    mrr = 0.0
    for rank, pred in enumerate(predictions, start=1):
        if any(get_iou(pred, truth) >= 0.05 for truth in truths):
            mrr = 1.0 / rank
            break

    return recall, mrr


def load_student_data(filepath: Path
                      ) -> (StudentSearchResultsAndAnswer |
                            StudentSearchResults | None):
    """Loads and validates the student results JSON file via Pydantic.

    Args:
        filepath (Path): Path to the results JSON file.

    Returns:
        The validated data as a Pydantic object, or None in case of an error.
    """
    content = filepath.read_text(encoding="utf-8")
    try:
        ans_data = StudentSearchResultsAndAnswer.model_validate_json(content)
        assert isinstance(ans_data, StudentSearchResultsAndAnswer)
        return ans_data
    except ValidationError:
        try:
            search_data = StudentSearchResults.model_validate_json(content)
            assert isinstance(search_data, StudentSearchResults)
            return search_data
        except ValidationError:
            return None


def evaluate(results_path: str, dataset_path: str) -> None:
    """CLI command to mathematically evaluate the RAG pipeline outputs.

    Args:
        results_path (str): Path to the generated JSON results.
        dataset_path (str): Path to the Ground Truth JSON dataset.
    """
    max_len = 2000
    ans_file = Path(results_path)
    ref_file = Path(dataset_path)

    if not ans_file.exists() or not ref_file.exists():
        print("Error: Missing input files.")
        return

    student_results = load_student_data(ans_file)
    if not student_results:
        print("Error: Invalid student JSON format.")
        return

    limit_k = student_results.k

    try:
        truth_data = RagDataset.model_validate_json(
            ref_file.read_text(encoding="utf-8"))
    except ValidationError:
        print("Error: Invalid ground truth JSON.")
        return

    reference_map: Dict[str, AnsweredQuestion] = {
        q.question_id: q for q in truth_data.rag_questions if isinstance(
            q, AnsweredQuestion)
    }

    processed = 0
    total_recall = 0.0
    total_mrr = 0.0

    for entry in student_results.search_results:
        if entry.question_id not in reference_map:
            continue

        expected = reference_map[entry.question_id]

        valid_preds = [
            src for src in entry.retrieved_sources
            if ((src.last_character_index - src.first_character_index)
                <= max_len)
        ][:limit_k]

        rec, mrr = score_query(valid_preds, expected.sources)
        total_recall += rec
        total_mrr += mrr
        processed += 1

    if processed == 0:
        print("ERROR: No matching queries evaluated.")
        return

    print(f"\n{'='*40}\n        RAG EVALUATION REPORT\n{'='*40}")
    print(f"Questions evaluated : {processed}")
    print(f"Recall @ k ({limit_k})        : {total_recall / processed:.4f}")
    print(f"MRR @ k ({limit_k})           : {total_mrr / processed:.4f}")
    print(f"{'='*40}")
