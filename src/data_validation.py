import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.logging_utils import (
    JsonEventLogger,
    build_iteration_dataset_path,
    build_iteration_log_path,
    load_env_file,
)


DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight, fast semantic embeddings
DEFAULT_DEDUP_THRESHOLD = 0.95


class RepairQAModel(BaseModel):
    id: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    equipment_problem: str = Field(..., min_length=1)
    tools_required: List[str] = Field(..., min_length=1)
    steps: List[str] = Field(..., min_length=3)
    safety_info: str = Field(..., min_length=1)
    tips: List[str] = Field(..., min_length=1)
    # Placeholder for a later scoring pipeline.
    judge_scoring: str = Field(default="")


def parse_args() -> argparse.Namespace:
    load_env_file()
    default_raw_input = build_iteration_dataset_path("raw_diy_dataset")
    default_valid_output = build_iteration_dataset_path("diy_dataset")
    default_invalid_output = build_iteration_dataset_path("invalid_diy_dataset")
    parser = argparse.ArgumentParser(
        description=(
            "Deduplicate raw DIY dataset rows and split unique vs duplicate/error rows. "
            "Optionally re-run Pydantic model_validate for defense-in-depth."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default=default_raw_input,
        help=f"Raw input JSONL path (default: {default_raw_input}).",
    )
    parser.add_argument(
        "--valid-output",
        type=str,
        default=default_valid_output,
        help=f"Deduplicated output JSONL path (default: {default_valid_output}).",
    )
    parser.add_argument(
        "--invalid-output",
        type=str,
        default=default_invalid_output,
        help=(
            "Duplicate/error output JSONL path (default: "
            f"{default_invalid_output})."
        ),
    )
    parser.add_argument(
        "--rerun-model-validate",
        action="store_true",
        help=(
            "Re-run RepairQAModel.model_validate on each input row before deduplication. "
            "Recommended for defense-in-depth."
        ),
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=DEFAULT_DEDUP_THRESHOLD,
        help=(
            f"Cosine similarity threshold for semantic deduplication (0.0-1.0). "
            f"Items with similarity >= threshold are flagged as duplicates. "
            f"Default: {DEFAULT_DEDUP_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model label for logging metadata (default: {DEFAULT_MODEL}).",
    )
    return parser.parse_args()


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _build_embedding_text(row: Dict[str, Any]) -> str:
    """Build semantic text for embedding: combines key fields for deduplication."""
    question = _normalize_text(row.get("question", ""))
    equipment_problem = _normalize_text(row.get("equipment_problem", ""))
    answer = _normalize_text(row.get("answer", ""))
    return f"{question} {equipment_problem} {answer}"


def deduplicate_dataset(
    input_path: str,
    valid_output_path: str,
    invalid_output_path: str,
    rerun_model_validate: bool,
    dedup_threshold: float,
    logger: JsonEventLogger,
) -> Tuple[int, int, int]:
    """
    Deduplicate raw dataset using semantic similarity (cosine distance on embeddings).
    Returns (unique_count, duplicate_count, invalid_count).
    """
    logger.print_and_log(
        f"Loading embedding model: {EMBEDDING_MODEL}",
        {"model": EMBEDDING_MODEL},
    )
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    
    Path(valid_output_path).write_text("", encoding="utf-8")
    Path(invalid_output_path).write_text("", encoding="utf-8")

    unique_count = 0
    duplicate_count = 0
    invalid_count = 0
    seen_embeddings: List[Tuple[str, np.ndarray]] = []  # List of (trace_id, embedding)

    with open(input_path, "r", encoding="utf-8") as source, open(
        valid_output_path, "a", encoding="utf-8"
    ) as valid_file, open(invalid_output_path, "a", encoding="utf-8") as invalid_file:
        for line_number, line in enumerate(source, start=1):
            clean = line.strip()
            if not clean:
                continue

            try:
                row = json.loads(clean)
                if not isinstance(row, dict):
                    raise ValueError("Row is not a JSON object.")

                row.setdefault("judge_scoring", "")

                if rerun_model_validate:
                    processed_row = RepairQAModel.model_validate(row).model_dump()
                else:
                    processed_row = row

                trace_id = str(processed_row.get("id", f"line_{line_number:05d}"))
                
                # Generate embedding for this item
                embedding_text = _build_embedding_text(processed_row)
                embedding = embedding_model.encode(embedding_text)
                
                # Check similarity against all previously seen items
                is_duplicate = False
                duplicate_of = None
                if seen_embeddings:
                    # Convert to numpy array for batch similarity computation
                    existing_embeddings = np.array([e[1] for e in seen_embeddings])
                    current_embedding = embedding.reshape(1, -1)
                    
                    # Compute cosine similarities
                    similarities = cosine_similarity(current_embedding, existing_embeddings)[0]
                    max_similarity = float(np.max(similarities))
                    max_idx = int(np.argmax(similarities))
                    
                    if max_similarity >= dedup_threshold:
                        is_duplicate = True
                        duplicate_of = seen_embeddings[max_idx][0]
                
                if is_duplicate:
                    duplicate_count += 1
                    duplicate_row = {
                        "line_number": line_number,
                        "trace_id": trace_id,
                        "duplicate_of": duplicate_of,
                        "reason": "semantic_duplicate",
                        "similarity": float(np.max(similarities)) if seen_embeddings else 0.0,
                        "row": processed_row,
                    }
                    invalid_file.write(json.dumps(duplicate_row, ensure_ascii=False) + "\n")
                else:
                    # Add to seen list
                    seen_embeddings.append((trace_id, embedding))
                    valid_file.write(json.dumps(processed_row, ensure_ascii=False) + "\n")
                    unique_count += 1
            except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as error:
                invalid_count += 1
                invalid_row = {
                    "line_number": line_number,
                    "error": str(error),
                    "raw_row": clean,
                    "judge_scoring": "",
                }
                invalid_file.write(json.dumps(invalid_row, ensure_ascii=False) + "\n")
                logger.print_and_log(
                    f"Invalid row at line {line_number}: {error}",
                    {"line_number": line_number, "error": str(error)},
                )

    return unique_count, duplicate_count, invalid_count


def main() -> None:
    args = parse_args()
    iteration = os.getenv("ITERATION", "1")
    log_path = build_iteration_log_path(extension=".jsonl")
    logger = JsonEventLogger(log_path=log_path, script_name="data_validation", model=args.model, iteration=iteration)

    logger.print_and_log(
        f"Deduplicating raw dataset from {args.input}",
        {
            "input": args.input,
            "valid_output": args.valid_output,
            "invalid_output": args.invalid_output,
            "rerun_model_validate": args.rerun_model_validate,
            "dedup_threshold": args.dedup_threshold,
            "embedding_model": EMBEDDING_MODEL,
        },
    )

    unique_count, duplicate_count, invalid_count = deduplicate_dataset(
        input_path=args.input,
        valid_output_path=args.valid_output,
        invalid_output_path=args.invalid_output,
        rerun_model_validate=args.rerun_model_validate,
        dedup_threshold=args.dedup_threshold,
        logger=logger,
    )

    logger.print_and_log(
        (
            "Deduplication complete. "
            f"unique={unique_count}, duplicates={duplicate_count}, invalid={invalid_count}. "
            f"Saved deduped rows to {args.valid_output} and duplicate/error rows to {args.invalid_output}"
        ),
        {
            "unique_count": unique_count,
            "duplicate_count": duplicate_count,
            "invalid_count": invalid_count,
            "valid_output": args.valid_output,
            "invalid_output": args.invalid_output,
            "rerun_model_validate": args.rerun_model_validate,
            "dedup_threshold": args.dedup_threshold,
            "log_path": log_path,
        },
    )


if __name__ == "__main__":
    main()
