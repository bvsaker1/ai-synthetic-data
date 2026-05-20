import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


CATEGORIES = ["appliance", "electrical", "plumbing", "general_home", "hvac"]
DEFAULT_LABELER = "human"
QUALITY_FIELDS = [
    "answer_completeness",
    "safety_specificity",
    "tool_realism",
    "scope_appropriateness",
    "context_clarity",
    "tip_usefulness",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Human/LLM-judge labeling CLI: sample 4 items from each required category "
            "and collect 6 binary quality labels."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Input dataset JSONL path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="human_labels.jsonl",
        help="Output labels JSONL path (default: human_labels.jsonl).",
    )
    parser.add_argument(
        "--dataset-output",
        type=str,
        default="human_labels_dataset.jsonl",
        help="Output selected items JSONL path for judge evaluation (default: human_labels_dataset.jsonl).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible sampling.",
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=4,
        help="Number of random items to sample from each required category (default: 4).",
    )
    return parser.parse_args()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                raise ValueError(f"Line {line_number}: expected JSON object.")
            rows.append(parsed)
    return rows


def read_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        parsed = json.load(file)
    if not isinstance(parsed, list):
        raise ValueError("Expected top-level JSON array in .json dataset.")
    rows = [row for row in parsed if isinstance(row, dict)]
    return rows


def read_dataset(path: str) -> List[Dict[str, Any]]:
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        return read_json(path)
    return read_jsonl(path)


def category_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {category: 0 for category in CATEGORIES}
    for row in rows:
        category = str(row.get("category", "")).strip()
        if category in counts:
            counts[category] += 1
    return counts


def select_balanced_samples(rows: List[Dict[str, Any]], per_category: int = 4) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    counts = category_counts(rows)

    for category in CATEGORIES:
        pool = [row for row in rows if str(row.get("category", "")).strip() == category]
        if len(pool) < per_category:
            counts_text = ", ".join(f"{name}={counts[name]}" for name in CATEGORIES)
            raise ValueError(
                f"Insufficient rows for balanced sampling: category '{category}' has {len(pool)} "
                f"rows; requires at least {per_category}. Counts: {counts_text}."
            )
        selected.extend(random.sample(pool, k=per_category))

    random.shuffle(selected)
    return selected


def _prompt_binary_quality(field_name: str) -> int:
    while True:
        answer = input(
            f"Quality '{field_name}' -> enter 0 (pass) or 1 (fail), or q to quit: "
        ).strip().lower()

        if answer == "q":
            raise KeyboardInterrupt("Labeling interrupted by user.")
        if answer in ("0", "1"):
            return int(answer)

        print("Invalid input. Enter 0, 1, or q.")


def build_label_row(item: Dict[str, Any]) -> Dict[str, Any]:
    trace_id = str(item.get("id", "")).strip() or "missing_id"
    category = str(item.get("category", "")).strip() or "unknown"
    label_row: Dict[str, Any] = {
        "trace_id": trace_id,
        "category": category,
        "labeler": DEFAULT_LABELER,
    }

    for field in QUALITY_FIELDS:
        label_row[field] = _prompt_binary_quality(field)

    label_row["overall_pass"] = not any(label_row[field] == 1 for field in QUALITY_FIELDS)
    return label_row


def print_item_for_review(item: Dict[str, Any], item_index: int, total: int) -> None:
    """Display the complete item for labeling review."""
    print("\n" + "=" * 100)
    print(f"ITEM {item_index}/{total}")
    print("=" * 100)
    
    # Display key fields in order
    print(f"\n📌 ID: {item.get('id', 'N/A')}")
    print(f"📂 Category: {item.get('category', 'N/A')}")
    
    print(f"\n❓ Question:\n   {item.get('question', 'N/A')}")
    
    print(f"\n🔧 Equipment Problem:\n   {item.get('equipment_problem', 'N/A')}")
    
    print(f"\n💡 Answer:\n   {item.get('answer', 'N/A')}")
    
    print(f"\n🛠️ Tools Required:")
    tools = item.get("tools_required", [])
    if isinstance(tools, list):
        for tool in tools:
            print(f"   • {tool}")
    else:
        print(f"   {tools}")
    
    print(f"\n📋 Steps:")
    steps = item.get("steps", [])
    if isinstance(steps, list):
        for i, step in enumerate(steps, start=1):
            print(f"   {i}. {step}")
    else:
        print(f"   {steps}")
    
    print(f"\n⚠️ Safety Info:\n   {item.get('safety_info', 'N/A')}")
    
    print(f"\n✨ Tips:")
    tips = item.get("tips", [])
    if isinstance(tips, list):
        for tip in tips:
            print(f"   • {tip}")
    else:
        print(f"   {tips}")
    
    print("\n" + "-" * 100)
    print("Review the 6 qualities one at a time:")
    print("  - answer_completeness: Is the answer complete and addresses the problem?")
    print("  - safety_specificity: Are safety warnings specific and relevant?")
    print("  - tool_realism: Are tools realistic and readily available?")
    print("  - scope_appropriateness: Is the scope appropriate for DIY? Escalate when needed?")
    print("  - context_clarity: Is the problem-answer alignment clear?")
    print("  - tip_usefulness: Are tips practical and non-obvious?")
    print("-" * 100)


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write list of rows to JSONL file (overwrites if exists)."""
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    if args.per_category <= 0:
        raise SystemExit("Error: --per-category must be greater than 0.")

    if args.seed is not None:
        random.seed(args.seed)

    try:
        rows = read_dataset(args.dataset)
        selected_items = select_balanced_samples(rows, per_category=args.per_category)
    except FileNotFoundError:
        raise SystemExit(f"Error: dataset file not found: {args.dataset}")
    except (ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"Error: {error}")

    print(
        f"Loaded {len(rows)} rows. Selected {len(selected_items)} rows "
        f"({args.per_category} from each of appliance/electrical/plumbing/general_home/hvac)."
    )
    print(f"Writing labels to: {args.output}")
    Path(args.output).unlink(missing_ok=True)
    print(f"Writing selected dataset to: {args.dataset_output}")
    write_jsonl(args.dataset_output, selected_items)

    completed = 0
    try:
        for idx, item in enumerate(selected_items, start=1):
            print_item_for_review(item=item, item_index=idx, total=len(selected_items))
            label_row = build_label_row(item=item)

            print("Label output:")
            print(json.dumps(label_row, indent=2, ensure_ascii=False))
            append_jsonl(args.output, label_row)
            completed += 1
    except KeyboardInterrupt:
        print("\nStopped early by user.")

    print(f"Completed {completed}/{len(selected_items)} items.")


if __name__ == "__main__":
    main()