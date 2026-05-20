import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from logging_utils import JsonEventLogger


QUALITY_COLUMNS = [
    "answer_completeness",
    "safety_specificity",
    "tool_realism",
    "scope_appropriateness",
    "context_clarity",
    "tip_usefulness",
]

QUALITY_AND_OVERALL_COLUMNS = QUALITY_COLUMNS + ["overall_pass"]

QUALITY_LABELS = {
    "answer_completeness": "Answer Completeness",
    "safety_specificity": "Safety Specificity",
    "tool_realism": "Tool Realism",
    "scope_appropriateness": "Scope Appropriateness",
    "context_clarity": "Context Clarity",
    "tip_usefulness": "Tip Usefulness",
    "overall_pass": "Overall Pass",
}

CATEGORY_ORDER = ["appliance", "electrical", "plumbing", "hvac", "general_home"]
CATEGORY_LABELS = {
    "appliance": "Appliance",
    "electrical": "Electrical",
    "plumbing": "Plumbing",
    "hvac": "HVAC",
    "general_home": "General Home",
}


def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise ValueError(f"Unsupported JSON top-level type in {path}: {type(data).__name__}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            clean = line.strip()
            if not clean:
                continue
            parsed = json.loads(clean)
            if not isinstance(parsed, dict):
                raise ValueError(f"{path}:{line_number} expected JSON object")
            rows.append(parsed)
    return rows


def load_human_labels(root: Path) -> pd.DataFrame:
    # Prefer JSONL, but support JSON if user later renames format.
    candidates = [root / "human_labels.jsonl", root / "human_labels.json"]
    label_path = next((p for p in candidates if p.exists()), None)
    if label_path is None:
        raise FileNotFoundError("Could not find human labels file (human_labels.jsonl or human_labels.json)")

    rows = read_json_or_jsonl(label_path)
    df = pd.DataFrame(rows)
    return normalize_label_df(df, labeler="human")


def discover_judge_label_files(root: Path, min_iteration: int = 3) -> List[Tuple[int, Path]]:
    files: List[Tuple[int, Path]] = []
    pattern = re.compile(r"judge_labels_(\d+)\.jsonl$")

    for file_path in root.glob("judge_labels_*.jsonl"):
        match = pattern.search(file_path.name)
        if not match:
            continue
        iteration = int(match.group(1))
        if iteration >= min_iteration:
            files.append((iteration, file_path))

    return sorted(files, key=lambda x: x[0])


def normalize_label_df(df: pd.DataFrame, labeler: str) -> pd.DataFrame:
    # Accept either trace_id or id and normalize to trace_id.
    if "trace_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "trace_id"})

    required = ["trace_id", "category"] + QUALITY_COLUMNS + ["overall_pass"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {labeler} labels: {missing}")

    normalized = df[required].copy()
    normalized["labeler"] = labeler

    for quality in QUALITY_COLUMNS:
        normalized[quality] = pd.to_numeric(normalized[quality], errors="coerce").fillna(1).astype(int)
        normalized[quality] = normalized[quality].clip(lower=0, upper=1)

    normalized["overall_pass"] = normalized["overall_pass"].astype(bool)
    normalized["category"] = normalized["category"].astype(str).str.lower()
    normalized["trace_id"] = normalized["trace_id"].astype(str)
    return normalized


def _analysis_log_path(root: Path, iteration: int) -> Path:
    return root / "logs" / f"dataset_log_{iteration}.jsonl"


def log_analysis_run_start(
    root: Path, iteration: int, human_file: Path, judge_file: Path, data_dir: Path, output_dir: Path, min_iteration: int
) -> JsonEventLogger:
    logger = JsonEventLogger(
        log_path=str(_analysis_log_path(root, iteration)),
        script_name="analysis",
        model="analysis",
        iteration=str(iteration),
    )
    startup_message = (
        "Analysis run started | "
        f"iteration={iteration} | "
        f"data_dir={data_dir} | output_dir={output_dir} | min_iteration={min_iteration} (default=3)"
    )
    logger.log(
        {
            "event": "analysis_start",
            "message": startup_message,
        }
    )
    return logger


def log_label_differences(
    logger: JsonEventLogger,
    iteration: int,
    human_df: pd.DataFrame,
    judge_df: pd.DataFrame,
) -> None:
    merged = human_df.merge(judge_df, on="trace_id", suffixes=("_human", "_judge"))
    for _, row in merged.iterrows():
        trace_id = str(row["trace_id"])
        category = str(row["category_human"] if "category_human" in row else row["category_judge"])

        mismatched_qualities: List[Dict[str, Any]] = []

        for quality in QUALITY_COLUMNS:
            human_score = int(row[f"{quality}_human"])
            judge_score = int(row[f"{quality}_judge"])
            if human_score != judge_score:
                mismatched_qualities.append(
                    {
                        "quality": QUALITY_LABELS[quality],
                        "human_score": human_score,
                        "judge_score": judge_score,
                    }
                )

        human_overall = bool(row["overall_pass_human"])
        judge_overall = bool(row["overall_pass_judge"])

        overall_pass_diff = human_overall != judge_overall
        if overall_pass_diff:
            mismatched_qualities.append(
                {
                    "quality": QUALITY_LABELS["overall_pass"],
                    "human_score": human_overall,
                    "judge_score": judge_overall,
                    "overall_pass_diff": True,
                }
            )

        if mismatched_qualities:
            logger.log(
                {
                    "event": "analysis_label_difference",
                    "iteration": iteration,
                    "trace_id": trace_id,
                    "category": category,
                    "mismatched_qualities": mismatched_qualities,
                    "mismatch_count": len(mismatched_qualities),
                    "overall_pass_diff": overall_pass_diff,
                }
            )


def quality_pass_rate_percent(df: pd.DataFrame) -> Dict[str, float]:
    rates: Dict[str, float] = {}
    # Convention for quality columns: 0 = pass, 1 = fail.
    for quality in QUALITY_COLUMNS:
        rates[quality] = float((df[quality] == 0).mean() * 100.0)
    # overall_pass is already a pass boolean.
    rates["overall_pass"] = float(df["overall_pass"].astype(bool).mean() * 100.0)
    return rates


def quality_fail_rate_percent_by_category(df: pd.DataFrame) -> pd.DataFrame:
    working = df[df["category"].isin(CATEGORY_ORDER)].copy()
    grouped = (
        working.groupby("category", dropna=False)[QUALITY_COLUMNS]
        .mean()
        .reindex(CATEGORY_ORDER)
        .fillna(0.0)
    )
    # overall_pass is pass-rate; convert to fail-rate for heatmap consistency.
    grouped["overall_pass"] = (
        1.0 - working.groupby("category", dropna=False)["overall_pass"].mean().reindex(CATEGORY_ORDER).fillna(1.0)
    )
    # Quality fields are fail-rate by mean (1=fail), now plus overall fail-rate.
    return grouped * 100.0


def save_pass_rate_chart(human_df: pd.DataFrame, judge_df: pd.DataFrame, iteration: int, out_dir: Path) -> None:
    human_rates = quality_pass_rate_percent(human_df)
    judge_rates = quality_pass_rate_percent(judge_df)

    chart_df = pd.DataFrame(
        {
            "Quality": [QUALITY_LABELS[q] for q in QUALITY_AND_OVERALL_COLUMNS] * 2,
            "Pass Rate (%)": [human_rates[q] for q in QUALITY_AND_OVERALL_COLUMNS]
            + [judge_rates[q] for q in QUALITY_AND_OVERALL_COLUMNS],
            "Source": ["Human"] * len(QUALITY_AND_OVERALL_COLUMNS) + ["Judge"] * len(QUALITY_AND_OVERALL_COLUMNS),
        }
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(data=chart_df, x="Quality", y="Pass Rate (%)", hue="Source")
    plt.ylim(0, 100)
    plt.title(f"Pass Rate for All Qualities (Iteration {iteration})")
    plt.ylabel("Percent")
    plt.xlabel("Qualities")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / f"pass_rate_all_qualities_iter_{iteration}.png", dpi=160)
    plt.close()


def save_heatmap_chart(judge_df: pd.DataFrame, iteration: int, out_dir: Path) -> None:
    heatmap_data = quality_fail_rate_percent_by_category(judge_df)
    heatmap_data = heatmap_data.rename(columns=QUALITY_LABELS, index=CATEGORY_LABELS)

    plt.figure(figsize=(12, 5.5))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="Reds",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Fail Rate (%)"},
    )
    plt.title(f"Heatmap of Failed Quality by Category - Iteration {iteration}")
    plt.ylabel("Categories")
    plt.xlabel("Qualities")
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0, va="center")
    plt.tight_layout()
    plt.savefig(out_dir / f"failed_quality_heatmap_iter_{iteration}.png", dpi=160)
    plt.close()


def save_agreement_chart(human_df: pd.DataFrame, judge_df: pd.DataFrame, iteration: int, out_dir: Path) -> None:
    merged = human_df.merge(judge_df, on="trace_id", suffixes=("_human", "_judge"))
    agreement = []
    for quality in QUALITY_AND_OVERALL_COLUMNS:
        value = float((merged[f"{quality}_human"] == merged[f"{quality}_judge"]).mean() * 100.0)
        agreement.append(value)

    chart_df = pd.DataFrame(
        {
            "Quality": [QUALITY_LABELS[q] for q in QUALITY_AND_OVERALL_COLUMNS],
            "Agreement (%)": agreement,
        }
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(data=chart_df, x="Quality", y="Agreement (%)", color="#1f77b4")
    plt.ylim(0, 100)
    plt.title(f"Human vs LLM as a Judge Agreement (Iteration {iteration})")
    plt.ylabel("Percent")
    plt.xlabel("Qualities")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / f"human_vs_llm_agreement_iter_{iteration}.png", dpi=160)
    plt.close()


def save_agreement_by_iteration_chart(
    human_df: pd.DataFrame,
    judge_by_iteration: Dict[int, pd.DataFrame],
    out_dir: Path,
) -> None:
    iterations = sorted(judge_by_iteration.keys())
    if not iterations:
        return

    rows: List[Dict[str, Any]] = []
    for iteration in iterations:
        judge_df = judge_by_iteration[iteration]
        merged = human_df.merge(judge_df, on="trace_id", suffixes=("_human", "_judge"))
        for quality in QUALITY_AND_OVERALL_COLUMNS:
            agreement_pct = float((merged[f"{quality}_human"] == merged[f"{quality}_judge"]).mean() * 100.0)
            rows.append(
                {
                    "Iteration": iteration,
                    "Quality": QUALITY_LABELS[quality],
                    "Agreement (%)": agreement_pct,
                }
            )

    chart_df = pd.DataFrame(rows)
    plt.figure(figsize=(12, 6.5))
    sns.lineplot(
        data=chart_df,
        x="Iteration",
        y="Agreement (%)",
        hue="Quality",
        marker="o",
        linewidth=2,
    )
    plt.ylim(0, 100)
    plt.title("Human vs LLM as a Judge Agreement by Iteration")
    plt.ylabel("Percent")
    plt.xlabel("Iteration")
    plt.xticks(iterations)
    plt.tight_layout()
    plt.savefig(out_dir / "human_vs_llm_agreement_by_iteration.png", dpi=160)
    plt.close()


def save_category_distribution_chart(dataset_df: pd.DataFrame, out_dir: Path) -> None:
    working = dataset_df.copy()
    if "category" not in working.columns:
        raise ValueError("Dataset file must include category column for distribution chart")

    working["category"] = working["category"].astype(str).str.lower()
    actual = (
        working[working["category"].isin(CATEGORY_ORDER)]
        .groupby("category")
        .size()
        .reindex(CATEGORY_ORDER)
        .fillna(0)
    )

    actual_pct = (actual / max(1, int(actual.sum()))) * 100.0
    benchmark_pct = pd.Series(100.0 / len(CATEGORY_ORDER), index=CATEGORY_ORDER)

    x = np.arange(len(CATEGORY_ORDER))
    width = 0.38

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, actual_pct.values, width=width, label="Actual")
    plt.bar(x + width / 2, benchmark_pct.values, width=width, label="Benchmark")
    plt.xticks(x, [CATEGORY_LABELS[c] for c in CATEGORY_ORDER], rotation=20, ha="right")
    plt.ylim(0, 100)
    plt.title("Category Distribution")
    plt.ylabel("Percent of Total Categories")
    plt.xlabel("Category Names")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "category_distribution_vs_benchmark.png", dpi=160)
    plt.close()


def save_dataset_quality_by_iteration_chart(
    human_df: pd.DataFrame,
    judge_by_iteration: Dict[int, pd.DataFrame],
    out_dir: Path,
) -> None:
    iterations = sorted(judge_by_iteration.keys())
    if not iterations:
        return

    human_rates = quality_pass_rate_percent(human_df)

    # Build a chart where each quality has grouped bars for each iteration,
    # and each iteration has a paired Human/Judge bar.
    n_qualities = len(QUALITY_AND_OVERALL_COLUMNS)
    n_iterations = len(iterations)
    pair_width = 0.08
    quality_gap = 0.75

    # Base x location for each quality cluster.
    base_positions = np.arange(n_qualities) * (n_iterations * 2 * pair_width + quality_gap)

    plt.figure(figsize=(max(12, n_iterations * 3), 6.5))

    for i, iteration in enumerate(iterations):
        judge_rates = quality_pass_rate_percent(judge_by_iteration[iteration])

        human_values = [human_rates[q] for q in QUALITY_AND_OVERALL_COLUMNS]
        judge_values = [judge_rates[q] for q in QUALITY_AND_OVERALL_COLUMNS]

        iter_offset = (i - (n_iterations - 1) / 2.0) * (2 * pair_width + 0.03)

        x_human = base_positions + iter_offset - pair_width / 2
        x_judge = base_positions + iter_offset + pair_width / 2

        plt.bar(
            x_human,
            human_values,
            width=pair_width,
            label=f"Iter {iteration} Human" if i == 0 else None,
            alpha=0.65,
            color="#6baed6",
        )
        plt.bar(
            x_judge,
            judge_values,
            width=pair_width,
            label=f"Iter {iteration} Judge" if i == 0 else None,
            alpha=0.85,
            color="#2171b5",
        )

        # Iteration annotations above the first quality cluster for readability.
        plt.text(
            x_judge[0],
            102,
            f"I{iteration}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )

    plt.xticks(base_positions, [QUALITY_LABELS[q] for q in QUALITY_AND_OVERALL_COLUMNS], rotation=20, ha="right")
    plt.ylim(0, 105)
    plt.title("Dataset Quality Pass Rate by Iteration")
    plt.ylabel("Percent of pass rate")
    plt.xlabel("Grouped by iteration for each quality")
    plt.legend(["Human", "Judge"], loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / "dataset_quality_pass_rate_by_iteration.png", dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run data analysis and generate visualizations comparing human vs LLM judge labels on home-repair QA items."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing human_labels.jsonl and judge_labels_*.jsonl files (default: script directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save visualization PNG files (default: script directory/visualizations)",
    )
    parser.add_argument(
        "--min-iteration",
        type=int,
        default=3,
        help="Minimum judge iteration number to analyze (default: 3)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid")

    root = args.data_dir if args.data_dir else Path(__file__).resolve().parent
    out_dir = args.output_dir if args.output_dir else root / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    human_df = load_human_labels(root)

    judge_files = discover_judge_label_files(root, min_iteration=args.min_iteration)
    if not judge_files:
        raise FileNotFoundError(f"No judge_labels_<iteration>.jsonl files found for iterations >= {args.min_iteration}")

    judge_by_iteration: Dict[int, pd.DataFrame] = {}
    human_label_path = root / "human_labels.jsonl"
    if not human_label_path.exists():
        human_label_path = root / "human_labels.json"

    for iteration, file_path in judge_files:
        rows = read_json_or_jsonl(file_path)
        judge_df = normalize_label_df(pd.DataFrame(rows), labeler="judge")
        judge_by_iteration[iteration] = judge_df

        analysis_logger = log_analysis_run_start(root, iteration, human_label_path, file_path, root, out_dir, args.min_iteration)
        log_label_differences(analysis_logger, iteration, human_df, judge_df)

        save_pass_rate_chart(human_df, judge_df, iteration, out_dir)
        save_heatmap_chart(judge_df, iteration, out_dir)
        save_agreement_chart(human_df, judge_df, iteration, out_dir)

    dataset_path = root / "human_labels_dataset.jsonl"
    dataset_rows = read_json_or_jsonl(dataset_path)
    dataset_df = pd.DataFrame(dataset_rows)
    save_category_distribution_chart(dataset_df, out_dir)

    save_dataset_quality_by_iteration_chart(human_df, judge_by_iteration, out_dir)
    save_agreement_by_iteration_chart(human_df, judge_by_iteration, out_dir)

    generated_files = sorted(out_dir.glob("*.png"))
    print("Generated visualizations:")
    for file_path in generated_files:
        print(f"- {file_path.name}")


if __name__ == "__main__":
    main()
