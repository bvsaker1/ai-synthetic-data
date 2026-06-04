import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.logging_utils import PROJECT_ROOT, get_iteration, load_env_file


def parse_args() -> argparse.Namespace:
    load_env_file()
    env_iteration = get_iteration()
    env_judge_prompt_iteration = os.getenv("JUDGE_PROMPT_ITERATION", env_iteration)

    parser = argparse.ArgumentParser(
        description=(
            "Run full iteration pipeline: dataset_generation -> data_validation -> "
            "judge_eval -> analysis (judge-only)."
        )
    )
    parser.add_argument(
        "--iteration",
        type=str,
        default=env_iteration,
        help=(
            "Target iteration for dataset, labels, logs, and outputs. "
            f"Defaults to ITERATION from environment ({env_iteration})."
        ),
    )
    parser.add_argument(
        "--judge-prompt-iteration",
        type=str,
        default=env_judge_prompt_iteration,
        help=(
            "Iteration number used only for judge prompt template selection "
            "(templates/judge_prompt_<n>.json). "
            "Defaults to JUDGE_PROMPT_ITERATION in environment, otherwise ITERATION."
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Items per template for dataset_generation (default: 10).",
    )
    parser.add_argument(
        "--judge-items-to-test",
        type=int,
        default=20,
        help="Number of items for judge_eval (default: 20).",
    )
    parser.add_argument(
        "--judge-random",
        action="store_true",
        default=True,
        help=(
            "Run judge_eval in random mode (default: enabled). "
            "With current judge_eval logic, random mode uses balanced category sampling."
        ),
    )
    parser.add_argument(
        "--no-judge-random",
        action="store_false",
        dest="judge_random",
        help="Disable random sampling for judge_eval and run sequentially.",
    )
    return parser.parse_args()


def run_step(step_name: str, command: list[str], env: dict[str, str]) -> None:
    print(f"\n[{step_name}] {' '.join(command)}")
    subprocess.run(command, check=True, env=env, cwd=str(PROJECT_ROOT))


def main() -> None:
    args = parse_args()

    if args.count <= 0:
        raise ValueError("--count must be greater than 0.")
    if args.judge_items_to_test <= 0:
        raise ValueError("--judge-items-to-test must be greater than 0.")

    iteration = args.iteration
    judge_prompt_iteration = args.judge_prompt_iteration

    raw_dataset_path = PROJECT_ROOT / "data" / f"raw_diy_dataset_{iteration}.jsonl"
    valid_dataset_path = PROJECT_ROOT / "data" / f"diy_dataset_{iteration}.jsonl"
    invalid_dataset_path = PROJECT_ROOT / "data" / f"invalid_diy_dataset_{iteration}.jsonl"
    judge_output_path = PROJECT_ROOT / "labels" / f"judge_labels_{iteration}.jsonl"
    judge_prompt_path = PROJECT_ROOT / "templates" / f"judge_prompt_{judge_prompt_iteration}.json"

    if not judge_prompt_path.exists():
        raise FileNotFoundError(
            f"Judge prompt template not found: {judge_prompt_path}. "
            "Set JUDGE_PROMPT_ITERATION or pass --judge-prompt-iteration to an existing template iteration."
        )

    python_exe = sys.executable
    env = os.environ.copy()
    env["ITERATION"] = str(iteration)

    print(
        "Running pipeline with "
        f"ITERATION={iteration}, "
        f"JUDGE_PROMPT_ITERATION={judge_prompt_iteration}, "
        "analysis mode=judge-only"
    )

    run_step(
        "dataset_generation",
        [
            python_exe,
            "-m",
            "src.dataset_generation",
            "--count",
            str(args.count),
            "--raw-output",
            str(raw_dataset_path),
        ],
        env,
    )

    run_step(
        "data_validation",
        [
            python_exe,
            "-m",
            "src.data_validation",
            "--input",
            str(raw_dataset_path),
            "--valid-output",
            str(valid_dataset_path),
            "--invalid-output",
            str(invalid_dataset_path),
        ],
        env,
    )

    run_step(
        "judge_eval",
        [
            python_exe,
            "-m",
            "src.judge_eval",
            "--iteration",
            str(iteration),
            "--dataset",
            str(valid_dataset_path),
            "--output",
            str(judge_output_path),
            "--prompt-template",
            str(judge_prompt_path),
            "--items-to-test",
            str(args.judge_items_to_test),
            *( ["--random"] if args.judge_random else [] ),
        ],
        env,
    )

    run_step(
        "analysis",
        [
            python_exe,
            "-m",
            "src.analysis",
            "--judge-only",
            "--min-iteration",
            str(iteration),
        ],
        env,
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
