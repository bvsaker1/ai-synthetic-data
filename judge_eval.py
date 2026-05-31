import argparse
import json
import os
import random
import re
import time
from collections import deque
from typing import Any, Deque, Dict, List, Tuple

from groq import Groq
from logging_utils import (
    JsonEventLogger,
    get_iteration,
    load_env_file,
)


# Evaluation Criteria for Home Repair QA Items (Q1-Q8)
# Q1 Answer Coherence:
#    The answer must read as a complete, natural response, not a disjointed list.
#    It should integrate tools, steps, safety, and tips into a narrative.
# Q2 Step Actionability:
#    Steps must be specific and executable for beginners, with concrete detail
#    (observable outcomes, quantities, thresholds) where relevant.
# Q3 Tool Realism:
#    Tools should be realistic homeowner tools commonly available in general
#    hardware stores, typically under $50 each.
# Q4 Safety Specificity:
#    safety_info must state a specific hazard and specific precaution for this
#    task. Generic warnings fail. safety_info must be at least 80 characters.
# Q5 Tip Usefulness:
#    Tips must be non-obvious, task-specific, and provide value beyond the steps.
# Q6 Problem-Answer Alignment:
#    The answer must directly address the exact equipment_problem.
# Q7 Appropriate Scope:
#    Guidance should stay within realistic DIY scope and escalate clearly when
#    professional help is needed.
# Q8 Category Accuracy:
#    The category must correctly match the repair domain.


# DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")
# DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-20b")
DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")
DEFAULT_MAX_TPM = int(os.getenv("JUDGE_MAX_TPM", "8000"))
DEFAULT_MAX_COMPLETION_TOKENS = int(os.getenv("JUDGE_MAX_COMPLETION_TOKENS", "800"))
DEFAULT_MAX_RETRIES = int(os.getenv("JUDGE_MAX_RETRIES", "5"))
DEFAULT_TPM_WINDOW_SECONDS = 60.0
MAX_ITEM_TEXT_CHARS = int(os.getenv("JUDGE_MAX_ITEM_TEXT_CHARS", "1400"))
MAX_ITEM_LIST_LENGTH = int(os.getenv("JUDGE_MAX_ITEM_LIST_LENGTH", "8"))
MAX_ITEM_LIST_ENTRY_CHARS = int(os.getenv("JUDGE_MAX_ITEM_LIST_ENTRY_CHARS", "180"))


def build_iteration_path(prefix: str, iteration: str, extension: str = ".jsonl") -> str:
    return f"{prefix}_{iteration}{extension}"


def load_judge_prompt_template(prompt_path: str) -> Dict[str, str]:
    """Load judge prompt template from JSON file.
    
    Args:
        prompt_path: Path to .json file containing prompt template.
    
    Returns:
        Dict with 'system_prompt' and 'user_prompt_template' keys.
    
    Raises:
        ValueError: If file format is invalid or required keys are missing.
    """
    with open(prompt_path, "r", encoding="utf-8") as file:
        template = json.load(file)

    if not isinstance(template, dict):
        raise ValueError(f"Prompt template must be a JSON object, not {type(template).__name__}.")

    system_prompt = str(template.get("system_prompt", "")).strip()
    raw_template = template.get("user_prompt_template", "")
    if isinstance(raw_template, list):
        user_prompt_template = "\n".join(raw_template).strip()
    else:
        user_prompt_template = str(raw_template).strip()

    if not system_prompt:
        raise ValueError("Prompt template missing non-empty 'system_prompt'.")
    if not user_prompt_template:
        raise ValueError("Prompt template missing non-empty 'user_prompt_template'.")
    if "__ITEM_JSON__" not in user_prompt_template:
        raise ValueError("user_prompt_template must include the __ITEM_JSON__ placeholder.")

    return {
        "system_prompt": system_prompt,
        "user_prompt_template": user_prompt_template,
    }


def parse_args() -> argparse.Namespace:
    load_env_file()
    parser = argparse.ArgumentParser(description="LLM judge for home-repair QA datasets.")
    parser.add_argument(
        "--iteration",
        type=str,
        default=get_iteration(),
        help="Iteration suffix used for default dataset, output, log, and prompt-template paths.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Input dataset path (.json or .jsonl). Defaults to diy_dataset_<iteration>.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output labels JSONL path. Defaults to judge_labels_<iteration>.jsonl.",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=None,
        help="Judge prompt template JSON path. Defaults to judge_prompt_<iteration>.json.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model name. Default: {DEFAULT_JUDGE_MODEL}",
    )
    parser.add_argument(
        "--max-tpm",
        type=int,
        default=DEFAULT_MAX_TPM,
        help=(
            "Soft token-per-minute cap for local throttling. "
            f"Default: {DEFAULT_MAX_TPM}"
        ),
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_MAX_COMPLETION_TOKENS,
        help=(
            "Max completion tokens per judge response to keep per-request token "
            f"usage bounded. Default: {DEFAULT_MAX_COMPLETION_TOKENS}"
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Max retries for transient 429/runtime API errors. Default: {DEFAULT_MAX_RETRIES}",
    )
    parser.add_argument(
        "--items-to-test",
        type=int,
        default=None,
        help=(
            "Number of dataset items to evaluate. In sequential mode, omitted means all items. "
            "In random mode, omitted means one random item."
        ),
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomly sample items instead of evaluating sequentially from the top of the dataset.",
    )
    args = parser.parse_args()
    args.dataset = args.dataset or build_iteration_path("diy_dataset", args.iteration)
    args.output = args.output or build_iteration_path("judge_labels", args.iteration)
    args.prompt_template = args.prompt_template or build_iteration_path("judge_prompt", args.iteration, extension=".json")
    return args


def _quality_key_to_label() -> Dict[str, str]:
    return {
        "Q1": "answer_coherence",
        "Q2": "step_actionability",
        "Q3": "tool_realism",
        "Q4": "safety_specificity",
        "Q5": "tip_usefulness",
        "Q6": "problem_answer_alignment",
        "Q7": "appropriate_scope",
        "Q8": "category_accuracy",
    }


def _quality_key_to_human_field() -> Dict[str, str]:
    return {
        "Q1": "answer_completeness",
        "Q2": "safety_specificity",
        "Q3": "tool_realism",
        "Q4": "scope_appropriateness",
        "Q5": "context_clarity",
        "Q6": "tip_usefulness",
    }


def build_human_compatible_label_row(
    trace_id: str,
    category: str,
    failed_qualities: List[Dict[str, Any]],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "trace_id": trace_id,
        "category": category,
        "labeler": "judge",
        "answer_completeness": 0,
        "safety_specificity": 0,
        "tool_realism": 0,
        "scope_appropriateness": 0,
        "context_clarity": 0,
        "tip_usefulness": 0,
        "overall_pass": True,
    }

    field_map = _quality_key_to_human_field()
    has_runtime_error = False

    for failure in failed_qualities:
        if not isinstance(failure, dict):
            continue
        quality_text = str(failure.get("quality", "")).strip().upper()
        if quality_text == "JUDGE_RUNTIME_ERROR":
            has_runtime_error = True
            continue
        q_key = quality_text.split(" ", 1)[0]
        mapped = field_map.get(q_key)
        if mapped:
            row[mapped] = 1

    if has_runtime_error:
        for field_name in field_map.values():
            row[field_name] = 1

    row["overall_pass"] = all(row[field_name] == 0 for field_name in field_map.values())
    return row


def build_judge_scoring(trace_id: str, failed_qualities: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Top-level flags follow 0=pass, 1=fail convention.
    scoring: Dict[str, Any] = {
        "trace_id": trace_id,
        "incomplete_answer": 0,
        "safety_violations": 0,
        "unrealistic_tools": 0,
        "overcomplicated_solution": 0,
        "missing_context": 0,
        "poor_quality_tips": 0,
        "overall_failure": False,
        "quality_scores": {
            "answer_coherence": 1,
            "step_actionability": 1,
            "tool_realism": 1,
            "safety_specificity": 1,
            "tip_usefulness": 1,
            "problem_answer_alignment": 1,
            "appropriate_scope": 1,
            "category_accuracy": 1,
        },
        "quality_pass": True,
    }

    quality_map = _quality_key_to_label()
    failed_labels: set[str] = set()
    for failure in failed_qualities:
        quality_text = str(failure.get("quality", "")).strip().upper()
        short_key = quality_text.split(" ", 1)[0]
        label = quality_map.get(short_key)
        if label:
            failed_labels.add(label)

    for label in failed_labels:
        scoring["quality_scores"][label] = 0

    # Map detailed failures into coarse flags.
    if "answer_coherence" in failed_labels or "step_actionability" in failed_labels:
        scoring["incomplete_answer"] = 1
    if "safety_specificity" in failed_labels:
        scoring["safety_violations"] = 1
    if "tool_realism" in failed_labels:
        scoring["unrealistic_tools"] = 1
    if "appropriate_scope" in failed_labels:
        scoring["overcomplicated_solution"] = 1
    if "problem_answer_alignment" in failed_labels or "category_accuracy" in failed_labels:
        scoring["missing_context"] = 1
    if "tip_usefulness" in failed_labels:
        scoring["poor_quality_tips"] = 1

    has_failure = bool(failed_labels)
    scoring["overall_failure"] = has_failure
    scoring["quality_pass"] = not has_failure
    return scoring


def _estimate_tokens(text: str) -> int:
    # Rough but safe estimate for planning TPM budget.
    return max(1, len(text) // 4)


def _truncate_text(value: Any, max_chars: int) -> str:
    text = str(value).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _compact_item_for_judging(item: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {
        "id": _truncate_text(item.get("id", ""), 120),
        "category": _truncate_text(item.get("category", ""), 120),
        "question": _truncate_text(item.get("question", ""), MAX_ITEM_TEXT_CHARS),
        "equipment_problem": _truncate_text(item.get("equipment_problem", ""), MAX_ITEM_TEXT_CHARS),
        "answer": _truncate_text(item.get("answer", ""), MAX_ITEM_TEXT_CHARS),
        "safety_info": _truncate_text(item.get("safety_info", ""), MAX_ITEM_TEXT_CHARS),
    }

    for key in ("tools_required", "steps", "tips"):
        raw = item.get(key, [])
        values = raw if isinstance(raw, list) else []
        compact[key] = [
            _truncate_text(entry, MAX_ITEM_LIST_ENTRY_CHARS)
            for entry in values[:MAX_ITEM_LIST_LENGTH]
            if str(entry).strip()
        ]

    return compact


def _parse_retry_after_seconds(error_text: str) -> float:
    # Groq errors often include: "Please try again in 6.165s" or "262.5ms".
    match = re.search(r"Please try again in\s+([0-9]*\.?[0-9]+)\s*(ms|s)", error_text)
    if not match:
        return 0.0
    value = float(match.group(1))
    unit = match.group(2)
    return value / 1000.0 if unit == "ms" else value


class TokenRateLimiter:
    def __init__(self, max_tokens_per_window: int, window_seconds: float = DEFAULT_TPM_WINDOW_SECONDS) -> None:
        if max_tokens_per_window <= 0:
            raise ValueError("max_tokens_per_window must be > 0")
        self.max_tokens_per_window = max_tokens_per_window
        self.window_seconds = window_seconds
        self._events: Deque[Tuple[float, int]] = deque()
        self._tokens_in_window = 0

    def _evict_old(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] <= cutoff:
            _, old_tokens = self._events.popleft()
            self._tokens_in_window -= old_tokens

    def reserve(self, estimated_tokens: int) -> None:
        tokens = max(1, estimated_tokens)
        while True:
            now = time.monotonic()
            self._evict_old(now)

            if self._tokens_in_window + tokens <= self.max_tokens_per_window:
                self._events.append((now, tokens))
                self._tokens_in_window += tokens
                return

            oldest_ts, _ = self._events[0]
            sleep_for = max(0.01, (oldest_ts + self.window_seconds) - now)
            time.sleep(sleep_for)


def select_items(
    items: List[Dict[str, Any]],
    items_to_test: int | None,
    random_mode: bool,
) -> List[Dict[str, Any]]:
    if not items:
        return []

    if items_to_test is not None and items_to_test <= 0:
        raise ValueError("--items-to-test must be greater than 0 when provided.")

    if random_mode:
        sample_size = 1 if items_to_test is None else min(items_to_test, len(items))
        return random.sample(items, k=sample_size)

    if items_to_test is None:
        return items

    return items[:items_to_test]


def read_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    if dataset_path.endswith(".json"):
        with open(dataset_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError("Expected top-level JSON array in dataset file.")
        return data

    items: List[Dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            clean = line.strip()
            if not clean:
                continue
            parsed = json.loads(clean)
            if not isinstance(parsed, dict):
                raise ValueError(f"Line {line_number}: expected JSON object.")
            items.append(parsed)
    return items


def _normalize_judge_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    resolved_label = raw.get("resolved_label")
    status = raw.get("status")
    overall_pass = raw.get("overall_pass")
    unresolved_reason = str(raw.get("unresolved_reason", "")).strip()
    failed_qualities = raw.get("failed_qualities", [])
    judge_summary = raw.get("judge_summary", "")

    if resolved_label not in (0, 1):
        if isinstance(overall_pass, bool):
            resolved_label = 0 if overall_pass else 1
        elif isinstance(overall_pass, int) and overall_pass in (0, 1):
            resolved_label = 0 if overall_pass == 1 else 1
        elif isinstance(overall_pass, str):
            lowered = overall_pass.strip().lower()
            if lowered in ("true", "1", "yes", "pass", "passed", "resolved"):
                resolved_label = 0
            elif lowered in ("false", "0", "no", "fail", "failed", "unresolved"):
                resolved_label = 1

    if resolved_label not in (0, 1):
        if isinstance(status, str) and status.upper() == "RESOLVED":
            resolved_label = 0
        elif isinstance(status, str) and status.upper() == "UNRESOLVED":
            resolved_label = 1
        else:
            resolved_label = 1

    status = "RESOLVED" if resolved_label == 0 else "UNRESOLVED"

    if not isinstance(failed_qualities, list):
        failed_qualities = []

    clean_failures: List[Dict[str, str]] = []
    for failure in failed_qualities:
        if not isinstance(failure, dict):
            continue
        quality = str(failure.get("quality", "")).strip()
        why = str(failure.get("why", "")).strip()
        if quality and why:
            clean_failures.append({"quality": quality, "why": why})

    if resolved_label == 0:
        clean_failures = []
        unresolved_reason = ""
    elif not clean_failures:
        clean_failures = [
            {
                "quality": "Q-unknown",
                "why": "Judge marked UNRESOLVED but did not provide structured failures.",
            }
        ]

    if resolved_label == 1 and not unresolved_reason:
        if clean_failures:
            unresolved_reason = "; ".join(
                f"{entry['quality']}: {entry['why']}" for entry in clean_failures
            )
        else:
            unresolved_reason = "One or more quality checks failed."

    overall_pass = resolved_label == 0

    return {
        "resolved_label": resolved_label,
        "status": status,
        "overall_pass": overall_pass,
        "unresolved_reason": unresolved_reason,
        "failed_qualities": clean_failures,
        "judge_summary": str(judge_summary),
    }


def judge_one_item(
    client: Groq,
    model: str,
    item: Dict[str, Any],
    limiter: TokenRateLimiter,
    max_completion_tokens: int,
    max_retries: int,
    prompt_template: Dict[str, str],
) -> Dict[str, Any]:
    compact_item = _compact_item_for_judging(item)
    item_json = json.dumps(compact_item, ensure_ascii=False, separators=(",", ":"))
    system_prompt = prompt_template["system_prompt"]
    user_prompt = prompt_template["user_prompt_template"].replace("__ITEM_JSON__", item_json)

    estimated_tokens = (
        _estimate_tokens(system_prompt)
        + _estimate_tokens(user_prompt)
        + max(1, max_completion_tokens)
    )

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        limiter.reserve(estimated_tokens)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                # Prefer deterministic output to reduce invalid JSON responses.
                temperature=0.3,
                max_completion_tokens=max_completion_tokens,
                response_format={"type": "json_object"},
            )
            break
        except Exception as error:
            last_error = error
            error_text = str(error)
            is_rate_limit = "rate_limit" in error_text.lower() or "429" in error_text
            if not is_rate_limit or attempt >= max_retries:
                raise

            retry_after = _parse_retry_after_seconds(error_text)
            backoff = min(12.0, 0.5 * (2**attempt))
            jitter = random.uniform(0.05, 0.35)
            sleep_for = max(retry_after, backoff) + jitter
            print(
                f"Rate limited on attempt {attempt + 1}/{max_retries + 1}; "
                f"sleeping {sleep_for:.2f}s before retry..."
            )
            time.sleep(sleep_for)
    else:
        if last_error is None:
            raise RuntimeError("Unknown judge failure.")
        raise last_error

    content = response.choices[0].message.content or "{}"
    raw = json.loads(content)
    if not isinstance(raw, dict):
        raise ValueError("Judge response must be a JSON object.")

    return _normalize_judge_output(raw)


def run_judge(
    dataset_path: str,
    output_path: str,
    prompt_template_path: str,
    model: str,
    max_tpm: int,
    max_completion_tokens: int,
    max_retries: int,
    items_to_test: int | None,
    random_mode: bool,
    iteration: str,
) -> None:
    load_env_file()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to .env or environment.")

    prompt_template = load_judge_prompt_template(prompt_template_path)
    all_items = read_dataset(dataset_path)
    items = select_items(all_items, items_to_test=items_to_test, random_mode=random_mode)
    client = Groq(api_key=api_key)
    limiter = TokenRateLimiter(max_tokens_per_window=max_tpm)
    runtime_error_count = 0
    log_path = build_iteration_path("logs/dataset_log", iteration)
    logger = JsonEventLogger(log_path=log_path, script_name="judge_eval", model=model, iteration=iteration)

    startup_message = (
        "Judge eval run started | "
        f"iteration={iteration} | "
        f"dataset={dataset_path} | output={output_path} | prompt_template={prompt_template_path} | "
        f"model={model} (default={DEFAULT_JUDGE_MODEL}) | "
        f"max_tpm={max_tpm} (default={DEFAULT_MAX_TPM}) | "
        f"max_completion_tokens={max_completion_tokens} (default={DEFAULT_MAX_COMPLETION_TOKENS}) | "
        f"max_retries={max_retries} (default={DEFAULT_MAX_RETRIES}) | "
        f"items_to_test={items_to_test} | random={random_mode} | "
        f"selection={'random' if random_mode else 'sequential'} | testing {len(items)}/{len(all_items)} items"
    )
    logger.log({"event": "judge_eval_start", "message": startup_message})
    print(startup_message)

    with open(output_path, "w", encoding="utf-8") as out:
        for index, item in enumerate(items, start=1):
            item_id = item.get("id", f"row_{index:05d}")
            category = item.get("category", "unknown")

            judgment = None
            last_error = None
            
            # Retry loop for handling runtime errors
            for attempt in range(max_retries + 1):
                try:
                    judgment = judge_one_item(
                        client=client,
                        model=model,
                        item=item,
                        limiter=limiter,
                        max_completion_tokens=max_completion_tokens,
                        max_retries=max_retries,
                        prompt_template=prompt_template,
                    )
                    break  # Success, exit retry loop
                except Exception as error:
                    last_error = error
                    if attempt < max_retries:
                        print(f"  {item_id}: Attempt {attempt + 1} failed, retrying...")
                        time.sleep(1)  # Brief delay before retry
                        continue
                    else:
                        # All retries exhausted
                        judgment = {
                            "resolved_label": 1,
                            "status": "UNRESOLVED",
                            "unresolved_reason": str(error),
                            "failed_qualities": [
                                {
                                    "quality": "JUDGE_RUNTIME_ERROR",
                                    "why": str(error),
                                }
                            ],
                            "judge_summary": "Judge failed to evaluate this item due to runtime/API/parsing error.",
                        }
                        logger.log(
                            {
                                "event": "runtime_error",
                                "trace_id": item_id,
                                "error": str(error),
                                "structural_validation": "failed",
                                "prompt_template": prompt_template_path,
                                "judge_scoring": {},
                            }
                        )

            # Final safety net: every UNRESOLVED row must include a non-empty reason.
            resolved_label = judgment.get("resolved_label", 1)
            status = judgment.get("status", "UNRESOLVED")
            unresolved_reason = str(judgment.get("unresolved_reason", "")).strip()
            failed_qualities = judgment.get("failed_qualities", [])
            judge_summary = str(judgment.get("judge_summary", "")).strip()

            # Print status to stdout
            status_display = "RESOLVED" if resolved_label == 0 else "UNRESOLVED"
            print(f"  {item_id}: {status_display}")

            if resolved_label == 1 and not unresolved_reason:
                if isinstance(failed_qualities, list) and failed_qualities:
                    reasons: List[str] = []
                    for failure in failed_qualities:
                        if not isinstance(failure, dict):
                            continue
                        quality = str(failure.get("quality", "")).strip()
                        why = str(failure.get("why", "")).strip()
                        if quality and why:
                            reasons.append(f"{quality}: {why}")
                    if reasons:
                        unresolved_reason = "; ".join(reasons)
                if not unresolved_reason:
                    unresolved_reason = (
                        judge_summary
                        or "Marked UNRESOLVED but no detailed reason was provided by the judge model."
                    )

            if resolved_label == 0:
                unresolved_reason = ""

            clean_failures = failed_qualities if isinstance(failed_qualities, list) else []
            label_row = build_human_compatible_label_row(
                trace_id=str(item_id),
                category=str(category),
                failed_qualities=clean_failures,
            )

            out.write(json.dumps(label_row, ensure_ascii=False) + "\n")
            judge_scoring = build_judge_scoring(
                trace_id=str(item_id),
                failed_qualities=clean_failures,
            )
            structural_validation = "failed" if any(
                isinstance(failure, dict)
                and str(failure.get("quality", "")).strip() == "JUDGE_RUNTIME_ERROR"
                for failure in clean_failures
            ) else "passed"

            logger.log(
                {
                    "event": "judge_item_result",
                    "message": f"Judged {index}/{len(items)}",
                    "id": item_id,
                    "category": category,
                    "structural_validation": structural_validation,
                    "overall_pass": label_row["overall_pass"],
                    "failed_qualities": clean_failures if not label_row["overall_pass"] else [],
                    "unresolved_reason": unresolved_reason if not label_row["overall_pass"] else "",
                }
            )

            has_runtime_error = any(
                isinstance(failure, dict)
                and str(failure.get("quality", "")).strip() == "JUDGE_RUNTIME_ERROR"
                for failure in clean_failures
            )
            if has_runtime_error:
                runtime_error_count += 1
                if runtime_error_count >= 3:
                    logger.print_and_log(
                        "Stopping early after 3 JUDGE_RUNTIME_ERROR results to avoid wasting API calls.",
                        structural_validation="failed",
                        prompt_template=prompt_template_path,
                        judge_scoring={},
                    )
                    break

    logger.print_and_log(
        f"Saved judge labels to {output_path}",
    )


def main() -> None:
    args = parse_args()
    print(f"Starting judge evaluation with model {args.model} on dataset {args.dataset}")
    run_judge(
        dataset_path=args.dataset,
        output_path=args.output,
        prompt_template_path=args.prompt_template,
        model=args.model,
        max_tpm=args.max_tpm,
        max_completion_tokens=args.max_completion_tokens,
        max_retries=args.max_retries,
        items_to_test=args.items_to_test,
        random_mode=args.random,
        iteration=args.iteration,
    )


if __name__ == "__main__":
    main()
