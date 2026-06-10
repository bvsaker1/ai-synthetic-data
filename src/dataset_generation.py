import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import instructor
from groq import Groq
from pydantic import BaseModel, ConfigDict, Field

from src.logging_utils import (
    JsonEventLogger,
    build_iteration_dataset_path,
    build_iteration_log_path,
    load_env_file,
    PROJECT_ROOT,
    utc_now_iso,
)
from templates.appliance_repair_template import build_appliance_repair_messages
from templates.electrical_repair_template import build_electrical_repair_messages
from templates.gen_home_repair_template import build_general_home_repair_messages
from templates.hvac_maintenance_template import build_hvac_maintenance_messages
from templates.plumbing_repair_template import build_plumbing_repair_messages


DEFAULT_DATASET_MODEL = "llama-3.3-70b-versatile"
DEFAULT_DATASET_TEMPERATURE = 0.7
MAX_LLM_CALLS_PER_ATTEMPT = 5
TEMPLATE_CHOICES = ["appliance", "electrical", "plumbing", "hvac", "general_home"]


def resolve_dataset_model_default() -> str:
    model = os.getenv("DATASET_MODEL", "").strip()
    if model:
        return model

    # Backward-compatible fallback for older env setups.
    fallback_model = os.getenv("GROQ_MODEL", "").strip()
    if fallback_model:
        return fallback_model
    return DEFAULT_DATASET_MODEL


def resolve_dataset_temperature_default() -> float:
    raw_value = os.getenv("DATASET_TEMPERATURE", "").strip()
    if not raw_value:
        return DEFAULT_DATASET_TEMPERATURE

    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(f"DATASET_TEMPERATURE must be a float, got: {raw_value!r}") from error

    if value < 0 or value > 2:
        raise ValueError(f"DATASET_TEMPERATURE must be between 0 and 2, got: {value}")
    return value


def build_iteration_prompt_log_path(iteration: str) -> str:
    return str(PROJECT_ROOT / "logs" / f"dataset_prompt_{iteration}.log")


def _get_prompt_content(messages: List[Dict[str, str]], role: str) -> str:
    for message in messages:
        if message.get("role") == role:
            return str(message.get("content", "")).strip()
    return ""


def format_prompt_run_header(
    iteration: str,
    model: str,
    count: int,
    selected_templates: List[str],
    template_names: List[str],
) -> str:
    divider = "#" * 100
    selected_text = ", ".join(selected_templates) if selected_templates else "all"
    template_name_text = ", ".join(template_names)
    return (
        f"{divider}\n"
        "DATASET PROMPT RUN HEADER\n"
        f"timestamp_utc: {utc_now_iso()}\n"
        f"iteration: {iteration}\n"
        f"model: {model}\n"
        f"count_per_template: {count}\n"
        f"selected_templates: {selected_text}\n"
        f"active_template_names: {template_name_text}\n"
        f"{divider}\n\n"
    )


def write_prompt_run_header(
    prompt_log_path: str,
    iteration: str,
    model: str,
    count: int,
    selected_templates: List[str],
    template_names: List[str],
) -> None:
    header = format_prompt_run_header(
        iteration=iteration,
        model=model,
        count=count,
        selected_templates=selected_templates,
        template_names=template_names,
    )
    with open(prompt_log_path, "a", encoding="utf-8") as prompt_log:
        prompt_log.write(header)
    print(header)


def format_prompt_block(
    category: str,
    template_name: str,
    messages: List[Dict[str, str]],
) -> str:
    system_prompt = _get_prompt_content(messages, "system")
    user_prompt = _get_prompt_content(messages, "user")
    divider = "=" * 100
    return (
        f"{divider}\n"
        "PROMPT METADATA\n"
        f"category: {category}\n"
        f"template_name: {template_name}\n"
        f"{divider}\n"
        "SYSTEM PROMPT:\n"
        f"{system_prompt}\n\n"
        "USER PROMPT:\n"
        f"{user_prompt}\n"
    )


def write_template_prompts(
    prompt_log_path: str,
    category: str,
    template_name: str,
    messages: List[Dict[str, str]],
) -> None:
    block = format_prompt_block(
        category=category,
        template_name=template_name,
        messages=messages,
    )
    with open(prompt_log_path, "a", encoding="utf-8") as prompt_log:
        prompt_log.write(block)
        prompt_log.write("\n")
    print(block)


class RepairQAModel(BaseModel):
    # generation stage validates the core content; id/category are assigned later.
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    equipment_problem: str = Field(..., min_length=1)
    tools_required: List[str] = Field(..., min_length=1)
    steps: List[str] = Field(..., min_length=3)
    safety_info: str = Field(..., min_length=1)
    tips: List[str] = Field(..., min_length=1)
    judge_scoring: str = Field(default="")

    model_config = ConfigDict(extra="ignore")


class RepairQABatchModel(BaseModel):
    items: List[RepairQAModel] = Field(..., min_length=1)


def call_llm_typed_batch(
    client: Any,
    model: str,
    temperature: float,
    messages: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_model=RepairQABatchModel,
    )
    return [item.model_dump() for item in response.items]


def generate_items_from_template(
    client: Any,
    model: str,
    temperature: float,
    template_builder_func,
    category_name: str,
    count: int,
    logger: JsonEventLogger,
) -> List[Dict[str, Any]]:
    messages = template_builder_func(count=count)
    valid_items = call_llm_typed_batch(client, model, temperature, messages)
    logger.print_and_log(
        f"  Instructor validated {len(valid_items)} items from {category_name}",
        {"category": category_name, "validated_count": len(valid_items)},
    )
    return valid_items


def generate_candidate_items(
    client: Any,
    model: str,
    temperature: float,
    template_builder_func,
    category_name: str,
    count: int,
    logger: JsonEventLogger,
) -> List[Dict[str, Any]]:
    collected_items: List[Dict[str, Any]] = []

    for llm_call in range(1, MAX_LLM_CALLS_PER_ATTEMPT + 1):
        try:
            batch_items = generate_items_from_template(
                client=client,
                model=model,
                temperature=temperature,
                template_builder_func=template_builder_func,
                category_name=category_name,
                count=count,
                logger=logger,
            )
        except Exception as error:
            logger.print_and_log(
                f"  LLM/Instructor error on {category_name} (LLM call {llm_call}): {error}",
                {
                    "category": category_name,
                    "llm_call": llm_call,
                    "error": str(error),
                },
            )
            continue

        collected_items.extend(batch_items)
        logger.print_and_log(
            f"  Collected {len(collected_items)}/{count} raw items from {category_name} (LLM call {llm_call})",
            {
                "category": category_name,
                "llm_call": llm_call,
                "collected": len(collected_items),
                "target": count,
            },
        )

        if len(collected_items) >= count:
            return collected_items[:count]

    raise RuntimeError(
        f"Could not collect {count} items from {category_name} template "
        f"({len(collected_items)}/{count}) after {MAX_LLM_CALLS_PER_ATTEMPT} calls."
    )


def save_raw_dataset_append(rows: List[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    load_env_file()
    default_dataset_model = resolve_dataset_model_default()
    default_dataset_temperature = resolve_dataset_temperature_default()
    default_raw_output = build_iteration_dataset_path("raw_diy_dataset")
    parser = argparse.ArgumentParser(description="Generate raw DIY repair QA dataset from 5 templates.")
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of items to generate per template (default: 10). Total raw items = count * 5.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_dataset_model,
        help=(
            "Groq model name. Reads DATASET_MODEL from .env when set "
            f"(fallback default: {default_dataset_model})."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=default_dataset_temperature,
        help=(
            "Sampling temperature (0-2). Reads DATASET_TEMPERATURE from .env when set "
            f"(fallback default: {default_dataset_temperature})."
        ),
    )
    parser.add_argument(
        "--raw-output",
        type=str,
        default=default_raw_output,
        help=f"Raw output JSONL path (default: {default_raw_output}).",
    )
    parser.add_argument(
        "--template",
        action="append",
        choices=TEMPLATE_CHOICES,
        help=(
            "Generate only selected template(s). Repeat flag for multiple values. "
            "If omitted, all templates are generated."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be greater than 0.")

    iteration = os.getenv("ITERATION", "1")
    log_path = build_iteration_log_path(extension=".jsonl")
    logger = JsonEventLogger(log_path=log_path, script_name="dataset_generation", model=args.model, iteration=iteration)
    prompt_log_path = build_iteration_prompt_log_path(iteration)
    Path(prompt_log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(prompt_log_path).write_text("", encoding="utf-8")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to .env or your shell environment.")

    groq_client = Groq(api_key=api_key)
    client = instructor.from_groq(groq_client)

    # Start a new raw file for this run.
    Path(args.raw_output).write_text("", encoding="utf-8")
    logger.print_and_log(
        f"Generating {args.count} items from each of 5 templates ({args.count * 5} total raw rows)...",
        {"raw_output": args.raw_output},
    )

    templates = [
        ("appliance", "appliance_repair", build_appliance_repair_messages),
        ("electrical", "electrical_repair", build_electrical_repair_messages),
        ("plumbing", "plumbing_repair", build_plumbing_repair_messages),
        ("hvac", "hvac_maintenance", build_hvac_maintenance_messages),
        ("general_home", "general_home_repair", build_general_home_repair_messages),
    ]

    selected_templates = TEMPLATE_CHOICES
    if args.template:
        selected = set(args.template)
        templates = [tpl for tpl in templates if tpl[0] in selected]
        selected_templates = sorted(selected)
        logger.print_and_log(
            f"Template filter active: {sorted(selected)}",
            {"selected_templates": sorted(selected)},
        )

    logger.print_and_log(
        f"Writing template prompts to {prompt_log_path}",
        {"prompt_log_path": prompt_log_path},
    )
    write_prompt_run_header(
        prompt_log_path=prompt_log_path,
        iteration=iteration,
        model=args.model,
        count=args.count,
        selected_templates=selected_templates,
        template_names=[template_name for _, template_name, _ in templates],
    )

    next_id = 1
    total_saved = 0
    for short_category, template_category_name, template_builder in templates:
        template_messages = template_builder(count=args.count)
        write_template_prompts(
            prompt_log_path=prompt_log_path,
            category=short_category,
            template_name=template_category_name,
            messages=template_messages,
        )
        logger.print_and_log(
            f"Logged prompts for {template_category_name} to {prompt_log_path}",
            {
                "category": short_category,
                "template": template_category_name,
                "prompt_log_path": prompt_log_path,
            },
        )

        logger.print_and_log(
            f"Generating {args.count} items from {template_category_name}...",
            {"category": short_category, "template": template_category_name},
        )

        raw_items = generate_candidate_items(
            client=client,
            model=args.model,
            temperature=args.temperature,
            template_builder_func=template_builder,
            category_name=template_category_name,
            count=args.count,
            logger=logger,
        )

        rows_to_append: List[Dict[str, Any]] = []
        for item in raw_items:
            row = dict(item)
            row["id"] = f"qa_{next_id:05d}"
            row["category"] = short_category
            rows_to_append.append(row)
            next_id += 1

        save_raw_dataset_append(rows_to_append, args.raw_output)
        total_saved += len(rows_to_append)
        logger.print_and_log(
            f"Saved {len(rows_to_append)} rows from {template_category_name} to {args.raw_output}",
            {
                "category": short_category,
                "template": template_category_name,
                "saved_rows": len(rows_to_append),
                "total_saved": total_saved,
            },
        )

    logger.print_and_log(
        f"Finished raw generation. Saved {total_saved} rows to {args.raw_output}",
        {"raw_output": args.raw_output, "total_saved": total_saved, "log_path": log_path},
    )


if __name__ == "__main__":
    main()
