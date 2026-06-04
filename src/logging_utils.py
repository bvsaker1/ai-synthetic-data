import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_env_file(env_path: str | None = None) -> None:
    env_file = Path(env_path) if env_path else PROJECT_ROOT / ".env"
    if not env_file.exists():
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():
        clean_line = line.strip()
        if not clean_line or clean_line.startswith("#"):
            continue
        if "=" not in clean_line:
            continue

        key, value = clean_line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_iteration(default: str = "1") -> str:
    return os.getenv("ITERATION", default)


def build_iteration_dataset_path(prefix: str, extension: str = ".jsonl") -> str:
    iteration = get_iteration()
    return str(PROJECT_ROOT / "data" / f"{prefix}_{iteration}{extension}")


def build_iteration_log_path(extension: str = ".jsonl") -> str:
    iteration = get_iteration()
    return str(PROJECT_ROOT / "logs" / f"dataset_log_{iteration}{extension}")


class JsonEventLogger:
    """Shared JSON event logger.

    Supports two call styles for compatibility:
    - log("event_name", {"k": "v"})
    - log({"event": "event_name", ...})
    """

    def __init__(self, log_path: str, script_name: str, model: str, iteration: str | None = None) -> None:
        self.log_path = log_path
        self.script_name = script_name
        self.model = model
        self.iteration = iteration or os.getenv("ITERATION", "1")

    def _write_row(self, row: Dict[str, Any]) -> None:
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _build_row(self, body: Dict[str, Any]) -> Dict[str, Any]:
        row: Dict[str, Any] = {"timestamp": utc_now_iso()}

        message = body.get("message")
        event = body.get("event")
        if message is None and isinstance(event, str):
            message = event
        if message is not None:
            row["message"] = message

        item_id = body.get("id", body.get("trace_id"))
        if item_id is not None:
            row["id"] = item_id

        category = body.get("category")
        if category is not None:
            row["category"] = category

        if self.script_name == "judge_eval":
            for key in (
                "judge_scoring",
                "structural_validation",
                "status",
                "resolved_label",
                "overall_pass",
                "failed_qualities",
                "unresolved_reason",
                "error",
            ):
                if key in body:
                    row[key] = body[key]
        elif self.script_name == "analysis":
            for key in (
                "iteration",
                "judge_file",
                "human_file",
                "quality",
                "mismatched_qualities",
                "mismatch_count",
                "human_score",
                "judge_score",
                "human_overall_pass",
                "judge_overall_pass",
                "overall_pass_diff",
            ):
                if key in body:
                    row[key] = body[key]

        return row

    def log(self, event_or_payload: str | Dict[str, Any], payload: Dict[str, Any] | None = None) -> None:
        if isinstance(event_or_payload, str):
            body: Dict[str, Any] = {"event": event_or_payload}
            if payload:
                body.update(payload)
        else:
            body = dict(event_or_payload)

        row = self._build_row(body)
        self._write_row(row)

    def print_and_log(self, message: str, extra: Dict[str, Any] | None = None, **kwargs: Any) -> None:
        print(message)
        payload: Dict[str, Any] = {"message": message}
        if extra:
            payload.update(extra)
        if kwargs:
            payload.update(kwargs)
        self.log("print", payload)
