"""Изолированные артефакты запуска; run.json публикуется после целого снимка."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4


def _write_text_atomic(contents: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".jev-checkpoint-", dir=path.parent) as directory:
        pending = Path(directory) / "checkpoint.jsonl"
        pending.write_text(contents, encoding="utf-8")
        pending.replace(path)


def write_results(results, path: Path) -> None:
    contents = "".join(result.model_dump_json(exclude_none=not hasattr(result, "fields")) + "\n"
                       for result in results)
    _write_text_atomic(contents, path)


def write_errors(errors: list[dict[str, str]], path: Path) -> None:
    _write_text_atomic("".join(json.dumps(error, ensure_ascii=False) + "\n" for error in errors), path)


def public_settings(settings) -> dict | None:
    """Явный список параметров эксперимента: ключи и session ID не сериализуются."""
    if settings is None:
        return None
    names = (
        "model", "api_protocol", "timeout_seconds", "max_retries", "max_output_tokens",
        "json_mode", "response_format", "temperature", "token_limit_parameter",
        "min_request_interval_seconds", "rate_limit_retries", "rate_limit_backoff_seconds",
        "input_price_per_million", "output_price_per_million", "max_len", "head_max_len",
    )
    return {name: getattr(settings, name) for name in names if hasattr(settings, name)}


def input_fingerprint(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class RunArtifacts:
    def __init__(self, output: Path, config: dict):
        if output.name in {"config.json", "metrics.json", "run.json"}:
            raise ValueError("Имя результата занято служебным артефактом запуска")
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:8]
        self.directory = output.parent / (output.stem + ".runs") / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.output = self.directory / output.name
        self.errors = self.directory / (output.stem + ".errors.jsonl")
        self.metrics = self.directory / "metrics.json"
        self.case_ids = config["case_ids"]
        self.files = {"predictions": self.output.name, "errors": self.errors.name,
                      "metrics": self.metrics.name, "config": "config.json"}
        _write_text_atomic(json.dumps({**config, "run_id": self.run_id}, ensure_ascii=False, indent=2) + "\n",
                           self.directory / "config.json")

    def checkpoint(self, results, errors, report, *, status="running", failure_kind=None, usage=None):
        # Сначала помечаем предыдущий снимок незавершённым. Если запись оборвётся,
        # старый manifest не выдаст новый неполный комплект за готовый результат.
        manifest = self.directory / "run.json"
        _write_text_atomic(json.dumps({"run_id": self.run_id, "status": "writing"}) + "\n", manifest)
        write_results(results, self.output)
        write_errors(errors, self.errors)
        _write_text_atomic(report.model_dump_json(indent=2) + "\n", self.metrics)
        processed = {row.case_id for row in results} | {error["case_id"] for error in errors}
        review_count = sum(getattr(getattr(row, "validation", None), "status", None) == "needs_review"
                           for row in results)
        summary = {
            "schema_version": 1, "run_id": self.run_id, "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "processed_case_ids": [cid for cid in self.case_ids if cid in processed],
            "unprocessed_case_ids": [cid for cid in self.case_ids if cid not in processed],
            "successful_cases": len(results) - review_count, "needs_review_cases": review_count,
            "failed_cases": len(errors),
            "metrics_scope": "processed_cases_including_failures", "files": self.files,
            "sha256": {key: hashlib.sha256((self.directory / name).read_bytes()).hexdigest()
                       for key, name in self.files.items()},
            "failure_kind": failure_kind, "usage": usage,
        }
        _write_text_atomic(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", manifest)


def load_run(directory: Path) -> dict:
    """Читает только связанный снимок: не смешивает файлы после сбоя записи."""
    summary = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if summary["status"] == "writing":
        raise ValueError("Снимок запуска не дописан")
    contents = {}
    for key in ("predictions", "errors", "metrics", "config"):
        name = summary["files"][key]
        if Path(name).name != name:
            raise ValueError("Некорректный путь артефакта запуска")
        data = (directory / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != summary["sha256"][key]:
            raise ValueError("Артефакты запуска не соответствуют сохранённому снимку")
        contents[key] = ([json.loads(line) for line in data.decode().splitlines() if line.strip()]
                         if key in ("predictions", "errors") else json.loads(data))
    return {"run": summary, **contents}
