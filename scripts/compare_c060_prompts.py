"""Сравнение только общих правил Jev на фиксированном ответе C060, без LLM.

По умолчанию сохраняет запросы без сети. --live выполняет 4 * repeats вызовов.
Ответ восстановлен из диагностического лога; это не запись прежнего HTTP-запроса.
"""

import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from urllib.parse import urlsplit

from src.cases.document_type import DocumentMessage
from src.client import DecisionModelClient, DecisionModelRateLimitError, decision_sdk_options
from src.extraction import ExtractionProposal, prepare_fields
from src.schemas.document_schemas import InvoiceFields
from src.settings import DecisionModelSettings
from src.verification import (
    VerificationError, VerificationQuestions, VerificationResponse, VerificationState,
    _RULES, build_verification_questions,
)


ROOT = Path(__file__).resolve().parents[1]
TARGET = "items[0].description::value_wrong"
VERSIONS = ("current", "v1_rules")
CANDIDATES = ("original", "wrong_product")


class DiagnosticResponse(VerificationResponse):
    model: str | None = None


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def build_experiment():
    case = next(json.loads(line) for line in (ROOT / "data/cases_v2.jsonl").read_text().splitlines()
                if json.loads(line)["case_id"] == "C060")
    messages = [DocumentMessage.model_validate(message) for message in case["messages"]]
    fields = {name: {"status": "missing", "value": None, "evidence": [], "calculation": None}
              for name in InvoiceFields.model_fields}
    fields["recipient_name"].update(
        status="fillable", value="ООО Класс", evidence=[{"message_id": "m1", "quote": "ООО Класс"}],
    )
    fields["items"].update(status="fillable", value=[
        {"description": name, "quantity": None, "unit": "шт.", "unit_price_rub": None,
         "quantity_calculation": None, "unit_price_calculation": None}
        for name in ("ручка", "маркер")
    ], evidence=[
        {"message_id": "m1", "quote": "ручки и маркеры вместе на 1600 рублей, всего 8 штук"},
        {"message_id": "m1", "quote": "Для обеих позиций единица измерения — шт."},
    ])
    fields["amount_rub"].update(
        status="fillable", value=1600, evidence=[{"message_id": "m1", "quote": "вместе на 1600 рублей"}],
    )
    archive = json.loads((ROOT / "prompts/versions/jev-v1/resolved_rules.json").read_text())
    old_rules = archive["verification_rules"]["_RULES"]
    if not isinstance(old_rules, str) or not old_rules.strip():
        raise ValueError("Общие правила снимка отсутствуют")
    rules = {"current": _RULES, "v1_rules": old_rules}
    experiment = {
        "case_id": "C060", "target_check": TARGET,
        "scope": "Only state.rules changes between versions; current Python and all current questions are fixed.",
        "candidate_origin": "Reconstructed first LLM answer from the C060 diagnostic log; not a historical HTTP capture.",
        "snapshot": "jev-v1",
        "rules_identical": old_rules == _RULES,
        "expected_error": {"original": False, "wrong_product": True},
        "rules_sha256": {version: digest(text) for version, text in rules.items()},
        "candidates": {}, "requests": {},
    }
    for candidate in CANDIDATES:
        raw = deepcopy(fields)
        if candidate == "wrong_product":
            raw["items"]["value"][0]["description"] = "карандаш"
        proposal = ExtractionProposal.model_validate({"fields": raw})
        checked = prepare_fields(proposal, messages, "invoice_draft")
        questions = build_verification_questions(
            checked, proposal=proposal, document_type="invoice_draft", messages=messages,
        )
        experiment["candidates"][candidate] = {
            "proposal": proposal.model_dump(mode="json"), "checked_fields": checked.model_dump(mode="json"),
        }
        for version, text in rules.items():
            state = VerificationState(messages=messages, document_type="invoice_draft", rules=text)
            experiment["requests"][f"{candidate}/{version}"] = {
                "state": state.model_dump(mode="json"), "questions": questions.model_dump(mode="json"),
            }
    return experiment


def summarize(results, *, candidates=CANDIDATES, versions=VERSIONS):
    summary = {}
    for candidate in candidates:
        for version in versions:
            key = f"{candidate}/{version}"
            rows = [row for row in results if row["request_id"] == key]
            ok = [row for row in rows if row["status"] == "ok"]
            scores = [row["target_score"] for row in ok]
            summary[key] = {
                "scores": scores, "mean": mean(scores) if scores else None,
                "target_flagged": sum(row["target_flagged"] for row in ok),
                "any_check_flagged": sum(bool(row["triggered_checks"]) for row in ok),
                "api_errors": len(rows) - len(ok),
            }
    return summary


def save_report(report, directory, client):
    report["summary"] = summarize(report["results"], candidates=report["candidate_names"], versions=report["versions"])
    if hasattr(client, "usage"):
        report["usage"] = asdict(client.usage)
    temporary = directory / "report.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(directory / "report.json")


async def run_comparison(experiment, client, directory, *, repeats, threshold, metadata=None):
    target = experiment["target_check"]
    candidate_names = tuple(experiment.get("candidate_names", CANDIDATES))
    version_names = tuple(experiment.get("versions", VERSIONS))
    report = {
        "status": "running", "planned_calls": len(candidate_names) * len(version_names) * repeats,
        "repeats": repeats, "threshold": threshold,
        "target_check": target, "candidate_names": list(candidate_names), "versions": list(version_names),
        "scope": experiment["scope"],
        "candidate_origin": experiment["candidate_origin"],
        "expected_error": experiment["expected_error"], "rules_sha256": experiment["rules_sha256"],
        "request_sha256": {key: digest(value) for key, value in experiment["requests"].items()},
        "connection": metadata or {}, "results": [],
    }
    save_report(report, directory, client)
    try:
        for repeat in range(1, repeats + 1):
            # Alternate order to reduce a systematic effect of server drift over time.
            versions = version_names if repeat % 2 else version_names[::-1]
            candidates = candidate_names if repeat % 2 else candidate_names[::-1]
            for candidate in candidates:
                for version in versions:
                    key = f"{candidate}/{version}"
                    request = experiment["requests"][key]
                    state = VerificationState.model_validate(request["state"])
                    questions = VerificationQuestions.model_validate(request["questions"])
                    row = {"request_id": key, "repeat": repeat}
                    started = perf_counter()
                    print(f"[{len(report['results']) + 1}/{report['planned_calls']}] {key}: запрос…", flush=True)
                    stop = False
                    try:
                        response = await client.ainvoke(state=state, questions=questions, response_model=DiagnosticResponse)
                        if set(response.answers) != set(questions.root):
                            raise VerificationError("Неполный или неожиданный набор проверок")
                        score = response.answers[target].noul
                        row.update(
                            status="ok", target_score=score, target_flagged=score >= threshold,
                            response_model=response.model,
                            answers={name: answer.noul for name, answer in response.answers.items()},
                            triggered_checks=[name for name, answer in response.answers.items() if answer.noul >= threshold],
                        )
                        print(f"  {target}: {score:.3f}; порог={threshold:.2f}", flush=True)
                    except Exception as error:
                        # Never persist remote bodies, credentials, or exception text.
                        row.update(status="error", error_type=type(error).__name__)
                        stop = isinstance(error, DecisionModelRateLimitError)
                        print(f"  Ошибка запроса: {type(error).__name__}; оценка не получена.", flush=True)
                    row["elapsed_seconds"] = perf_counter() - started
                    report["results"].append(row)
                    save_report(report, directory, client)
                    if stop:
                        report["status"] = "incomplete"
                        return report
        report["status"] = "complete" if all(row["status"] == "ok" for row in report["results"]) else "incomplete"
        return report
    except (asyncio.CancelledError, KeyboardInterrupt):
        report["status"] = "interrupted"
        raise
    finally:
        save_report(report, directory, client)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Выполнить API-запросы; без флага только сохранить входы")
    parser.add_argument("--repeats", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--threshold", type=float, default=.8, help="Порог только для отчёта, по умолчанию 0.80")
    parser.add_argument("--output-dir", type=Path, help="Новая папка; существующие результаты не перезаписываются")
    args = parser.parse_args(argv)
    if not 0 < args.threshold < 1:
        parser.error("Порог должен быть числом строго между 0 и 1")
    try:
        experiment = build_experiment()
        settings = DecisionModelSettings() if args.live and not experiment["rules_identical"] else None
        directory = args.output_dir or ROOT / ".scratch" / (
            "c060-prompt-compare-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        )
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "requests.json").write_text(
            json.dumps(experiment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        planned = 0 if experiment["rules_identical"] else 4 * args.repeats
        print(f"C060: {planned} запросов Jev; LLM не вызывается. Порог={args.threshold:.2f}.")
        print("Меняется только state.rules; v1_rules не откатывает остальные вопросы или Python.")
        print(f"Зафиксированные входы: {directory / 'requests.json'}", flush=True)
        if experiment["rules_identical"]:
            print("Текущие общие правила уже совпадают с v1. Сравнение не требуется; API не вызывался.")
            return 0
        if not args.live:
            print("Запросы подготовлены без сети. Для измерения добавьте --live.")
            return 0
        options = decision_sdk_options(settings, max_retries=0)
        client = DecisionModelClient(options.pop("api_key"), sdk_options=options, name=settings.model)
        print(f"Отчёт после каждого ответа: {directory / 'report.json'}", flush=True)
        report = asyncio.run(run_comparison(
            experiment, client, directory, repeats=args.repeats, threshold=args.threshold,
            metadata={"requested_model": settings.model, "host": urlsplit(settings.base_url).hostname,
                      "timeout_seconds": settings.timeout_seconds, "sdk_max_retries": 0},
        ))
        for key, row in report["summary"].items():
            scores = ", ".join(f"{score:.3f}" for score in row["scores"]) or "нет ответов"
            print(f"{key}: [{scores}]; срабатываний описания {row['target_flagged']}/{len(row['scores'])}; "
                  f"срабатываний любых проверок {row['any_check_flagged']}; ошибок API {row['api_errors']}")
        print("Для original ожидается низкая оценка ошибки; для wrong_product — высокая.")
        print(f"Статус: {report['status']}. Отчёт: {directory / 'report.json'}")
        return 0 if report["status"] == "complete" else 1
    except KeyboardInterrupt:
        print("Прервано; полученные ответы сохранены в указанном отчёте.")
        return 130
    except Exception as error:
        print(f"Сравнение остановлено: {type(error).__name__}; текст ошибки скрыт. "
              "Проверьте входные файлы, DECISION_* и что папка вывода ещё не существует.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
