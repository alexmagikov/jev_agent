"""Сравнение вопросов о расчётной цене на сохранённом кандидате T014, без генерации LLM.

Без --live только сохраняет входы. С --live выполняет 4 * repeats запросов Jev.
"""

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

from scripts.compare_c060_prompts import digest, run_comparison
from src.cases.document_type import DocumentMessage
from src.client import DecisionModelClient, decision_sdk_options
from src.extraction import ExtractionProposal, prepare_fields
from src.settings import DecisionModelSettings
from src.verification import VerificationState, _RULES, _build_verification_checks, build_verification_questions


ROOT = Path(__file__).resolve().parents[1]
TARGET = "items[0].unit_price_rub::calculation_operands_wrong"
CANDIDATES = ("correct_700", "stale_count_800")
VERSIONS = ("generic", "calculated_price")


def build_experiment(trace_path):
    source = trace_path.read_bytes()
    record = None
    for line in source.decode("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("case_id") == "T014":
                record = row
    if record is None or not record.get("candidates"):
        raise ValueError("В трассе нет проверенного Python кандидата T014")
    saved = record["candidates"][-1]
    messages = [DocumentMessage.model_validate(m) for m in record["messages"]]
    proposal = ExtractionProposal.model_validate(saved["proposal"])
    checked = prepare_fields(proposal, messages, "invoice_draft")
    if checked.model_dump(mode="json") != saved["checked_fields"]:
        raise ValueError("Сохранённые проверенные поля не соответствуют кандидату")
    plan = proposal.fields["items"].value[0].unit_price_calculation
    if checked.items.value[0].unit_price_rub != 700 or plan is None:
        raise ValueError("Для T014 нужна исходная расчётная цена 700")
    count_indices = [i for i, operand in enumerate(plan.operands)
                     if operand.source.message_id == "m3" and operand.source.quote == "8"]
    if len(count_indices) != 1:
        raise ValueError("Не найден единственный операнд актуального количества T014")

    stale = proposal.model_copy(deep=True)
    old_count = stale.fields["items"].value[0].unit_price_calculation.operands[count_indices[0]].source
    old_count.message_id, old_count.quote = "m1", "7"
    state = VerificationState(messages=messages, document_type="invoice_draft", rules=_RULES).model_dump(mode="json")
    experiment = {
        "case_id": "T014", "target_check": TARGET, "candidate_names": CANDIDATES, "versions": VERSIONS,
        "scope": "Only the target question text and true/false criteria change; state, field payloads and all other questions stay fixed.",
        "candidate_origin": "Last Python-validated T014 candidate from trace; control replaces only the count operand source with the earlier count.",
        "source_trace_sha256": hashlib.sha256(source).hexdigest(),
        "expected_error": {"correct_700": False, "stale_count_800": True},
        "rules_sha256": {version: digest(_RULES) for version in VERSIONS},
        "candidates": {}, "requests": {},
    }
    for name, candidate, price in zip(CANDIDATES, (proposal, stale), (700, 800), strict=True):
        fields = prepare_fields(candidate, messages, "invoice_draft")
        if fields.items.value[0].unit_price_rub != price:
            raise ValueError("Формула контрольного кандидата не дала ожидаемую цену")
        questions = build_verification_questions(fields, proposal=candidate, messages=messages).model_dump(mode="json")
        generic = deepcopy(questions)
        generic[TARGET]["instructions"]["question"] = _build_verification_checks(fields)["items[0].unit_price_rub::value_wrong"]
        generic[TARGET]["criteria"] = {
            "true": "Yes. The specific error in instructions.question is present.",
            "false": "No. The specific error is absent; unrelated errors do not count.",
        }
        if generic[TARGET] == questions[TARGET]:
            raise ValueError("Вопросы совпадают: отдельная проверка расчётной цены отсутствует")
        experiment["candidates"][name] = {
            "proposal": candidate.model_dump(mode="json"), "checked_fields": fields.model_dump(mode="json"),
        }
        for version, payload in zip(VERSIONS, (generic, questions), strict=True):
            experiment["requests"][f"{name}/{version}"] = {"state": deepcopy(state), "questions": payload}
    return experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True, help="Трасса аудита с кандидатом T014")
    parser.add_argument("--live", action="store_true", help="Разрешить API-запросы Jev")
    parser.add_argument("--repeats", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--threshold", type=float, default=.8)
    parser.add_argument("--output-dir", type=Path, help="Новая папка; существующая не перезаписывается")
    args = parser.parse_args(argv)
    if not 0 < args.threshold < 1:
        parser.error("Порог должен быть строго между 0 и 1")
    try:
        experiment = build_experiment(args.trace)
        directory = args.output_dir or ROOT / ".scratch" / (
            "t014-price-compare-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f"))
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "requests.json").write_text(json.dumps(experiment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"T014: {4 * args.repeats} запросов Jev при --live; LLM не вызывается. Порог={args.threshold:.2f}.")
        print("Сравниваются общий вопрос и вопрос о выборе операндов расчётной цены; вопрос об операции, кандидаты и контекст зафиксированы.")
        print(f"Входы: {directory / 'requests.json'}", flush=True)
        if not args.live:
            print("Подготовлено без сети. Для измерения добавьте --live.")
            return 0
        settings = DecisionModelSettings()
        options = decision_sdk_options(settings, max_retries=0)
        client = DecisionModelClient(options.pop("api_key"), sdk_options=options, name=settings.model)
        report = asyncio.run(run_comparison(
            experiment, client, directory, repeats=args.repeats, threshold=args.threshold,
            metadata={"requested_model": settings.model, "host": urlsplit(settings.base_url).hostname,
                      "timeout_seconds": settings.timeout_seconds, "sdk_max_retries": 0},
        ))
        for key, row in report["summary"].items():
            scores = ", ".join(f"{score:.3f}" for score in row["scores"]) or "нет ответов"
            print(f"{key}: [{scores}]; срабатываний {row['target_flagged']}/{len(row['scores'])}; ошибок API {row['api_errors']}")
        print("Для correct_700 ожидается низкая оценка ошибки, для stale_count_800 — высокая.")
        print(f"Статус: {report['status']}. Отчёт: {directory / 'report.json'}")
        return 0 if report["status"] == "complete" else 1
    except KeyboardInterrupt:
        print("Прервано; завершённые ответы сохранены в отчёте.")
        return 130
    except Exception as error:
        print(f"Сравнение остановлено: {type(error).__name__}; текст ошибки скрыт. Проверьте трассу и новую папку вывода.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
