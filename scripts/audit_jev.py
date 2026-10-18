"""Парный аудит проверки Jev: общий ответ LLM, реальные исправления, неизменные метрики."""

import argparse
import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from src.cases.document_type import NONE_REASON_TEXT
from src.extraction import ExtractionProposal, prepare_fields
from src.llm import AsyncLLMExtractor, EXTRACTION_PROMPT
from src.main import DEFAULT_INPUT_PATH, DEFAULT_LABELS_PATH, load_cases, load_labels
from src.metrics import calculate_metrics
from src.field_rules import equivalent_field_values
from src.pipeline import DocumentTypeDecision, PipelineError, process_selected_document
from src.settings import JevClientSettings, LLMSettings, PipelineSettings


class AuditBudgetError(RuntimeError):
    """Бюджет вызовов эксперимента исчерпан; это не ошибка проверяющей модели."""


@dataclass
class APICallBudget:
    maximum: int
    used: int = 0

    def __post_init__(self):
        if self.maximum < 1:
            raise ValueError("Бюджет должен быть положительным")

    def spend(self):
        if self.used >= self.maximum:
            raise AuditBudgetError("Бюджет API-вызовов исчерпан")
        self.used += 1


class BudgetedExtractor(AsyncLLMExtractor):
    def __init__(self, settings, budget):
        super().__init__(settings)
        self.budget = budget

    async def _send(self, payload):
        self.budget.spend()
        return await super()._send(payload)


class AuditJev:
    def __init__(self, sdk):
        self.sdk = sdk
        self.input_tokens = self.output_tokens = 0
        self.models = set()

    async def ainvoke(self, state, questions, response_model):
        response = await self.sdk.system_one(
            state=state.model_dump(mode="json"), questions=questions.model_dump(mode="json"),
        )
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        self.models.add(response.model)
        return response_model.model_validate(response.model_dump(mode="json"))


async def audit_cases(cases, extractor, jev, *, max_jev_calls=10, max_attempts=2,
                      wrong_threshold=0.7, on_event=None, initial_seeds=None):
    """Не принимает labels. Проверяются все последовательные кейсы без выбора по результату."""
    budget = APICallBudget(max_jev_calls)
    records = []

    def emit(record):
        if on_event is not None:
            on_event(record)

    for case in cases:
        if budget.used >= budget.maximum:
            break
        record = {"case_id": case.case_id, "split": case.split, "before": None, "after": None,
                  "messages": [message.model_dump(mode="json") for message in case.messages],
                  "seed_proposal": None, "candidates": [], "jev_rounds": [], "status": "started"}
        records.append(record)
        stage = "baseline"
        seed = None

        def capture_seed(proposal, fields):
            nonlocal seed
            seed = proposal
            record["seed_proposal"] = proposal.model_dump(mode="json")
            emit(record)

        def capture_candidate(proposal, fields):
            record["candidates"].append({"proposal": proposal.model_dump(mode="json"),
                                         "checked_fields": fields.model_dump(mode="json")})
            emit(record)

        class LoggedJev:
            async def ainvoke(self, state, questions, response_model):
                budget.spend()
                entry = {"questions": questions.model_dump(mode="json")}
                record["jev_rounds"].append(entry)
                emit(record)
                response = await jev.ainvoke(state, questions, response_model)
                entry["answers"] = response.model_dump(mode="json")["answers"]
                emit(record)
                return response

        try:
            initial = None
            if initial_seeds is None:
                classification = await extractor.classify(case.messages)
                decision = DocumentTypeDecision(
                    document_type=classification.document_type, confidence=classification.confidence,
                    reason=NONE_REASON_TEXT[classification.none_reason] if classification.none_reason else None,
                )
            else:
                decision, initial = initial_seeds[case.case_id]
            record["decision"] = decision.model_dump(mode="json")
            before = await process_selected_document(
                case.case_id, case.messages, decision, None, extractor, use_jev=False,
                max_attempts=max_attempts, on_candidate=capture_seed, initial_proposal=initial,
            )
            record["before"] = before.model_dump(mode="json")
            emit(record)
            if decision.document_type == "none":
                record.update(after=before.model_dump(mode="json"), status="no_document")
            else:
                stage = "enhanced"
                after = await process_selected_document(
                    case.case_id, case.messages, decision, LoggedJev(), extractor,
                    initial_proposal=seed, on_candidate=capture_candidate,
                    max_attempts=max_attempts, wrong_threshold=wrong_threshold,
                )
                record.update(after=after.model_dump(mode="json"), status="complete")
        except AuditBudgetError:
            record["status"] = "budget_exhausted"
            emit(record)
            break
        except OSError:
            raise  # Сбой сохранения не должен выглядеть как результат API-проверки.
        except Exception as error:
            # SDK-ошибка может содержать тело сервера: не сохраняем её текст или traceback.
            record["error_type"] = type(error).__name__
            if isinstance(error, PipelineError) and stage == "enhanced" and error.__cause__ is None:
                record["status"] = "rejected"
            else:
                record["status"] = "api_error"
                emit(record)
                break
        emit(record)
    return records


def load_initial_seeds(path, cases):
    """Только диалог, решение и исходный кандидат; никакие оценки/эталоны не переносятся."""
    latest = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                latest[record["case_id"]] = record
    seeds = {}
    for case in cases:
        record = latest.get(case.case_id)
        if record is None or record.get("before") is None:
            raise ValueError(f"Нет исходного результата для {case.case_id}")
        if record.get("messages") != [message.model_dump(mode="json") for message in case.messages]:
            raise ValueError(f"Изменился исходный диалог {case.case_id}")
        decision = DocumentTypeDecision.model_validate(record.get("decision"))
        proposal = None
        if decision.document_type != "none":
            proposal = ExtractionProposal.model_validate(record.get("seed_proposal"))
            checked = prepare_fields(proposal, case.messages, decision.document_type).model_dump(mode="json")
        else:
            checked = {}
        if record["before"].get("fields") != checked or record["before"].get("document_type") != decision.document_type:
            raise ValueError(f"Исходный кандидат не соответствует сохранённому результату {case.case_id}")
        seeds[case.case_id] = decision, proposal
    return seeds


def build_report(records, labels):
    targets = {row["case_id"]: row for row in labels}
    selected = [targets[record["case_id"]] for record in records]
    report = {"method": "paired verification-only: shared LLM classification and identical seed",
              "primary_target": "exact field status and value; evidence metrics reported separately",
              "attempted_cases": len(records), "complete_pairs": 0, "incomplete_cases": [],
              "field_changes": {"corrected": 0, "regressed": 0, "details": []},
              "alias_aware_field_changes": {"corrected": 0, "regressed": 0, "details": []},
              "item_attribute_changes": {"corrected": 0, "regressed": 0, "details": []},
              "remaining_mismatches": []}

    def correct(result, target, name, *, allow_aliases=False):
        if result is None or result["document_type"] != target["document_type"]:
            return False
        field = result.get("fields", {}).get(name, {})
        expected = target["fields"][name]
        same_value = (equivalent_field_values(target["document_type"], name, field.get("value"), expected.get("value"))
                      if allow_aliases else field.get("value") == expected.get("value"))
        return field.get("status") == expected.get("status") and same_value

    def expected_attributes(target):
        items = target["fields"].get("items", {})
        if target["document_type"] == "invoice_draft" and items.get("status") == "fillable":
            for index, item in enumerate(items["value"]):
                for attribute in ("description", "quantity", "unit", "unit_price_rub"):
                    yield index, attribute, item[attribute]

    def correct_attribute(result, target, index, attribute, expected):
        if result is None or result["document_type"] != target["document_type"]:
            return False
        field = result.get("fields", {}).get("items", {})
        items = field.get("value")
        if field.get("status") != "fillable" or not isinstance(items, list) or index >= len(items):
            return False
        return attribute in items[index] and items[index][attribute] == expected

    for side in ("before", "after"):
        predictions = [record[side] for record in records if record[side] is not None]
        report[side] = {"strict_metrics": calculate_metrics(selected, predictions).model_dump(mode="json"),
                        "expected_fillable_values": 0, "correct_expected_values": 0,
                        "total_expected_fields": 0, "correct_field_status_values": 0,
                        "expected_item_attributes": 0, "correct_item_attributes": 0,
                        "alias_aware_correct_expected_values": 0}
        for record in records:
            target = targets[record["case_id"]]
            for name, field in target["fields"].items():
                report[side]["total_expected_fields"] += 1
                report[side]["correct_field_status_values"] += int(correct(record[side], target, name))
                if field["status"] == "fillable":
                    report[side]["expected_fillable_values"] += 1
                    report[side]["correct_expected_values"] += int(correct(record[side], target, name))
                    report[side]["alias_aware_correct_expected_values"] += int(correct(record[side], target, name, allow_aliases=True))
            for index, attribute, expected in expected_attributes(target):
                report[side]["expected_item_attributes"] += 1
                report[side]["correct_item_attributes"] += int(correct_attribute(record[side], target, index, attribute, expected))

    for record in records:
        if record["before"] is None or record["after"] is None:
            report["incomplete_cases"].append(record["case_id"])
            continue
        report["complete_pairs"] += 1
        target = targets[record["case_id"]]
        attribute_mismatches = []
        for index, attribute, expected in expected_attributes(target):
            old, new = (correct_attribute(record[side], target, index, attribute, expected)
                        for side in ("before", "after"))
            path = f"items[{index}].{attribute}"
            if old != new:
                kind = "corrected" if new else "regressed"
                report["item_attribute_changes"][kind] += 1
                report["item_attribute_changes"]["details"].append({"case_id": record["case_id"], "field": path, "change": kind})
            if not new:
                attribute_mismatches.append(path)
        for name in target["fields"]:
            for counter, allow_aliases in (("field_changes", False), ("alias_aware_field_changes", True)):
                old, new = (correct(record[side], target, name, allow_aliases=allow_aliases)
                            for side in ("before", "after"))
                if old != new:
                    kind = "corrected" if new else "regressed"
                    report[counter][kind] += 1
                    report[counter]["details"].append({"case_id": record["case_id"], "field": name, "change": kind})
            if not correct(record["after"], target, name):
                scores = {key: answer["noul"] for round_ in record["jev_rounds"][-1:]
                          for key, answer in round_.get("answers", {}).items()
                          if key.split("::", 1)[0].split("[", 1)[0] == name}
                report["remaining_mismatches"].append({
                    "case_id": record["case_id"], "field": name,
                    "permitted_alias_only": correct(record["after"], target, name, allow_aliases=True),
                    "max_error_probability": max(scores.values(), default=None), "checks": scores,
                    **({"attribute_mismatches": attribute_mismatches} if name == "items" else {}),
                })
    delta = report["field_changes"]["corrected"] - report["field_changes"]["regressed"]
    report["conclusion"] = ("incomplete" if report["incomplete_cases"] or not records else
                            "observed_improvement_not_statistical_proof" if delta > 0 else
                            "observed_regression" if delta < 0 else "no_observed_advantage")
    return report


def positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Нужно положительное число")
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Разрешить платные API-запросы")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--limit", type=positive, default=5)
    parser.add_argument("--max-jev-calls", type=positive, default=10)
    parser.add_argument("--max-llm-calls", type=positive, default=30)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed-trace", type=Path,
                        help="Повторно проверить исходные кандидаты из trace.jsonl без новой baseline-генерации")
    return parser


async def run_live(args):
    cases = load_cases(args.input, args.split)[:args.limit]
    if not cases:
        raise ValueError("В выбранной выборке нет кейсов")
    labels = load_labels(args.labels, {case.case_id for case in cases})  # Только локальный scoring.
    initial_seeds = load_initial_seeds(args.seed_trace, cases) if args.seed_trace else None
    llm_settings, jev_settings, pipeline = LLMSettings(), JevClientSettings(), PipelineSettings()
    # Аудит считает фактические вызовы без скрытых сетевых повторов SDK/LLM.
    llm_settings = llm_settings.model_copy(update={"max_retries": 0, "rate_limit_retries": 0})
    directory = args.output_dir or Path(".scratch") / ("jev-audit-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f"))
    directory.mkdir(parents=True, exist_ok=False)
    budget = APICallBudget(args.max_llm_calls)
    with (directory / "trace.jsonl").open("x", encoding="utf-8") as trace:
        def checkpoint(record):
            trace.write(json.dumps(record, ensure_ascii=False) + "\n")
            trace.flush()

        async with AsyncTypeSafeClient(api_key=jev_settings.api_key.get_secret_value(),
                                      retry=RetryPolicy(max_retries=0), timeout=30) as sdk:
            jev = AuditJev(sdk)
            async with BudgetedExtractor(llm_settings, budget) as extractor:
                print(f"Парный аудит: {len(cases)} кейсов, до {args.max_jev_calls} вызовов Jev / {budget.maximum} LLM", flush=True)
                records = await audit_cases(cases, extractor, jev, max_jev_calls=args.max_jev_calls,
                                            max_attempts=pipeline.max_attempts,
                                            wrong_threshold=pipeline.wrong_threshold, on_event=checkpoint,
                                            initial_seeds=initial_seeds)
                report = build_report(records, labels)
                report.update(selected_cases=[case.case_id for case in cases],
                              not_attempted=[case.case_id for case in cases if case.case_id not in {r["case_id"] for r in records}],
                              llm_model=llm_settings.model, wrong_threshold=pipeline.wrong_threshold,
                              max_attempts=pipeline.max_attempts,
                              llm_max_output_tokens=llm_settings.max_output_tokens,
                              llm_temperature=llm_settings.temperature,
                              dataset_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                              labels_sha256=hashlib.sha256(args.labels.read_bytes()).hexdigest(),
                              seed_trace_sha256=hashlib.sha256(args.seed_trace.read_bytes()).hexdigest() if args.seed_trace else None,
                              baseline_origin="saved_trace" if initial_seeds is not None else "fresh_llm",
                              extraction_prompt_sha256=hashlib.sha256(EXTRACTION_PROMPT.encode()).hexdigest(),
                              verification_code_sha256=hashlib.sha256((Path(__file__).resolve().parents[1] / "src/verification.py").read_bytes()).hexdigest(),
                              field_rules_sha256=hashlib.sha256((Path(__file__).resolve().parents[1] / "src/field_rules.py").read_bytes()).hexdigest(),
                              llm_http_calls=budget.used, llm_usage=asdict(extractor.usage),
                              jev_api_calls=sum(len(record["jev_rounds"]) for record in records),
                              jev_models=sorted(jev.models),
                              jev_usage={"input_tokens": jev.input_tokens, "output_tokens": jev.output_tokens})
                if report["not_attempted"]:
                    report["conclusion"] = "incomplete"
    (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Полных пар: {report['complete_pairs']}/{len(cases)}; исправлено полей: {report['field_changes']['corrected']}; "
          f"испорчено: {report['field_changes']['regressed']}", flush=True)
    changes = report["item_attribute_changes"]
    print(f"Атрибуты позиций отдельно (не складываются с полями): исправлено {changes['corrected']}; "
          f"испорчено {changes['regressed']}", flush=True)
    for side, title in (("before", "До Jev"), ("after", "После Jev")):
        values = report[side]
        print(f"{title}: верных ожидаемых значений {values['correct_expected_values']}/{values['expected_fillable_values']}; "
              f"с разрешёнными вариантами {values['alias_aware_correct_expected_values']}/{values['expected_fillable_values']}", flush=True)
        print(f"  Поля целиком (статус + значение): {values['correct_field_status_values']}/{values['total_expected_fields']}; "
              f"атрибуты позиций: {values['correct_item_attributes']}/{values['expected_item_attributes']}", flush=True)
        metrics = values["strict_metrics"]["fields"]
        print(f"  precision={metrics['precision']:.2%}; recall={metrics['recall']:.2%}; "
              f"лишних заполнений={metrics['false_positives']}", flush=True)
    for item in report["remaining_mismatches"]:
        reason = "разрешённый вариант написания" if item["permitted_alias_only"] else "расхождение с разметкой"
        print(f"  {item['case_id']} {item['field']}: {reason}; максимальная оценка Jev={item['max_error_probability']}", flush=True)
        if item.get("attribute_mismatches"):
            print("    Не совпадают атрибуты: " + ", ".join(item["attribute_mismatches"]), flush=True)
    print(f"Вывод: {report['conclusion']}. Отчёт: {directory / 'report.json'}", flush=True)
    return not report["incomplete_cases"] and not report["not_attempted"]


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if not args.live:
        parser.error("Для платного запуска нужен явный --live. Ключи читаются только из .env.")
    try:
        complete = asyncio.run(run_live(args))
    except ValidationError:
        parser.exit(1, "Ошибка настроек или ответа API; значения скрыты.\n")
    except Exception as error:
        parser.exit(1, f"Аудит остановлен: {type(error).__name__}; текст ошибки скрыт.\n")
    raise SystemExit(0 if complete else 1)
