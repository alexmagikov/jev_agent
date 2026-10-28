import asyncio
import json
from argparse import ArgumentParser
from dataclasses import asdict, is_dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from src.cases.document_type import DocumentMessage
from src.client import decision_sdk_options, AsyncCLM, DecisionModelClient, AsyncJeff, AsyncLaya, DecisionUsage
from src.schemas.document_schemas import DocumentResult
from src.extraction import validate_messages
from src.llm import AsyncLLMExtractor, LLMUsage
from src.metrics import AcceptanceMetrics, FieldMetrics, MetricsReport, calculate_metrics
from src.pipeline import PipelineError, PipelineRateLimitError, classify_document, process_document
from src.schemas.schemas import DocumentType, SplitFilter, DataSplit, Probability
from src.settings import CLMSettings, JeffSettings, DecisionModelSettings, LayaSettings, LLMSettings, PipelineSettings
from src.run_artifacts import RunArtifacts, input_fingerprint, public_settings, write_errors, write_results

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_ADAPTER = TypeAdapter(DocumentResult)

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "cases.jsonl"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "data" / "labels.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "document_types.jsonl"
DEFAULT_FULL_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "predictions.jsonl"
DEFAULT_LLM_ONLY_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "predictions.llm-only.jsonl"


class InputCase(BaseModel):
    """Одна строка входного файла `cases.jsonl`."""

    case_id: str = Field(min_length=1)
    split: DataSplit
    messages: list[DocumentMessage] = Field(min_length=1)


class DocumentTypeResult(BaseModel):
    """Промежуточный результат этапа выбора типа документа."""

    case_id: str
    document_type: DocumentType
    confidence: Probability
    reason: str | None = None


def load_cases(path: Path, split: SplitFilter = "dev") -> list[InputCase]:
    """Читает и валидирует JSONL, оставляя только выбранную выборку."""

    cases: list[InputCase] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            try:
                case = InputCase.model_validate_json(line)
            except ValidationError as error:
                raise ValueError(
                    f"Некорректная строка {line_number} в {path}"
                ) from error

            # Один case_id в разных split дал бы неоднозначную разметку.
            if case.case_id in seen:
                raise ValueError(f"Повторяющийся case_id в {path}: {case.case_id}")
            seen.add(case.case_id)
            if split == "all" or case.split == split:
                validate_messages(case.messages)
                cases.append(case)

    return cases


def load_labels(path: Path, case_ids: set[str]) -> list[dict[str, object]]:
    """Загружает из labels.jsonl эталоны только для выбранных кейсов."""

    labels: list[dict[str, object]] = []
    found_case_ids: set[str] = set()
    with path.open(encoding="utf-8") as labels_file:
        for line_number, line in enumerate(labels_file, start=1):
            if not line.strip():
                continue

            try:
                label = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Некорректная JSON-строка {line_number} в {path}"
                ) from error

            if not isinstance(label, dict) or not isinstance(
                label.get("case_id"), str
            ):
                raise ValueError(
                    f"Некорректная разметка в строке {line_number} файла {path}"
                )

            case_id = label["case_id"]
            if case_id not in case_ids:
                continue
            if case_id in found_case_ids:
                raise ValueError(f"Повторяющийся case_id в {path}: {case_id}")

            try:
                LABEL_ADAPTER.validate_python(label)
            except ValidationError:
                # Не раскрываем значения полей разметки в сообщении или traceback.
                raise ValueError(
                    f"Некорректная разметка в строке {line_number} файла {path}"
                ) from None
            labels.append(label)
            found_case_ids.add(case_id)

    missing_case_ids = case_ids - found_case_ids
    if missing_case_ids:
        missing = ", ".join(sorted(missing_case_ids))
        raise ValueError(f"В {path} нет разметки для кейсов: {missing}")

    return labels


async def classify_cases(
    cases: list[InputCase],
    client: DecisionModelClient,
) -> list[DocumentTypeResult]:
    """Последовательно классифицирует диалоги выбранной моделью решений."""

    results: list[DocumentTypeResult] = []
    for case in cases:
        decision = await classify_document(case.messages, client)
        results.append(
            DocumentTypeResult(
                case_id=case.case_id,
                document_type=decision.document_type,
                confidence=decision.confidence,
                reason=decision.reason,
            )
        )

    return results


def print_metrics(report: MetricsReport) -> None:
    """Печатает точность классификации и метрики полей."""

    print(
        "Точность типа документа: "
        f"{report.document_type_correct}/{report.total_cases} "
        f"({report.document_type_accuracy:.1%})"
    )
    print(f"Покрытие результатами: {report.predicted_cases}/{report.total_cases} ({_percent(report.prediction_coverage)})")
    if report.field_metrics_evaluated:
        print(f"Полностью правильные документы: {report.exact_document_correct}/{report.total_cases} "
              f"({_percent(report.exact_document_accuracy)})")
    if report.acceptance is not None:
        print_acceptance_metrics(report.acceptance, report.total_cases)
    if report.document_type_mismatches:
        print("\nНесовпадения по типу документа:")
        for mismatch in report.document_type_mismatches:
            predicted = (
                mismatch.predicted
                if mismatch.predicted is not None
                else "нет предсказания"
            )
            confidence = (
                f"{mismatch.confidence:.2f}"
                if mismatch.confidence is not None
                else "н/д"
            )
            print(
                f"  {mismatch.case_id}: ожидалось={mismatch.expected}, "
                f"предсказано={predicted}, confidence={confidence}"
            )
    else:
        print("Несовпадений по типу документа нет.")
    if not report.field_metrics_evaluated:
        return
    print("Метрики полей:")
    _print_field_metrics("  Все шаблоны", report.fields)
    errors = [(expected, actual, count) for expected, columns in report.fields.status_confusion.items()
              for actual, count in columns.items() if actual != expected and count]
    if errors:
        print("  Ошибки статусов (ожидалось → получено):")
        for expected, actual, count in errors:
            print(f"    {expected} → {actual}: {count}")
    for document_type, metrics in report.by_document_type.items():
        _print_field_metrics(f"  {document_type}", metrics)


def _percent(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "н/д"


def print_acceptance_metrics(metrics: AcceptanceMetrics, total_cases: int) -> None:
    """Общий вывод принятия результатов для CLI и сторон парного аудита."""
    print(f"Автоматически приняты: {metrics.accepted_cases}/{total_cases} "
          f"({_percent(metrics.automatic_acceptance_rate)})")
    print(f"Ошибки среди принятых: {metrics.accepted_incorrect}/{metrics.accepted_cases} "
          f"({_percent(metrics.accepted_error_rate)})")
    print(f"Правильные автоматически принятые: {metrics.accepted_correct}/{total_cases} "
          f"({_percent(metrics.correct_automatic_acceptance_rate)})")
    print(f"Требуют проверки: {metrics.needs_review_cases}/{total_cases} "
          f"({_percent(metrics.needs_review_rate)}); правильных: {metrics.needs_review_correct}; "
          f"ошибочных: {metrics.needs_review_incorrect}")
    if metrics.unknown_validation_cases:
        print(f"Результаты без статуса проверки: {metrics.unknown_validation_cases}")


def _print_field_metrics(name: str, metrics: FieldMetrics) -> None:
    print(
        f"{name}: \n"
        f"        precision={_percent(metrics.precision)},\n"
        f"        recall={_percent(metrics.recall)},\n"
        f"        value_accuracy={_percent(metrics.value_accuracy)},\n"
        f"        value_precision={_percent(metrics.value_precision)},\n"
        f"        value_recall={_percent(metrics.value_recall)},\n"
        f"        верных значений из всех ожидаемых={metrics.correct_values}/{metrics.expected_values},\n"
        f"        alias_aware_value_accuracy={_percent(metrics.alias_aware_value_accuracy)},\n"
        f"        alias_aware_value_recall={_percent(metrics.alias_aware_value_recall)},\n"
        f"        status_accuracy={_percent(metrics.status_accuracy)} ({metrics.correct_statuses}/{metrics.expected_fields}),\n"
        f"        field_accuracy={_percent(metrics.field_accuracy)} ({metrics.correct_status_values}/{metrics.expected_fields}),\n"
        f"        совпадение наборов ID источников={_percent(metrics.evidence_id_exact_match)},\n"
        f"        лишние заполнения={_percent(metrics.overfill_rate)}\n"
        f"        (TP={metrics.true_positives}, FP={metrics.false_positives}, "
        f"FN={metrics.false_negatives})\n"
    )


def print_llm_usage(usage: LLMUsage) -> None:
    """Печатает только расход и время, без диалогов и реквизитов."""
    print(
        f"LLM: запросов={usage.requests}, сетевых повторов={usage.retries}, "
        f"время извлечения={usage.elapsed_seconds:.2f} с"
    )
    if not usage.requests:
        return
    if usage.rate_limit_errors:
        print(f"  Ответов HTTP 429: {usage.rate_limit_errors}")
    if usage.usage_reports:
        print(
            f"  Токены: вход={usage.input_tokens}, выход={usage.output_tokens}; "
            f"usage получен для {usage.usage_reports}/{usage.responses} ответов"
        )
    else:
        print("  Токены: н/д (API не сообщил usage)")
    cost = f"${usage.estimated_cost:.6f}" if usage.estimated_cost is not None else "н/д"
    print(f"  Оценка стоимости по usage и заданным тарифам: {cost}")


def estimate_usage_costs(
    llm_usage: LLMUsage | None, decision_usage: DecisionUsage | None, *,
    decision_settings=None, include_decision: bool = False,
) -> dict[str, float | None]:
    """USD по usage и заданным тарифам; неполные сведения не считаются нулевой ценой."""

    llm_cost = 0.0
    if llm_usage is not None and llm_usage.requests:
        llm_cost = (llm_usage.estimated_cost
                    if llm_usage.requests == llm_usage.responses == llm_usage.usage_reports else None)
    
    decision_cost = 0.0
    if include_decision:
        decision_cost = None
        if decision_usage is not None:
            if not decision_usage.sdk_calls:
                decision_cost = 0.0
            elif decision_usage.sdk_calls == decision_usage.responses == decision_usage.usage_reports:
                input_price = getattr(decision_settings, "input_price_per_million", None)
                output_price = getattr(decision_settings, "output_price_per_million", None)
                if input_price is not None and output_price is not None:
                    decision_cost = (decision_usage.input_tokens * input_price
                                     + decision_usage.output_tokens * output_price) / 1_000_000
    total = llm_cost + decision_cost if llm_cost is not None and decision_cost is not None else None
    return {"llm_usd": llm_cost, "decision_usd": decision_cost, "total_usd": total}


def print_decision_usage(usage: DecisionUsage | None, *, name: str = "Decision model",
                         estimated_cost: float | None = None) -> None:
    if usage is None:
        print(f"{name}: статистика SDK н/д")
    else:
        print(f"{name}: вызовов SDK={usage.sdk_calls}, ответов={usage.responses}, время={usage.elapsed_seconds:.2f} с")
        print("  Внутренние HTTP-повторы SDK отдельно не измеряются.")
        if usage.usage_reports:
            print(f"  Токены {name}: вход={usage.input_tokens}, выход={usage.output_tokens}; "
                  f"usage получен для {usage.usage_reports}/{usage.responses} ответов")
        else:
            print(f"  Токены {name}: н/д (нет полного usage)")
    cost = f"${estimated_cost:.6f}" if estimated_cost is not None else "н/д"
    print(f"  Оценка стоимости {name} по usage и заданным тарифам: {cost}")
    print(f"  Стоимость {name} не включена в оценку стоимости LLM.")
    if name == "CLM":
        print("  Usage CLM считает токены кодировщика на промахах кеша, не полный текст HTTP-запроса.")


async def main(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path | None = None,
    labels_path: Path = DEFAULT_LABELS_PATH,
    split: SplitFilter = "dev",
    mode: Literal["classification", "full", "llm-only"] = "full",
    limit: int | None = None,
    debug: bool = False,
    fail_fast: bool = False,
    verifier: Literal["jeff", "clm", "jev", "laya", "decision"] = "jeff",
    case_ids: list[str] | None = None,
) -> Path:
    if mode not in ("classification", "full", "llm-only"):
        raise ValueError("Неизвестный режим обработки")
    if verifier not in ("jeff", "clm", "jev", "laya", "decision"):
        raise ValueError("Неизвестная модель проверки")
    if limit is not None and limit < 1:
        raise ValueError("--limit должен быть положительным")
    
    document_mode = mode in ("full", "llm-only")
    default_outputs = {
        "classification": DEFAULT_OUTPUT_PATH, "full": DEFAULT_FULL_OUTPUT_PATH,
        "llm-only": DEFAULT_LLM_ONLY_OUTPUT_PATH,
    }
    output_path = output_path or default_outputs[mode]
    if not output_path.name or output_path.is_dir() or output_path.name in {"config.json", "metrics.json", "run.json"}:
        raise ValueError("--output должен задавать имя файла результата, отличное от config.json, metrics.json и run.json")
    errors_path = output_path.with_name(output_path.stem + ".errors.jsonl")
    protected_paths = {input_path.resolve(), labels_path.resolve()}
    if output_path.resolve() in protected_paths or errors_path.resolve() in protected_paths:
        raise ValueError("Путь результата не должен совпадать с датасетом или разметкой")
    if errors_path.resolve() == output_path.resolve():
        raise ValueError("Пути результата и ошибок не должны совпадать")

    # Обнаружение ошибок до платных запросов
    cases = load_cases(input_path, split)
    if case_ids:
        requested_ids = set(case_ids)
        missing_ids = requested_ids - {case.case_id for case in cases}
        if missing_ids:
            raise ValueError(
                f"Кейсы не найдены в выборке '{split}': {', '.join(sorted(missing_ids))}"
            )
        cases = [case for case in cases if case.case_id in requested_ids]
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise ValueError("В выбранной выборке нет кейсов")
    
    labels = load_labels(labels_path, {case.case_id for case in cases})
    
    run_started = perf_counter()
    stopped_error: PipelineError | None = None
    client = None
    verifier_name = {"jeff": "Jeff", "clm": "CLM", "jev": "Jev", "laya": "Laya", "decision": "Decision model"}[verifier]
    decision_settings = None
    if mode != "llm-only":
        if verifier == "laya":
            decision_settings = LayaSettings()
            client = AsyncLaya(decision_settings)
        elif verifier == "jeff":
            decision_settings = JeffSettings()
            client = AsyncJeff(decision_settings)
        elif verifier == "clm":
            decision_settings = CLMSettings()
            client = AsyncCLM(decision_settings)
        else:
            decision_settings = DecisionModelSettings()
            options = decision_sdk_options(decision_settings)
            client = DecisionModelClient(api_key=options.pop("api_key"), sdk_options=options, name=decision_settings.model)
            verifier_name = decision_settings.model if verifier == "decision" else "Jev"

    pipeline_settings = PipelineSettings() if document_mode else None
    llm_settings = LLMSettings() if document_mode else None
    wrong_threshold = getattr(decision_settings, "wrong_threshold", pipeline_settings.wrong_threshold) if document_mode else None
    run = RunArtifacts(output_path, {
        "mode": mode, "verifier": verifier if mode != "llm-only" else None,
        "input": input_fingerprint(input_path), "labels": input_fingerprint(labels_path),
        "split": split, "limit": limit, "case_ids": [case.case_id for case in cases],
        "debug": debug, "fail_fast": fail_fast, "requested_output": str(output_path.resolve()),
        "llm": public_settings(llm_settings), "decision": public_settings(decision_settings),
        "max_attempts": pipeline_settings.max_attempts if pipeline_settings else None,
        "wrong_threshold": wrong_threshold,
    })
    
    output_path, errors_path = run.output, run.errors
    results = []
    errors: list[dict[str, str]] = []
    labels_by_id = {label["case_id"]: label for label in labels}
    extractor = None

    def checkpoint(status="running", failure_kind=None):
        processed = {row.case_id for row in results} | {error["case_id"] for error in errors}
        evaluated = [label for label in labels if label["case_id"] in processed]
        report = calculate_metrics(evaluated, [row.model_dump(mode="json") for row in results],
                                   evaluate_fields=document_mode)
        decision_usage = getattr(client, "usage", None)
        costs = estimate_usage_costs(extractor.usage if extractor is not None else None,
                                     decision_usage, decision_settings=decision_settings,
                                     include_decision=client is not None)
        run.checkpoint(results, errors, report, status=status, failure_kind=failure_kind, usage={
            "llm": asdict(extractor.usage) if extractor is not None else None,
            "decision": asdict(decision_usage) if is_dataclass(decision_usage) else None,
            "cost_estimate": costs,
            "elapsed_seconds": perf_counter() - run_started,
        })
        return report

    checkpoint()
    print(f"Каталог запуска: {run.directory}", flush=True)
    print(f"Результат: {output_path}", flush=True)
    print(f"Ошибки: {errors_path}", flush=True)
    print(f"Метрики: {run.metrics}", flush=True)

    async def process_cases():
        nonlocal stopped_error
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {case.case_id}", flush=True)
            if debug and document_mode:
                print(f"[{case.case_id}] Ожидаемый результат из labels.jsonl (не передаётся модели):", flush=True)
                print(json.dumps(labels_by_id[case.case_id], ensure_ascii=False, indent=2), flush=True)
            try:
                if document_mode:
                    result = await process_document(
                        case.case_id, case.messages, client, extractor,
                        max_attempts=pipeline_settings.max_attempts,
                        wrong_threshold=wrong_threshold, debug=debug, use_verifier=mode == "full",
                    )
                else:
                    result = (await classify_cases([case], client))[0]
            except PipelineError as error:
                # Ошибки программы и отмена не превращаются в ошибки данных кейса.
                errors.append({"case_id": case.case_id, "error": str(error)})
                print(f"[{case.case_id}] Ошибка кейса: {error}", flush=True)
                if fail_fast or isinstance(error, PipelineRateLimitError):
                    stopped_error = error
            else:
                results.append(result)
                validation = getattr(result, "validation", None)
                if validation is not None and validation.status == "needs_review":
                    print(f"[{case.case_id}] Документ сохранён; требуется проверка:", flush=True)
                    for issue in validation.issues:
                        print(f"  {issue.path or 'document'}: {issue.stage}/{issue.code}", flush=True)
                if debug and document_mode:
                    print(f"[{case.case_id}] Итог после проверок:", flush=True)
                    print(result.model_dump_json(indent=2), flush=True)
            checkpoint()
            if stopped_error is not None:
                break

    if document_mode:
        if mode == "llm-only":
            print("Режим llm-only: классификация LLM → извлечение LLM → Python; без внешнего верификатора.", flush=True)
        else:
            print(f"Режим full: классификация {verifier_name} → извлечение LLM → Python → проверка {verifier_name} → ограниченный ремонт.", flush=True)
            if verifier in ("clm", "jeff", "laya"):
                print(f"{verifier_name}: порог={wrong_threshold:g}; калибровка вероятности ошибки на наших данных не подтверждена.", flush=True)
            if verifier == "laya":
                print(f"Laya: max_len={decision_settings.max_len}, head_max_len={decision_settings.head_max_len}; длинный контекст может усекаться моделью.", flush=True)
        if debug:
            print("DEBUG: ответы модели и эталоны могут содержать реквизиты; не публикуйте вывод.", flush=True)
    try:
        if document_mode:
            async with AsyncLLMExtractor(llm_settings, debug=debug) as extractor:
                await process_cases()
        else:
            await process_cases()
    except BaseException as error:
        # Сохраняем отметку о прерывании, затем пробрасываем исходную ошибку.
        # При ошибке файловой системы повторная запись может тоже упасть.
        if not isinstance(error, OSError):
            try:
                checkpoint("interrupted", type(error).__name__)
            except OSError:
                error.add_note("Не удалось сохранить отметку о прерывании запуска")
        raise
    finally:
        if document_mode:
            costs = estimate_usage_costs(extractor.usage if extractor is not None else None,
                                         getattr(client, "usage", None), decision_settings=decision_settings,
                                         include_decision=client is not None)
            if extractor is not None:
                print_llm_usage(extractor.usage)
            if client is not None:
                print_decision_usage(getattr(client, "usage", None), name=verifier_name,
                                     estimated_cost=costs["decision_usd"])
            total_cost = f"${costs['total_usd']:.6f}" if costs["total_usd"] is not None else "н/д"
            print(f"Оценка общей стоимости запуска: {total_cost}")
            print(f"Полное время прогона ({mode}): {perf_counter() - run_started:.2f} с", flush=True)

    report = checkpoint("stopped" if stopped_error is not None else "completed",
                        type(stopped_error).__name__ if stopped_error is not None else None)

    print(f"Обработано кейсов: {report.total_cases}")
    review_count = sum(getattr(getattr(row, "validation", None), "status", None) == "needs_review"
                       for row in results)
    print(f"Успешно: {len(results) - review_count}; с ошибкой: {len(errors)}")
    if review_count:
        print(f"Требуют проверки: {review_count}")
    if stopped_error is not None:
        print(f"Прогон остановлен. Не обработано: {len(cases) - report.total_cases}")
        print("Частичные метрики: только обработанные кейсы, включая неуспешные.")
    else:
        print("Метрики учитывают все выбранные кейсы, включая неуспешные.")
    print(f"Результат: {output_path}")
    print_metrics(report)
    if stopped_error is not None:
        raise stopped_error
    return output_path


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Обработка документов с Jeff/CLM/Jev/Laya (full) или только LLM (llm-only)."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=None,
                        help="Имя результата; каждый запуск создаёт <имя>.runs/<run_id>/ рядом. "
                             "По умолчанию outputs/document_types.jsonl, predictions.jsonl (full) "
                             "или predictions.llm-only.jsonl (llm-only); фактический путь печатается в консоли.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--mode", choices=("classification", "full", "llm-only"), default="classification")
    parser.add_argument("--verifier", choices=("jeff", "clm", "jev", "laya", "decision"), default="jeff",
                        help="Классификатор и верификатор; по умолчанию собственный Jeff API.")
    parser.add_argument("--case-id", action="append", dest="case_ids",
                        help="Обработать только этот case_id из выбранного split; можно повторять. "
                             "Порядок — как в датасете, повторы ID игнорируются.")
    parser.add_argument("--limit", type=int,
                        help="Обработать первые N кейсов после фильтрации по --split и --case-id.")
    parser.add_argument(
        "--debug", action="store_true",
        help="В full/llm-only вывести эталон, ответ LLM и замечания проверок; вывод может содержать реквизиты.",
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="Остановиться на первой ошибке кейса; по умолчанию продолжать остальные.",
    )
    parser.add_argument(
        "--split",
        choices=("dev", "test", "all"),
        default="dev",
        help="Выборка для обработки; по умолчанию dev.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        asyncio.run(
            main(
                input_path=arguments.input,
                output_path=arguments.output,
                labels_path=arguments.labels,
                split=arguments.split,
                mode=arguments.mode,
                limit=arguments.limit,
                debug=arguments.debug,
                fail_fast=arguments.fail_fast,
                verifier=arguments.verifier,
                case_ids=arguments.case_ids,
            )
        )
    except ValidationError as error:
        # ValidationError настроек может содержать значения ключей: не печатаем её целиком.
        fields = ", ".join(".".join(map(str, item["loc"])) for item in error.errors(include_input=False))
        parser.exit(1, f"Ошибка валидации настроек или ответа API: {fields}\n")
    except (PipelineError, ValueError, OSError) as error:
        parser.exit(1, f"Ошибка: {error}\n")
