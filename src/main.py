import asyncio
import json
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from src.cases.document_type import DocumentMessage
from src.client import decision_sdk_options, AsyncCLM, DecisionModelClient, AsyncJeff, AsyncLaya, DecisionUsage
from src.schemas.document_schemas import DocumentResult
from src.extraction import validate_messages
from src.llm import AsyncLLMExtractor, LLMUsage
from src.metrics import FieldMetrics, MetricsReport, calculate_metrics
from src.pipeline import PipelineError, PipelineRateLimitError, classify_document, process_document
from src.schemas.schemas import DocumentType, SplitFilter, DataSplit, Probability
from src.settings import CLMSettings, JeffSettings, DecisionModelSettings, LayaSettings, LLMSettings, PipelineSettings

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

            if split == "all" or case.split == split:
                if case.case_id in seen:
                    raise ValueError(f"Повторяющийся case_id в {path}: {case.case_id}")
                validate_messages(case.messages)
                seen.add(case.case_id)
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


def write_results(results: list[DocumentTypeResult] | list[DocumentResult], path: Path) -> None:
    """Сохраняет по одному результату на строку в формате JSONL."""

    contents = "\n".join(
        result.model_dump_json(exclude_none=isinstance(result, DocumentTypeResult))
        for result in results
    )
    _write_text_atomic(contents + ("\n" if contents else ""), path)


def _write_text_atomic(contents: str, path: Path) -> None:
    """Заменяет один файл целиком, не оставляя оборванный JSONL при ошибке записи."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".jev-checkpoint-", dir=path.parent) as directory:
        pending = Path(directory) / "checkpoint.jsonl"
        pending.write_text(contents, encoding="utf-8")
        pending.replace(path)


def write_errors(errors: list[dict[str, str]], path: Path) -> None:
    contents = "".join(json.dumps(error, ensure_ascii=False) + "\n" for error in errors)
    _write_text_atomic(contents, path)


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
    print("Метрики полей:")
    _print_field_metrics("  Все шаблоны", report.fields)
    for document_type, metrics in report.by_document_type.items():
        _print_field_metrics(f"  {document_type}", metrics)


def _print_field_metrics(name: str, metrics: FieldMetrics) -> None:
    correct_values = round(metrics.true_positives * metrics.value_accuracy) if metrics.value_accuracy is not None else 0
    expected_values = metrics.true_positives + metrics.false_negatives
    value_accuracy = (
        f"{metrics.value_accuracy:.1%}"
        if metrics.value_accuracy is not None
        else "н/д"
    )
    evidence_accuracy = (
        f"{metrics.evidence_accuracy:.1%}"
        if metrics.evidence_accuracy is not None
        else "н/д"
    )
    alias_aware_accuracy = (
        f"{metrics.alias_aware_value_accuracy:.1%}"
        if metrics.alias_aware_value_accuracy is not None else "н/д"
    )
    print(
        f"{name}: \n"
        f"        precision={metrics.precision:.1%},\n"
        f"        recall={metrics.recall:.1%},\n"
        f"        value_accuracy={value_accuracy},\n"
        f"        верных значений из всех ожидаемых={correct_values}/{expected_values},\n"
        f"        alias_aware_value_accuracy={alias_aware_accuracy},\n"
        f"        evidence_accuracy={evidence_accuracy},\n"
        f"        необоснованные заполнения={metrics.unsupported_fill_rate:.1%}\n"
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


def print_decision_usage(usage: DecisionUsage | None, *, name: str = "Decision model") -> None:
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
) -> None:
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
    errors_path = output_path.with_name(output_path.stem + ".errors.jsonl")
    protected_paths = {input_path.resolve(), labels_path.resolve()}
    if output_path.resolve() in protected_paths or (document_mode and errors_path.resolve() in protected_paths):
        raise ValueError("Путь результата не должен совпадать с датасетом или разметкой")
    if document_mode and errors_path.resolve() == output_path.resolve():
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
            settings = DecisionModelSettings()
            options = decision_sdk_options(settings)
            client = DecisionModelClient(api_key=options.pop("api_key"), sdk_options=options, name=settings.model)
            verifier_name = settings.model if verifier == "decision" else "Jev"

    if document_mode:
        pipeline_settings = PipelineSettings()
        wrong_threshold = decision_settings.wrong_threshold if decision_settings else pipeline_settings.wrong_threshold
        results = []
        errors: list[dict[str, str]] = []
        labels_by_id = {label["case_id"]: label for label in labels}
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
        async with AsyncLLMExtractor(LLMSettings(), debug=debug) as extractor:
            try:
                for index, case in enumerate(cases, start=1):
                    print(f"[{index}/{len(cases)}] {case.case_id}", flush=True)
                    if debug:
                        print(f"[{case.case_id}] Ожидаемый результат из labels.jsonl (не передаётся модели):", flush=True)
                        print(json.dumps(labels_by_id[case.case_id], ensure_ascii=False, indent=2), flush=True)

                    try:
                        result = await process_document(
                            case.case_id, case.messages, client, extractor,
                            max_attempts=pipeline_settings.max_attempts,
                            wrong_threshold=wrong_threshold,
                            debug=debug,
                            use_verifier=mode == "full",
                        )
                    except PipelineError as error:
                        # Не ловим Exception: ошибки программы и отмена должны быть видны.
                        errors.append({"case_id": case.case_id, "error": str(error)})
                        print(f"[{case.case_id}] Ошибка кейса: {error}", flush=True)
                        if fail_fast or isinstance(error, PipelineRateLimitError):
                            write_errors(errors, errors_path)
                            stopped_error = error
                            break
                    else:
                        results.append(result)
                        if debug:
                            print(f"[{case.case_id}] Итог после проверок:", flush=True)
                            print(result.model_dump_json(indent=2), flush=True)
                    # Снимок только текущего прогона; не смешиваем его с прежними записями.
                    write_results(results, output_path)
                    write_errors(errors, errors_path)
            finally:
                print_llm_usage(extractor.usage)
                if client is not None:
                    print_decision_usage(getattr(client, "usage", None), name=verifier_name)
                print(f"Полное время прогона ({mode}): {perf_counter() - run_started:.2f} с", flush=True)
    else:
        results = await classify_cases(cases, client)
        write_results(results, output_path)

    evaluated_labels = labels
    if stopped_error is not None:
        attempted_ids = {result.case_id for result in results} | {error["case_id"] for error in errors}
        evaluated_labels = [label for label in labels if label["case_id"] in attempted_ids]
    report = calculate_metrics(
        evaluated_labels,
        [result.model_dump(mode="json") for result in results],
    )

    print(f"Обработано кейсов: {len(evaluated_labels)}")
    if document_mode:
        print(f"Успешно: {len(results)}; с ошибкой: {len(errors)}")
        print(f"Ошибки: {errors_path}")
        if stopped_error is not None:
            print(f"Прогон остановлен. Не обработано: {len(cases) - len(evaluated_labels)}")
            print("Частичные метрики: только обработанные кейсы, включая неуспешные.")
        else:
            print("Метрики учитывают все выбранные кейсы, включая неуспешные.")
    if stopped_error is not None and not results:
        print(f"Новых результатов нет; файл {output_path} не изменялся.")
    else:
        print(f"Результат: {output_path}")
    print_metrics(report)
    if stopped_error is not None:
        raise stopped_error


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Обработка документов с Jeff/CLM/Jev/Laya (full) или только LLM (llm-only)."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=None,
                        help="По умолчанию outputs/document_types.jsonl, predictions.jsonl (full) "
                             "или predictions.llm-only.jsonl (llm-only).")
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
        help="В full/llm-only остановиться на первой ошибке кейса; по умолчанию продолжать остальные.",
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
