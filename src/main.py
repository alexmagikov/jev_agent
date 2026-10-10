import asyncio
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from src.cases.document_type import DocumentMessage
from src.client import AsyncJev
from src.metrics import FieldMetrics, MetricsReport, calculate_metrics
from src.pipeline import classify_document
from src.schemas import DocumentType, SplitFilter, DataSplit, Probability
from src.settings import JevClientSettings

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "cases.jsonl"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "data" / "labels.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "document_types.jsonl"


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

            labels.append(label)
            found_case_ids.add(case_id)

    missing_case_ids = case_ids - found_case_ids
    if missing_case_ids:
        missing = ", ".join(sorted(missing_case_ids))
        raise ValueError(f"В {path} нет разметки для кейсов: {missing}")

    return labels


def write_results(results: list[DocumentTypeResult], path: Path) -> None:
    """Сохраняет по одному результату на строку в формате JSONL."""

    contents = "\n".join(
        result.model_dump_json(exclude_none=True) for result in results
    )
    path.write_text(contents + ("\n" if contents else ""), encoding="utf-8")


async def classify_cases(
    cases: list[InputCase],
    client: AsyncJev,
) -> list[DocumentTypeResult]:
    """Последовательно классифицирует диалоги через JEV."""

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
    print(
        f"{name}: \n"
        f"        precision={metrics.precision:.1%},\n"
        f"        recall={metrics.recall:.1%},\n"
        f"        value_accuracy={value_accuracy},\n"
        f"        evidence_accuracy={evidence_accuracy},\n"
        f"        необоснованные заполнения={metrics.unsupported_fill_rate:.1%}\n"
        f"        (TP={metrics.true_positives}, FP={metrics.false_positives}, "
        f"FN={metrics.false_negatives})\n"
    )


async def main(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    labels_path: Path = DEFAULT_LABELS_PATH,
    split: SplitFilter = "dev",
) -> None:
    settings = JevClientSettings()
    client = AsyncJev(api_key=settings.api_key.get_secret_value())

    cases = load_cases(input_path, split)
    results = await classify_cases(cases, client)
    write_results(results, output_path)

    case_ids = {case.case_id for case in cases}
    labels = load_labels(labels_path, case_ids)
    report = calculate_metrics(
        labels,
        [result.model_dump(exclude_none=True) for result in results],
    )

    print(f"Обработано кейсов: {len(results)}")
    print(f"Результат: {output_path}")
    print_metrics(report)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Определяет тип запрошенного документа в диалогах."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument(
        "--split",
        choices=("dev", "test", "all"),
        default="dev",
        help="Выборка для обработки; по умолчанию dev.",
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    asyncio.run(
        main(
            input_path=arguments.input,
            output_path=arguments.output,
            labels_path=arguments.labels,
            split=arguments.split,
        )
    )
