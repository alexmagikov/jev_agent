import asyncio
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from src.cases.document_type import DocumentMessage
from src.client import AsyncJev
from src.pipeline import classify_document
from src.schemas import DocumentType
from src.settings import JevClientSettings


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = REPOSITORY_ROOT / "cases.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "document_types.jsonl"


class InputCase(BaseModel):
    """Одна строка входного файла `cases.jsonl`."""

    case_id: str = Field(min_length=1)
    split: DataSplit
    messages: list[DocumentMessage] = Field(min_length=1)


class DocumentTypeResult(BaseModel):
    """Промежуточный результат этапа выбора типа документа."""

    case_id: str
    document_type: DocumentType
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
                reason=decision.reason,
            )
        )

    return results


def write_results(results: list[DocumentTypeResult], path: Path) -> None:
    """Сохраняет по одному результату на строку в формате JSONL."""

    contents = "\n".join(
        result.model_dump_json(exclude_none=True) for result in results
    )
    path.write_text(contents + ("\n" if contents else ""), encoding="utf-8")


async def main(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    split: SplitFilter = "dev",
) -> None:
    settings = JevClientSettings()
    client = AsyncJev(api_key=settings.api_key.get_secret_value())

    cases = load_cases(input_path, split)
    results = await classify_cases(cases, client)
    write_results(results, output_path)

    print(f"Обработано кейсов: {len(results)}")
    print(f"Результат: {output_path}")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Определяет тип запрошенного документа в диалогах."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
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
            split=arguments.split,
        )
    )
