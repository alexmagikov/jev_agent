import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.cases.document_type import DocumentTypeAnswer, NoneReasonAnswer
from src.main import InputCase, classify_cases, load_cases, print_metrics, write_results
from src.metrics import calculate_metrics


class FakeJev:
    async def ainvoke(self, state: Any, questions: Any, response_model: Any) -> Any:
        cancelled = "Отмена" in state.messages[-1].text
        document_type = "none" if cancelled else "invoice_draft"
        none_reason = "cancelled" if cancelled else "no_request"

        return SimpleNamespace(
            answers=SimpleNamespace(
                document_type=DocumentTypeAnswer(
                    choice=document_type,
                    confidence=0.73,
                    probabilities={
                        "invoice_draft": float(document_type == "invoice_draft"),
                        "transfer_draft": 0.0,
                        "none": float(document_type == "none"),
                    },
                ),
                none_reason=NoneReasonAnswer(
                    choice=none_reason,
                    confidence=1.0,
                    probabilities={
                        "no_request": float(none_reason == "no_request"),
                        "other_document": 0.0,
                        "cancelled": float(none_reason == "cancelled"),
                    },
                ),
            )
        )


def test_load_cases_filters_split(tmp_path: Path) -> None:
    input_path = tmp_path / "cases.jsonl"
    rows = [
        {
            "case_id": "C001",
            "split": "dev",
            "messages": [{"id": "m1", "role": "user", "text": "Сделай счёт."}],
        },
        {
            "case_id": "C002",
            "split": "test",
            "messages": [{"id": "m1", "role": "user", "text": "Сделай счёт."}],
        },
    ]
    input_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    cases = load_cases(input_path, split="dev")

    assert [case.case_id for case in cases] == ["C001"]


def test_classifies_cases_and_writes_jsonl(tmp_path: Path) -> None:
    cases = [
        InputCase.model_validate(
            {
                "case_id": "C008",
                "split": "dev",
                "messages": [
                    {"id": "m1", "role": "user", "text": "Сделай счёт."},
                    {"id": "m2", "role": "user", "text": "Отмена."},
                ],
            }
        )
    ]

    results = asyncio.run(classify_cases(cases, client=FakeJev()))
    output_path = tmp_path / "document_types.jsonl"
    write_results(results, output_path)

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row == {
        "case_id": "C008",
        "document_type": "none",
        "confidence": 0.73,
        "reason": "Пользователь отменил ранее сделанный запрос на подготовку черновика.",
    }


@pytest.mark.parametrize(
    ("predicted", "display", "confidence", "confidence_display"),
    [
        ("none", "none", 0.73, "0.73"),
        ("none", "none", 0.0, "0.00"),
        ("none", "none", None, "н/д"),
        (None, "нет предсказания", None, "н/д"),
    ],
)
def test_print_metrics_shows_document_type_mismatches(
    capsys: pytest.CaptureFixture[str],
    predicted: str | None,
    display: str,
    confidence: float | None,
    confidence_display: str,
) -> None:
    labels = [{"case_id": "C1", "document_type": "invoice_draft", "fields": {}}]
    predictions = (
        [{"case_id": "C1", "document_type": predicted,
          **({"confidence": confidence} if confidence is not None else {})}]
        if predicted is not None else []
    )

    print_metrics(calculate_metrics(labels, predictions))

    output = capsys.readouterr().out
    assert "Несовпадения по типу документа:" in output
    assert (
        f"C1: ожидалось=invoice_draft, предсказано={display}, "
        f"confidence={confidence_display}"
    ) in output


def test_print_metrics_reports_no_document_type_mismatches(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = [{"case_id": "C1", "document_type": "none", "fields": {}}]

    print_metrics(calculate_metrics(rows, rows))

    assert "Несовпадений по типу документа нет." in capsys.readouterr().out
