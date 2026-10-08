import asyncio
from types import SimpleNamespace
from typing import Any

from src.cases.document_type import (
    DocumentMessage,
    DocumentTypeAnswer,
    DocumentTypeRequest,
    DocumentTypeResponse,
    DocumentTypeState,
    NoneReasonAnswer,
)
from src.pipeline import classify_document


class FakeJev:
    def __init__(self, document_type: str, none_reason: str) -> None:
        self.document_type = document_type
        self.none_reason = none_reason
        self.call: tuple[Any, Any, Any] | None = None

    async def ainvoke(self, state: Any, questions: Any, response_model: Any) -> Any:
        self.call = (state, questions, response_model)
        return SimpleNamespace(
            answers=SimpleNamespace(
                document_type=DocumentTypeAnswer(
                    choice=self.document_type,
                    confidence=1.0,
                    probabilities={
                        "invoice_draft": float(self.document_type == "invoice_draft"),
                        "transfer_draft": float(self.document_type == "transfer_draft"),
                        "none": float(self.document_type == "none"),
                    },
                ),
                none_reason=NoneReasonAnswer(
                    choice=self.none_reason,
                    confidence=1.0,
                    probabilities={
                        "no_request": float(self.none_reason == "no_request"),
                        "other_document": float(self.none_reason == "other_document"),
                        "cancelled": float(self.none_reason == "cancelled"),
                    },
                ),
            )
        )


def test_classifies_none_and_returns_its_reason() -> None:
    client = FakeJev(document_type="none", none_reason="cancelled")

    decision = asyncio.run(
        classify_document(
            messages=[
                DocumentMessage(id="m1", role="user", text="Сделай счёт."),
                DocumentMessage(id="m2", role="user", text="Отмена, счёт не нужен."),
            ],
            client=client,
        )
    )

    assert decision.document_type == "none"
    assert decision.reason == (
        "Пользователь отменил ранее сделанный запрос на подготовку черновика."
    )
    assert client.call is not None
    state, questions, response_model = client.call
    assert isinstance(state, DocumentTypeState)
    assert isinstance(questions, DocumentTypeRequest)
    assert response_model is DocumentTypeResponse


def test_ignores_none_reason_for_supported_document() -> None:
    client = FakeJev(document_type="invoice_draft", none_reason="no_request")

    decision = asyncio.run(
        classify_document(
            messages=[
                DocumentMessage(
                    id="m1",
                    role="user",
                    text="Выставь счёт ООО Ромашка.",
                )
            ],
            client=client,
        )
    )

    assert decision.document_type == "invoice_draft"
    assert decision.reason is None
