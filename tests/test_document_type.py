import pytest
from pydantic import ValidationError

from src.cases.document_type import (
    DocumentMessage,
    DocumentTypeAnswer,
    DocumentTypeCase,
    DocumentTypeState,
    NoneReasonAnswer,
    reason_text,
)


def test_serializes_state_and_questions_for_typesafe() -> None:
    case = DocumentTypeCase(
        state=DocumentTypeState(
            messages=[
                DocumentMessage(
                    id="m1",
                    role="user",
                    text="Выставь счёт ООО Ромашка.",
                )
            ]
        )
    )

    state = case.state.model_dump(mode="json")
    questions = case.questions.model_dump(mode="json")

    assert state["messages"][0]["id"] == "m1"
    assert questions["document_type"]["type"] == "choice"
    assert set(questions["document_type"]["criteria"]) == {
        "invoice_draft",
        "transfer_draft",
        "none",
    }
    assert set(questions["none_reason"]["criteria"]) == {
        "no_request",
        "other_document",
        "cancelled",
    }


def test_rejects_unknown_message_role() -> None:
    with pytest.raises(ValidationError):
        DocumentMessage(id="m1", role="system", text="internal")


def test_maps_none_reason_to_output_text() -> None:
    document_type = DocumentTypeAnswer(
        choice="none",
        confidence=0.9,
        probabilities={
            "invoice_draft": 0.05,
            "transfer_draft": 0.05,
            "none": 0.9,
        },
    )
    none_reason = NoneReasonAnswer(
        choice="cancelled",
        confidence=1.0,
        probabilities={
            "no_request": 0.0,
            "other_document": 0.0,
            "cancelled": 1.0,
        },
    )

    assert reason_text(document_type, none_reason) == (
        "Пользователь отменил ранее сделанный запрос на подготовку черновика."
    )


def test_has_no_none_reason_for_selected_document() -> None:
    document_type = DocumentTypeAnswer(
        choice="invoice_draft",
        confidence=1.0,
        probabilities={
            "invoice_draft": 1.0,
            "transfer_draft": 0.0,
            "none": 0.0,
        },
    )
    none_reason = NoneReasonAnswer(
        choice="no_request",
        confidence=0.0,
        probabilities={
            "no_request": 1 / 3,
            "other_document": 1 / 3,
            "cancelled": 1 / 3,
        },
    )

    assert reason_text(document_type, none_reason) is None
