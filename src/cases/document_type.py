from typing import Literal

from pydantic import BaseModel, Field

from src.cases.base import Case
from src.schemas import Probability, Response, DocumentType, NoneReasonType


class DocumentMessage(BaseModel):
    """Одна реплика из входного диалога."""

    id: str = Field(min_length=1)
    role: Literal["user", "assistant"]
    text: str = Field(min_length=1)


class DocumentTypeState(BaseModel):
    """Диалог, по которому определяется актуальный запрос пользователя."""

    messages: list[DocumentMessage] = Field(min_length=1)


class DocumentTypeOptions[ValueT](BaseModel):
    invoice_draft: ValueT
    transfer_draft: ValueT
    none: ValueT


class NoneReasonOptions[ValueT](BaseModel):
    no_request: ValueT
    other_document: ValueT
    cancelled: ValueT


class DocumentTypeInstructions(BaseModel):
    question: str = (
        "Which supported document draft does the user explicitly want prepared now, if any?"
    )
    state_scope: str = (
        "Consider all `messages` in chronological order."
    )
    evidence_rule: str = (
        "Use only messages with role `user` as evidence of the request. "
        "Messages with role `assistant` only provide conversational context."
    )
    precedence_rule: str = (
        "A later user message that replaces, postpones, or cancels an earlier "
        "request takes precedence."
    )
    exclusion_rule: str = (
        "Questions about where a feature is located, how it works, or how to "
        "obtain a document are not requests to prepare a supported draft."
    )


class DocumentTypeQuestion(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: DocumentTypeInstructions = DocumentTypeInstructions()
    criteria: DocumentTypeOptions[str] = DocumentTypeOptions[str](
        invoice_draft=(
            "There is a current explicit request to prepare a payment invoice draft. "
            "Questions about where invoices are located or how invoices work do not qualify."
        ),
        transfer_draft=(
            "There is a current explicit request to prepare a transfer of the user's own "
            "money to the user's own card. A completed transfer is not implied."
        ),
        none=(
            "There is no current request for either supported draft: no request was made, "
            "the user only asks for information, requests another document or operation, "
            "or explicitly cancelled the earlier request."
        ),
    )


class NoneReasonQuestion(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: str = (
        "If the user does not currently request an `invoice_draft` or `transfer_draft` in "
        "`messages`, what is the main reason? Read the conversation chronologically and treat "
        "a later explicit cancellation as overriding an earlier request. This answer is ignored "
        "when a supported draft is currently requested."
    )
    criteria: NoneReasonOptions[str] = NoneReasonOptions[str](
        no_request=(
            "The conversation contains no request to prepare a document draft. "
            "The user only asks where a feature is located, how it works, or how to obtain a "
            "document, without asking to prepare a draft."
        ),
        other_document=(
            "The user requests a different document, an unsupported transfer, or another "
            "operation instead of either supported draft."
        ),
        cancelled="The user explicitly cancelled or withdrew an earlier supported request.",
    )


class DocumentTypeRequest(BaseModel):
    document_type: DocumentTypeQuestion = DocumentTypeQuestion()
    none_reason: NoneReasonQuestion = NoneReasonQuestion()


class DocumentTypeAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: DocumentType
    confidence: Probability
    probabilities: DocumentTypeOptions[Probability]


class NoneReasonAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: NoneReasonType
    confidence: Probability
    probabilities: NoneReasonOptions[Probability]


class DocumentTypeAnswers(BaseModel):
    document_type: DocumentTypeAnswer
    none_reason: NoneReasonAnswer


DocumentTypeResponse = Response[DocumentTypeAnswers]


NONE_REASON_TEXT: dict[NoneReasonType, str] = {
    "no_request": (
        "В диалоге нет запроса на подготовку поддерживаемого черновика. "
        "Пользователь запрашивает информацию, но не просит подготовить черновик."
    ),
    "other_document": (
        "Пользователь просит другой документ или операцию, не поддерживаемые заданием."
    ),
    "cancelled": "Пользователь отменил ранее сделанный запрос на подготовку черновика.",
}


def reason_text(
    document_type: DocumentTypeAnswer,
    none_reason: NoneReasonAnswer,
) -> str | None:
    """Возвращает обязательное текстовое объяснение для `document_type: none`."""

    if document_type.choice != "none":
        return None
    return NONE_REASON_TEXT[none_reason.choice]


class DocumentTypeCase(
    Case[DocumentTypeState, DocumentTypeRequest, DocumentTypeResponse]
):
    """Определяет, какой из поддерживаемых черновиков сейчас нужен пользователю."""

    state: DocumentTypeState
    questions: DocumentTypeRequest = DocumentTypeRequest()
    response_model: type[DocumentTypeResponse] = DocumentTypeResponse
