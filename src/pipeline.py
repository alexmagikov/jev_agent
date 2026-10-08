
from pydantic import BaseModel

from src.cases.document_type import (
    DocumentMessage,
    DocumentTypeCase,
    DocumentTypeState,
    reason_text,
)
from src.client import AsyncJev
from src.schemas import DocumentType


class DocumentTypeDecision(BaseModel):
    """Результат первого этапа обработки диалога."""

    document_type: DocumentType
    reason: str | None = None


async def classify_document(
    messages: list[DocumentMessage],
    client: AsyncJev,
) -> DocumentTypeDecision:
    """Определяет актуальный тип черновика и объясняет результат `none`.

    Оба вопроса (`document_type` и `none_reason`) отправляются в JEV одним
    запросом. Причина используется только тогда, когда JEV выбрал `none`.
    """

    case = DocumentTypeCase(state=DocumentTypeState(messages=messages))
    response = await case.run(client)

    document_type = response.answers.document_type
    none_reason = response.answers.none_reason

    return DocumentTypeDecision(
        document_type=document_type.choice,
        reason=reason_text(document_type, none_reason),
    )
