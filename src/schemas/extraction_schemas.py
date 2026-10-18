from typing import Literal

from pydantic import Field, model_validator

from .schemas import DocumentType, NoneReasonType, Probability

from .document_schemas import (
    DocumentSchema,
    FieldStatus,
    InvoiceFields,
    InvoiceItem,
    Number,
    TransferFields,
)

type SupportedDocumentType = Literal["invoice_draft", "transfer_draft"]
type PreparedFields = InvoiceFields | TransferFields


class DocumentClassification(DocumentSchema):
    """Тип и причина отказа, предложенные LLM, без проверки Jev."""

    document_type: DocumentType
    confidence: Probability
    none_reason: NoneReasonType | None

    @model_validator(mode="after")
    def validate_none_reason(self) -> "DocumentClassification":
        if (self.document_type == "none") != (self.none_reason is not None):
            raise ValueError("none_reason обязателен только для document_type='none', иначе null")
        return self


class SourceQuote(DocumentSchema):
    message_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)

class Operand(DocumentSchema):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=40)
    source: SourceQuote


class Calculation(DocumentSchema):
    """Формула для того значения, рядом с которым она указана."""

    expression: str = Field(min_length=1, max_length=256)
    operands: list[Operand] = Field(min_length=1, max_length=32)


class ProposedInvoiceItem(InvoiceItem):
    """Позиция до вычислений; планы расчётов остаются рядом с её числами."""

    quantity_calculation: Calculation | None = None
    unit_price_calculation: Calculation | None = None


class ProposedField(DocumentSchema):
    status: FieldStatus
    value: str | Number | list[ProposedInvoiceItem] | None
    evidence: list[SourceQuote]
    calculation: Calculation | None = None


class ExtractionProposal(DocumentSchema):
    fields: dict[str, ProposedField]

class ExtractionError(ValueError):
    """Некорректное предложение модели, а не отсутствующие данные пользователя."""
