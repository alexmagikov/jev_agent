"""Схемы итоговых черновиков; сами по себе не извлекают данные из диалога."""

from datetime import date
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from .schemas import Probability

type FieldStatus = Literal["fillable", "missing", "ambiguous", "invalid"]
type Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
type MessageId = Annotated[str, Field(min_length=1)]


def validate_iso_date(value: str) -> str:
    date.fromisoformat(value)
    return value


type ISODate = Annotated[
    str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$"), AfterValidator(validate_iso_date)
]


class DocumentSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FieldResult[ValueT](DocumentSchema):
    """Решение о заполнении одного поля и его источники."""

    status: FieldStatus
    value: ValueT | None
    evidence: list[MessageId]
    derivation: str | None

    @model_validator(mode="after")
    def validate_value_status(self) -> FieldResult[ValueT]:
        if self.status == "fillable" and self.value is None:
            raise ValueError("При status='fillable' значение не может быть null")
        if self.status == "fillable" and isinstance(self.value, str) and not self.value.strip():
            raise ValueError("При status='fillable' строка не может быть пустой или состоять из пробелов")
        if self.status != "fillable" and self.value is not None:
            raise ValueError("Незаполняемое поле должно содержать value=null")
        return self


class InvoiceItem(DocumentSchema):
    """Позиция счёта: неизвестные части явно передаются как null."""

    description: str | None
    quantity: Number | None
    unit: str | None
    unit_price_rub: Number | None


class InvoiceFields(DocumentSchema):
    recipient_name: FieldResult[str]
    recipient_inn: FieldResult[str]
    recipient_email: FieldResult[str]
    recipient_account: FieldResult[str]
    recipient_bik: FieldResult[str]
    subject: FieldResult[str]
    items: FieldResult[list[InvoiceItem]]
    amount_rub: FieldResult[Number]
    vat_mode: FieldResult[Literal["no_vat", "rate"]]
    vat_rate_percent: FieldResult[Number]
    basis_number: FieldResult[str]
    basis_date: FieldResult[ISODate]


class TransferFields(DocumentSchema):
    amount_rub: FieldResult[Number]
    destination_card_masked: FieldResult[str]
    destination_bank: FieldResult[str]
    source_account_kind: FieldResult[str]
    recipient_relation: FieldResult[str]


class ResultBase(DocumentSchema):
    case_id: str = Field(min_length=1)
    # Эталонная разметка не содержит уверенности; предсказания могут её передавать.
    confidence: Probability | None = None


class InvoiceResult(ResultBase):
    document_type: Literal["invoice_draft"]
    fields: InvoiceFields


class TransferResult(ResultBase):
    document_type: Literal["transfer_draft"]
    fields: TransferFields


class NoDocumentResult(ResultBase):
    document_type: Literal["none"]
    fields: dict[str, object] = Field(max_length=0)
    reason: str = Field(min_length=1, pattern=r"\S")


type DocumentResult = Annotated[
    InvoiceResult | TransferResult | NoDocumentResult,
    Field(discriminator="document_type"),
]
