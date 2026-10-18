import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from src.schemas.document_schemas import (
    DocumentResult, FieldResult, InvoiceFields, InvoiceItem, ISODate, Number,
    TransferFields,
)


def test_field_preserves_explicit_nulls() -> None:
    field = FieldResult[str](
        status="missing", value=None, evidence=[], derivation=None
    )
    assert field.model_dump(mode="json") == {
        "status": "missing", "value": None, "evidence": [], "derivation": None
    }


@pytest.mark.parametrize("status", ["missing", "ambiguous", "invalid"])
def test_unfillable_field_rejects_value(status: str) -> None:
    with pytest.raises(ValidationError):
        FieldResult[str](status=status, value="value", evidence=[], derivation=None)


def test_fillable_field_rejects_null() -> None:
    with pytest.raises(ValidationError):
        FieldResult[str](status="fillable", value=None, evidence=["m1"], derivation=None)


def test_partial_item_keeps_unknown_parts_null() -> None:
    item = InvoiceItem(
        description="Разработка сайта", quantity=None, unit=None, unit_price_rub=None
    )
    assert item.model_dump(mode="json")["unit_price_rub"] is None


def test_dev_labels_match_result_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    dev_ids = {
        case["case_id"]
        for line in (root / "data/cases.jsonl").read_text().splitlines()
        if (case := json.loads(line))["split"] == "dev"
    }
    adapter = TypeAdapter(DocumentResult)
    for line in (root / "data/labels.jsonl").read_text().splitlines():
        label = json.loads(line)
        if label["case_id"] in dev_ids:
            result = adapter.validate_python(label)
            assert result.model_dump(mode="json", exclude_unset=True) == label


@pytest.mark.parametrize("fields,reason", [({"amount_rub": {}}, "Нет запроса"), ({}, "")])
def test_none_rejects_nonempty_fields_or_empty_reason(fields: dict, reason: str) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(DocumentResult).validate_python({
            "case_id": "C1", "document_type": "none", "fields": fields, "reason": reason
        })


def test_selected_template_requires_all_fields() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(DocumentResult).validate_python({
            "case_id": "C1", "document_type": "transfer_draft", "fields": {}
        })


def test_field_rejects_unknown_properties() -> None:
    with pytest.raises(ValidationError):
        FieldResult[str](
            status="missing", value=None, evidence=[], derivation=None, typo="x"
        )


def test_numeric_field_rejects_boolean() -> None:
    with pytest.raises(ValidationError):
        FieldResult[Number](status="fillable", value=True, evidence=["m1"], derivation=None)


def test_template_fields_match_template_file() -> None:
    root = Path(__file__).resolve().parents[1]
    templates = json.loads((root / "data/form_templates.json").read_text())
    assert set(InvoiceFields.model_fields) == set(templates["invoice_draft"]["fields"])
    assert set(TransferFields.model_fields) == set(templates["transfer_draft"]["fields"])


def test_document_type_cannot_use_other_template() -> None:
    fields = {
        name: {"status": "missing", "value": None, "evidence": [], "derivation": None}
        for name in TransferFields.model_fields
    }
    with pytest.raises(ValidationError):
        TypeAdapter(DocumentResult).validate_python({
            "case_id": "C1", "document_type": "invoice_draft", "fields": fields
        })


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_numeric_field_rejects_nonfinite_value(value: float) -> None:
    with pytest.raises(ValidationError):
        FieldResult[Number](status="fillable", value=value, evidence=["m1"], derivation=None)


@pytest.mark.parametrize("value", ["2026-02-30", "10.10.2026"])
def test_basis_date_rejects_invalid_date(value: str) -> None:
    with pytest.raises(ValidationError):
        FieldResult[ISODate](
            status="fillable", value=value, evidence=["m1"], derivation=None
        )


def test_zero_vat_rate_is_fillable_not_missing() -> None:
    field = FieldResult[Number](
        status="fillable", value=0, evidence=["m1"], derivation=None
    )
    assert field.value == 0
