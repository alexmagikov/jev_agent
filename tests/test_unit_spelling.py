"""Единицы измерения сохраняются как в источнике, включая точку сокращения."""

import json
from pathlib import Path

import pytest

from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionProposal, prepare_fields
from src.field_rules import ITEM_UNIT_RULE
from src.llm import build_extraction_prompt
from src.metrics import calculate_metrics
from src.schemas.document_schemas import InvoiceFields
from src.verification import build_verification_questions


DATA = Path(__file__).resolve().parents[1] / "data"


def record(filename):
    return next(row for line in (DATA / filename).read_text().splitlines()
                if (row := json.loads(line))["case_id"] == "C060")


@pytest.mark.parametrize("filename", ["labels_v2.jsonl", "labels_hard30.jsonl"])
def test_c060_labels_preserve_explicit_unit_dot_in_both_datasets(filename):
    label = record(filename)
    assert [item["unit"] for item in label["fields"]["items"]["value"]] == ["шт.", "шт."]


@pytest.mark.parametrize("unit", ["шт.", "шт", "кг", "комплекта"])
def test_llm_and_jev_share_unit_spelling_and_python_does_not_rewrite_it(unit):
    messages = [DocumentMessage(id="m1", role="user", text=f"Бумага: 2 {unit} по 150 рублей")]
    fields = {name: {"status": "missing", "value": None, "evidence": []}
              for name in InvoiceFields.model_fields}
    fields["items"] = {"status": "fillable", "value": [{
        "description": "бумага", "quantity": 2, "unit": unit, "unit_price_rub": 150,
    }], "evidence": [{"message_id": "m1", "quote": messages[0].text}]}
    candidate = ExtractionProposal.model_validate({"fields": fields})
    prepared = prepare_fields(candidate, messages, "invoice_draft")
    questions = build_verification_questions(prepared, proposal=candidate, messages=messages).root
    q = questions["items[0].unit::value_wrong"]
    assert prepared.items.value[0].unit == unit
    assert q.instructions["field"]["value"] == unit
    assert q.instructions["field"]["evidence"][0]["quote"] == messages[0].text
    assert ITEM_UNIT_RULE in q.instructions["rules"]
    assert ITEM_UNIT_RULE in build_extraction_prompt("invoice_draft")
    assert "шт. stays шт." in ITEM_UNIT_RULE


def test_correct_c060_candidate_with_source_units_matches_strict_value_metrics():
    case = record("cases_v2.jsonl")
    messages = [DocumentMessage.model_validate(m) for m in case["messages"]]
    fields = {name: {"status": "missing", "value": None, "evidence": []}
              for name in InvoiceFields.model_fields}
    values = {"recipient_name": "ООО Класс", "amount_rub": 1600, "items": [
        {"description": "ручка", "quantity": None, "unit": "шт.", "unit_price_rub": None},
        {"description": "маркер", "quantity": None, "unit": "шт.", "unit_price_rub": None},
    ]}
    for name, value in values.items():
        fields[name] = {"status": "fillable", "value": value,
                        "evidence": [{"message_id": "m1", "quote": messages[0].text}]}
    candidate = ExtractionProposal.model_validate({"fields": fields})
    prepared = prepare_fields(candidate, messages, "invoice_draft")
    prediction = {"case_id": "C060", "document_type": "invoice_draft",
                  "fields": prepared.model_dump(mode="json")}
    report = calculate_metrics([record("labels_v2.jsonl")], [prediction])
    assert report.fields.true_positives == 3
    assert report.fields.value_accuracy == 1
    assert report.fields.false_positives == report.fields.false_negatives == 0
