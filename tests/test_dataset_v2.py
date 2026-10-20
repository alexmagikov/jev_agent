"""Контракт расширенного синтетического набора; никаких вызовов моделей."""

import json
from collections import Counter
from pathlib import Path

import pytest

from src.main import load_cases, load_labels
from src.metrics import calculate_metrics
from src.extraction import _invalid_format


DATA = Path(__file__).resolve().parents[1] / "data"


def rows(name):
    return [json.loads(line) for line in (DATA / name).read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def dataset():
    cases = load_cases(DATA / "cases_v2.jsonl", "all")
    labels = load_labels(DATA / "labels_v2.jsonl", {case.case_id for case in cases})
    return cases, {label["case_id"]: label for label in labels}


def test_v2_preserves_every_original_record_and_adds_exactly_forty():
    for original, extended in (("cases.jsonl", "cases_v2.jsonl"),
                               ("labels.jsonl", "labels_v2.jsonl")):
        before, after = rows(original), rows(extended)
        assert len(before) == 41 and len(after) == 81
        assert after[:len(before)] == before
        assert len({row["case_id"] for row in after}) == len(after)
        assert {row["case_id"] for row in after} == {f"C{index:03}" for index in range(1, 82)}


def test_v2_is_reproducible_and_inputs_do_not_contain_answers():
    from scripts.build_dataset_v2 import build_contents

    for name, contents in build_contents().items():
        assert (DATA / name).read_text(encoding="utf-8") == contents
    for case in rows("cases_v2.jsonl"):
        assert set(case) == {"case_id", "split", "messages"}
        assert all(set(message) == {"id", "role", "text"} for message in case["messages"])


def test_new_cases_have_complete_labels_user_evidence_and_balanced_splits(dataset):
    cases, labels = dataset
    assert len(cases) == len(labels) == 81
    new_cases = [case for case in cases if int(case.case_id[1:]) >= 42]
    assert Counter(case.split for case in new_cases) == {"dev": 20, "test": 20}
    assert Counter(labels[case.case_id]["document_type"] for case in new_cases) == {
        "invoice_draft": 24, "transfer_draft": 10, "none": 6,
    }
    for case in cases:
        assert set(labels[case.case_id]) <= {"case_id", "document_type", "fields", "reason"}
        user_ids = {message.id for message in case.messages if message.role == "user"}
        for field in labels[case.case_id]["fields"].values():
            assert set(field["evidence"]) <= user_ids
            assert len(field["evidence"]) == len(set(field["evidence"]))
            if field["status"] in ("fillable", "ambiguous", "invalid"):
                assert field["evidence"]
            else:
                assert field["value"] is None and field["evidence"] == []
        for name, field in labels[case.case_id]["fields"].items():
            if field["status"] == "fillable":
                assert not _invalid_format(name, field["value"]), (case.case_id, name)


@pytest.mark.parametrize("split,count", [("dev", 48), ("test", 33), ("all", 81)])
def test_v2_works_with_existing_loaders_and_metrics(split, count):
    cases = load_cases(DATA / "cases_v2.jsonl", split)
    assert len(cases) == count
    labels = load_labels(DATA / "labels_v2.jsonl", {case.case_id for case in cases})
    report = calculate_metrics(labels, labels)
    assert report.total_cases == count
    assert report.document_type_accuracy == 1
    assert report.fields.value_accuracy == 1
    assert report.fields.evidence_accuracy == 1


@pytest.mark.parametrize("case_id,field,status,expected", [
    ("C042", "recipient_inn", "fillable", "7701234501"),
    ("C043", "amount_rub", "fillable", 11000),
    ("C044", "recipient_inn", "missing", None),
    ("C044", "vat_rate_percent", "missing", None),
    ("C045", "recipient_email", "fillable", "new@example.com"),
    ("C046", "recipient_inn", "invalid", None),
    ("C046", "recipient_account", "invalid", None),
    ("C046", "recipient_bik", "invalid", None),
    ("C047", "vat_rate_percent", "fillable", 0),
    ("C048", "amount_rub", "missing", None),
    ("C049", "amount_rub", "fillable", 4500),
    ("C050", "amount_rub", "fillable", 1060),
    ("C051", "amount_rub", "fillable", 480),
    ("C052", "amount_rub", "fillable", 400),
    ("C053", "basis_number", "fillable", "0007-А/26"),
    ("C053", "basis_date", "fillable", "2026-10-05"),
    ("C054", "amount_rub", "fillable", 9000),
    ("C055", "recipient_name", "fillable", "ООО Рассвет"),
    ("C056", "recipient_account", "fillable", "40702810000000000056"),
    ("C057", "recipient_name", "ambiguous", None),
    ("C057", "recipient_inn", "ambiguous", None),
    ("C058", "recipient_email", "invalid", None),
    ("C059", "amount_rub", "ambiguous", None),
    ("C060", "amount_rub", "fillable", 1600),
    ("C061", "amount_rub", "fillable", 450),
    ("C062", "amount_rub", "fillable", 12000),
    ("C063", "amount_rub", "fillable", 2500),
    ("C064", "amount_rub", "fillable", 27000),
    ("C065", "amount_rub", "fillable", 3200),
    ("C066", "amount_rub", "fillable", 8500),
    ("C067", "destination_card_masked", "missing", None),
    ("C067", "destination_bank", "missing", None),
    ("C068", "destination_card_masked", "ambiguous", None),
    ("C068", "destination_bank", "ambiguous", None),
    ("C069", "amount_rub", "missing", None),
    ("C070", "amount_rub", "fillable", 2345.67),
    ("C071", "destination_card_masked", "fillable", "****7171"),
    ("C072", "destination_card_masked", "missing", None),
    ("C073", "amount_rub", "fillable", 3000),
    ("C074", "destination_card_masked", "fillable", "2202 **** **** 9090"),
    ("C075", "destination_card_masked", "fillable", "****7575"),
])
def test_new_scenario_decisions(dataset, case_id, field, status, expected):
    actual = dataset[1][case_id]["fields"][field]
    assert actual["status"] == status
    assert actual["value"] == expected


def test_item_normalization_calculations_and_unknowns(dataset):
    labels = dataset[1]

    def items(case_id):
        return labels[case_id]["fields"]["items"]["value"]

    assert items("C048") == [dict(description="песок", quantity=None, unit="кг", unit_price_rub=85)]
    assert items("C049") == [dict(description="консультация", quantity=6, unit=None, unit_price_rub=750)]
    assert [item["description"] for item in items("C050")] == ["ножницы", "Листья"]
    assert items("C051")[0]["quantity"] == 2 * 3
    assert items("C052") == [dict(description="папка", quantity=4, unit="штуки", unit_price_rub=100)]
    assert all(item["quantity"] is None and item["unit_price_rub"] is None for item in items("C060"))
    assert all(item["quantity"] is None and item["unit_price_rub"] == 150 for item in items("C061"))
    assert items("C063")[0]["description"] == 'Набор "Лето"'
    assert labels["C050"]["fields"]["amount_rub"]["value"] == 2 * 350 + 3 * 120
    assert labels["C061"]["fields"]["amount_rub"]["value"] == 150 + 150 + 150
    assert labels["C064"]["fields"]["amount_rub"]["value"] == 3 * 9000


@pytest.mark.parametrize("case_id", [f"C{index:03}" for index in range(76, 82)])
def test_none_cases_do_not_invent_document_fields(dataset, case_id):
    label = dataset[1][case_id]
    assert label["document_type"] == "none"
    assert label["fields"] == {}
    assert label["reason"].strip()
