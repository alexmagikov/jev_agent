"""Контракт новой смешанной выборки; без запросов к моделям."""

from collections import Counter
from decimal import Decimal
import hashlib
import json

import pytest

from scripts import build_evaluation50_dataset as builder
from src.extraction import ExtractionProposal, invalid_format, prepare_fields
from src.main import load_cases, load_labels


@pytest.fixture(scope="module")
def dataset():
    cases = load_cases(builder.DATA / "cases_evaluation50.jsonl", "test")
    labels = load_labels(builder.DATA / "labels_evaluation50.jsonl", {case.case_id for case in cases})
    manifest = json.loads((builder.DATA / "evaluation50.manifest.json").read_text())
    return cases, {row["case_id"]: row for row in labels}, manifest


def test_new_mixed_scenarios_are_balanced_interleaved_and_not_old_dialogues(dataset):
    cases, labels, manifest = dataset
    assert len(cases) == len(labels) == 50
    assert [case.case_id for case in cases] == [f"E{i:03}" for i in range(1, 51)]
    assert all(case.split == "test" for case in cases)
    assert manifest["human_reviewed"] is False
    assert manifest["input_sha256"] == hashlib.sha256((builder.DATA / "cases_evaluation50.jsonl").read_bytes()).hexdigest()
    assert manifest["labels_sha256"] == hashlib.sha256((builder.DATA / "labels_evaluation50.jsonl").read_bytes()).hexdigest()
    groups = {row["case_id"]: row["group"] for row in manifest["cases"]}
    assert Counter(groups.values()) == {"ordinary": 25, "tricky": 25}
    assert [groups[case.case_id] for case in cases[:6]] == ["ordinary", "tricky"] * 3
    for group in ("ordinary", "tricky"):
        assert Counter(labels[case.case_id]["document_type"] for case in cases if groups[case.case_id] == group) == {
            "invoice_draft": 12, "transfer_draft": 8, "none": 5,
        }
    dialogues = {tuple((m.role, " ".join(m.text.casefold().split())) for m in case.messages) for case in cases}
    for name in ("cases.jsonl", "cases_v2.jsonl", "cases_tricky35.jsonl"):
        old = load_cases(builder.DATA / name, "all")
        assert not {case.case_id for case in old} & set(labels)
        assert not dialogues & {tuple((m.role, " ".join(m.text.casefold().split())) for m in case.messages) for case in old}


def test_case_inputs_have_no_labels_or_difficulty_hints_and_evidence_is_user_only(dataset):
    cases, labels, _ = dataset
    for line in (builder.DATA / "cases_evaluation50.jsonl").read_text().splitlines():
        assert set(json.loads(line)) == {"case_id", "split", "messages"}
    for case in cases:
        user_ids = {m.id for m in case.messages if m.role == "user"}
        for name, field in labels[case.case_id]["fields"].items():
            assert set(field["evidence"]) <= user_ids
            if field["status"] == "missing":
                assert field["evidence"] == []
            else:
                assert field["evidence"], (case.case_id, name)
            if field["status"] == "fillable":
                assert not invalid_format(name, field["value"]), (case.case_id, name)


@pytest.mark.parametrize("case_id,amount", [
    ("E001", Decimal(3) * 1250),
    ("E006", Decimal(2) * 1700 + Decimal(3) * 450),
    ("E007", Decimal(7) * 180 + Decimal(2) * 310),
    ("E012", (Decimal(3) * 260 + Decimal(4) * 180) * Decimal("0.9") + 1900),
    ("E018", Decimal("28120.50") - Decimal("8350.50")),
    ("E023", Decimal(2) * 2450),
    ("E029", Decimal("1.25") * 1600),
    ("E031", Decimal(4) * 2400),
    ("E036", Decimal(2) * 5000),
    ("E040", Decimal(18400) * 25 / 100),
    ("E049", Decimal(25) * Decimal("48.5")),
])
def test_arithmetic_uses_final_values_and_correct_discount_scope(dataset, case_id, amount):
    field = dataset[1][case_id]["fields"]["amount_rub"]
    assert Decimal(str(field["value"])) == amount
    assert field["derivation"]


@pytest.mark.parametrize("case_id,name,status,value", [
    ("E004", "destination_bank", "fillable", "ВТБ"),
    ("E004", "destination_card_masked", "fillable", "****9048"),
    ("E008", "recipient_name", "ambiguous", None),
    ("E008", "recipient_inn", "ambiguous", None),
    ("E010", "destination_card_masked", "missing", None),
    ("E010", "amount_rub", "fillable", 9400),
    ("E011", "vat_mode", "fillable", "rate"),
    ("E011", "vat_rate_percent", "fillable", 0),
    ("E016", "recipient_inn", "invalid", None),
    ("E016", "recipient_bik", "invalid", None),
    ("E016", "basis_date", "invalid", None),
    ("E016", "recipient_email", "fillable", "pay@teplosfera.example"),
    ("E022", "recipient_relation", "fillable", "self"),
    ("E022", "destination_bank", "missing", None),
    ("E022", "destination_card_masked", "missing", None),
    ("E027", "amount_rub", "missing", None),
    ("E028", "destination_card_masked", "fillable", "2200 **** **** 0073"),
    ("E030", "amount_rub", "fillable", 3100),
    ("E030", "vat_rate_percent", "fillable", 20),
    ("E030", "basis_date", "fillable", "2025-06-12"),
    ("E032", "recipient_inn", "fillable", "7714123456"),
    ("E032", "recipient_account", "missing", None),
    ("E032", "recipient_bik", "missing", None),
    ("E034", "destination_bank", "fillable", "Альфа-Банк"),
    ("E034", "destination_card_masked", "fillable", "****2280"),
    ("E037", "amount_rub", "missing", None),
    ("E039", "source_account_kind", "missing", None),
    ("E042", "basis_number", "fillable", "15-П"),
    ("E042", "basis_date", "invalid", None),
    ("E043", "recipient_inn", "fillable", "771234567890"),
    ("E043", "basis_date", "fillable", "2026-03-03"),
    ("E045", "destination_bank", "missing", None),
    ("E046", "amount_rub", "fillable", 12000),
    ("E048", "amount_rub", "ambiguous", None),
])
def test_scope_status_and_ownership_labels(dataset, case_id, name, status, value):
    actual = dataset[1][case_id]["fields"][name]
    assert (actual["status"], actual["value"]) == (status, value)


@pytest.mark.parametrize("case_id,name,evidence", [
    ("E002", "recipient_name", ["m1", "m3"]),
    ("E010", "amount_rub", ["m3", "m5"]),
    ("E010", "destination_bank", ["m1", "m3"]),
    ("E022", "amount_rub", ["m1", "m3"]),
    ("E028", "recipient_relation", ["m1", "m3", "m5"]),
    ("E030", "basis_number", ["m3", "m5"]),
])
def test_evidence_lists_every_user_message_stating_the_current_value(dataset, case_id, name, evidence):
    assert dataset[1][case_id]["fields"][name]["evidence"] == evidence


def test_unknown_attributes_and_unselected_goods_are_not_invented(dataset):
    labels = dataset[1]
    assert labels["E012"]["fields"]["items"]["value"][2] == {
        # ITEM_DESCRIPTION_RULE сохраняет характеристики: «Настройка всей поставки».
        "description": "настройка всей поставки", "quantity": None, "unit": None, "unit_price_rub": None,
    }
    assert len(labels["E020"]["fields"]["items"]["value"]) == 1
    assert labels["E020"]["fields"]["amount_rub"]["value"] == 6400
    assert all(item["unit_price_rub"] is None for item in labels["E030"]["fields"]["items"]["value"])
    assert labels["E036"]["fields"]["items"]["value"] == [
        {"description": "стол", "quantity": 2, "unit": "шт.", "unit_price_rub": 5000},
    ]
    assert labels["E037"]["fields"]["items"]["value"][0]["quantity"] is None
    assert labels["E048"]["fields"]["items"]["value"][0]["unit_price_rub"] is None
    for cid in ("E013", "E014", "E025", "E026", "E035", "E038", "E041", "E044", "E047", "E050"):
        assert labels[cid]["document_type"] == "none" and labels[cid]["fields"] == {}


def test_generator_and_artifacts_are_reproducible():
    assert all((builder.DATA / name).read_text(encoding="utf-8") == contents
               for name, contents in builder.build_contents().items())


def test_every_supported_label_has_a_python_valid_witness_with_sourced_calculations(dataset):
    plans = {
        "E001": ("q * price", [("q", "m1", "3"), ("price", "m1", "1250")]),
        "E006": ("days_p * price_p + days_s * price_s", [("days_p", "m3", "2"), ("price_p", "m1", "1700"),
                  ("days_s", "m1", "3"), ("price_s", "m5", "450")]),
        "E007": ("q1 * p1 + q2 * p2", [("q1", "m1", "7"), ("p1", "m1", "180"), ("q2", "m1", "2"), ("p2", "m1", "310")]),
        "E012": ("(q1 * p1 + q2 * p2) * (1 - discount / 100) + work", [
            ("q1", "m1", "3"), ("p1", "m1", "260"), ("q2", "m1", "4"), ("p2", "m1", "180"),
            ("discount", "m3", "10"), ("work", "m1", "1900")]),
        "E018": ("balance - reserve", [("balance", "m5", "28120,50"), ("reserve", "m3", "8350,50")]),
        "E023": ("q * price", [("q", "m1", "2"), ("price", "m1", "2450")]),
        "E024": ("q * price", [("q", "m5", "3"), ("price", "m1", "1850")]),
        "E029": ("q * price", [("q", "m1", "1,25"), ("price", "m1", "1600")]),
        "E031": ("q * price", [("q", "m1", "4"), ("price", "m1", "2400")]),
        "E036": ("q * price", [("q", "m1", "2"), ("price", "m1", "5000")]),
        "E040": ("balance * pct / 100", [("balance", "m3", "18 400"), ("pct", "m1", "25")]),
        "E049": ("q * price", [("q", "m1", "25"), ("price", "m1", "48,5")]),
    }
    for case in dataset[0]:
        label = dataset[1][case.case_id]
        if label["document_type"] == "none":
            continue
        messages = {m.id: m.text for m in case.messages}
        fields = {name: {"status": value["status"], "value": value["value"], "calculation": None,
                        "evidence": [{"message_id": mid, "quote": messages[mid]} for mid in value["evidence"]]}
                  for name, value in label["fields"].items()}
        if case.case_id in plans:
            expression, operands = plans[case.case_id]
            fields["amount_rub"].update(value=None, calculation={"expression": expression, "operands": [
                {"name": name, "source": {"message_id": mid, "quote": quote}} for name, mid, quote in operands]})
        checked = prepare_fields(ExtractionProposal.model_validate({"fields": fields}), case.messages,
                                 label["document_type"]).model_dump(mode="json")
        assert all((actual["status"], actual["value"]) == (label["fields"][name]["status"], label["fields"][name]["value"])
                   for name, actual in checked.items()), case.case_id
