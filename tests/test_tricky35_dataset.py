"""Новые длинные сценарии, согласованность эталонов и совместимость CLI без API."""

import json
from collections import Counter
from copy import deepcopy
from decimal import Decimal

import pytest

from scripts import build_tricky35_dataset as builder
from src.extraction import ExtractionProposal, invalid_format, prepare_fields
from src.main import load_cases, load_labels
from src.metrics import calculate_metrics


@pytest.fixture(scope="module")
def dataset():
    cases = load_cases(builder.DATA / "cases_tricky35.jsonl", "all")
    labels = load_labels(builder.DATA / "labels_tricky35.jsonl", {case.case_id for case in cases})
    return cases, {row["case_id"]: row for row in labels}


def test_all_thirty_five_dialogues_are_new_long_and_have_separate_answers(dataset):
    cases, labels = dataset
    old = load_cases(builder.DATA / "cases_v2.jsonl", "all")
    assert len(cases) == len(labels) == 35
    assert {case.case_id for case in cases} == {f"T{i:03}" for i in range(1, 36)}
    assert not {case.case_id for case in cases} & {case.case_id for case in old}
    assert not {tuple(m.text for m in case.messages) for case in cases} & {
        tuple(m.text for m in case.messages) for case in old
    }
    assert Counter(case.split for case in cases) == {"dev": 18, "test": 17}
    assert Counter(row["document_type"] for row in labels.values()) == {
        "invoice_draft": 20, "transfer_draft": 10, "none": 5,
    }
    for row in map(json.loads, (builder.DATA / "cases_tricky35.jsonl").read_text().splitlines()):
        assert set(row) == {"case_id", "split", "messages"}
        assert len(row["messages"]) >= 7
        assert sum(len(m["text"]) for m in row["messages"]) >= 1100, row["case_id"]
        assert all(set(m) == {"id", "role", "text"} for m in row["messages"])


def test_labels_use_only_user_sources_and_valid_field_contracts(dataset):
    cases, labels = dataset
    for case in cases:
        user_ids = {m.id for m in case.messages if m.role == "user"}
        for name, field in labels[case.case_id]["fields"].items():
            assert set(field["evidence"]) <= user_ids, (case.case_id, name)
            assert len(field["evidence"]) == len(set(field["evidence"]))
            if field["status"] == "missing":
                assert field["value"] is None and field["evidence"] == []
            else:
                assert field["evidence"], (case.case_id, name)
            if field["status"] == "fillable":
                assert not invalid_format(name, field["value"]), (case.case_id, name)
            else:
                assert field["value"] is None and field["derivation"] is None


@pytest.mark.parametrize("split,count", [("dev", 18), ("test", 17), ("all", 35)])
def test_cli_loaders_and_metrics_accept_new_files(split, count):
    cases = load_cases(builder.DATA / "cases_tricky35.jsonl", split)
    labels = load_labels(builder.DATA / "labels_tricky35.jsonl", {case.case_id for case in cases})
    report = calculate_metrics(labels, labels)
    assert report.total_cases == report.document_type_correct == count
    assert report.fields.value_accuracy == report.fields.evidence_id_exact_match == 1


def test_generator_reproduces_committed_artifacts_without_models():
    contents = builder.build_contents()
    assert set(contents) == {"cases_tricky35.jsonl", "labels_tricky35.jsonl", "tricky35.md"}
    for name, text in contents.items():
        assert (builder.DATA / name).read_text(encoding="utf-8") == text


@pytest.mark.parametrize("case_id,expected", [
    ("T001", Decimal(4) * 2400),
    ("T002", Decimal(2) * 45000 + Decimal(2) * 10500 - 6000),
    ("T005", Decimal(2) * Decimal("1499.50")),
    ("T012", (Decimal("0.75") * 1280 + Decimal("2.5") * 96) * Decimal("0.90")),
    ("T013", Decimal(12000) + 8500),
    ("T016", Decimal(3) * 290 + Decimal(2) * 1750),
    ("T017", Decimal(10) * 80 + Decimal(5) * 95 - 25),
    ("T020", Decimal(6) * 280 + Decimal(2) * 600),
    ("T029", Decimal("18250.50") - Decimal("5250.50")),
])
def test_hand_labelled_totals_follow_final_operands_not_decoys(dataset, case_id, expected):
    field = dataset[1][case_id]["fields"]["amount_rub"]
    assert field["status"] == "fillable"
    assert Decimal(str(field["value"])) == expected
    assert field["derivation"]


@pytest.mark.parametrize("case_id,name,status,value", [
    ("T003", "recipient_inn", "invalid", None),
    ("T003", "recipient_bik", "invalid", None),
    ("T003", "recipient_email", "fillable", "orders@format.example"),
    ("T004", "recipient_inn", "missing", None),
    ("T006", "amount_rub", "missing", None),
    ("T007", "subject", "missing", None),
    ("T008", "amount_rub", "fillable", 12000),
    ("T009", "vat_rate_percent", "fillable", 0),
    ("T009", "basis_date", "fillable", "2026-11-02"),
    ("T010", "vat_rate_percent", "missing", None),
    ("T011", "recipient_name", "ambiguous", None),
    ("T014", "recipient_email", "missing", None),
    ("T015", "basis_date", "fillable", "2028-02-29"),
    ("T018", "amount_rub", "ambiguous", None),
    ("T020", "vat_mode", "missing", None),
    ("T022", "destination_card_masked", "ambiguous", None),
    ("T023", "destination_card_masked", "fillable", "****7822"),
    ("T024", "destination_card_masked", "missing", None),
    ("T025", "destination_card_masked", "missing", None),
    ("T026", "amount_rub", "fillable", 2700),
    ("T028", "destination_card_masked", "fillable", "0000 **** **** 0099"),
    ("T030", "amount_rub", "missing", None),
])
def test_key_traps_have_explicit_expected_decisions(dataset, case_id, name, status, value):
    field = dataset[1][case_id]["fields"][name]
    assert (field["status"], field["value"]) == (status, value)


def test_item_associations_and_quotes_survive_normalization(dataset):
    labels = dataset[1]
    assert labels["T005"]["fields"]["items"]["value"][0]["description"] == 'Набор "Тихий вечер"'
    assert labels["T007"]["fields"]["items"]["value"][1]["quantity"] == 8
    assert labels["T014"]["fields"]["items"]["value"][0]["unit_price_rub"] == 700
    assert [item["description"] for item in labels["T016"]["fields"]["items"]["value"]] == ["ножницы", "Весы"]
    assert [item["unit_price_rub"] for item in labels["T017"]["fields"]["items"]["value"]] == [80, 95]
    for case_id in (f"T{i:03}" for i in range(31, 36)):
        assert labels[case_id]["document_type"] == "none"
        assert labels[case_id]["fields"] == {} and labels[case_id]["reason"].strip()


def plan(expression, **operands):
    return {"expression": expression, "operands": [
        {"name": name, "source": {"message_id": mid, "quote": quote}}
        for name, (mid, quote) in operands.items()
    ]}


# Independent extraction fixtures prove that the hand-written gold values can
# pass the existing quote/unit/number/calculation checks without changing Python.
PLANS = {
    "T001": {"amount_rub": plan("hours * price", hours=("m5", "4"), price=("m5", "2400"))},
    "T002": {"amount_rub": plan("n * p + m * q - discount", n=("m1", "2"), p=("m1", "45000"),
                               m=("m5", "2"), q=("m3", "10500"), discount=("m7", "6000"))},
    "T005": {"amount_rub": plan("count * price", count=("m5", "2"), price=("m1", "1499,50"))},
    "T007": {"items.1.quantity": plan("total - pens", total=("m1", "12"), pens=("m3", "4"))},
    "T012": {"amount_rub": plan("(a * b + c * d) * (1 - discount / 100)",
                               a=("m1", "0,75"), b=("m1", "1280"), c=("m1", "2,5"),
                               d=("m1", "96"), discount=("m5", "10"))},
    "T013": {"amount_rub": plan("mount + setup", mount=("m1", "12000"), setup=("m1", "8500"))},
    "T014": {"items.0.unit_price_rub": plan("total / count", total=("m1", "5600"), count=("m3", "8"))},
    "T016": {"amount_rub": plan("n * p + m * q", n=("m1", "3"), p=("m1", "290"),
                               m=("m3", "2"), q=("m3", "1750"))},
    "T017": {"amount_rub": plan("n * p + m * q - discount", n=("m1", "10"), p=("m1", "80"),
                               m=("m1", "5"), q=("m1", "95"), discount=("m5", "25"))},
    "T020": {"amount_rub": plan("n * p + m * q", n=("m1", "6"), p=("m1", "280"),
                               m=("m1", "2"), q=("m1", "600"))},
    "T029": {"amount_rub": plan("(balance + cents / 100) - (reserve + reserve_cents / 100)",
                               balance=("m3", "18250"), cents=("m3", "50"),
                               reserve=("m5", "5250"), reserve_cents=("m5", "50"))},
}


@pytest.mark.parametrize("case_id", [f"T{i:03}" for i in range(1, 31)])
def test_gold_document_has_a_valid_extraction_path_through_python(dataset, case_id):
    cases, labels = dataset
    case = next(case for case in cases if case.case_id == case_id)
    label = labels[case_id]
    messages = {message.id: message.text for message in case.messages}
    fields = {name: {"status": field["status"], "value": deepcopy(field["value"]),
                     "evidence": [{"message_id": mid, "quote": messages[mid]} for mid in field["evidence"]]}
              for name, field in label["fields"].items()}
    for path, calculation in PLANS.get(case_id, {}).items():
        if path.startswith("items."):
            _, index, attribute = path.split(".")
            target = fields["items"]["value"][int(index)]
            target[attribute] = None
            target[{"quantity": "quantity_calculation", "unit_price_rub": "unit_price_calculation"}[attribute]] = calculation
        else:
            fields[path].update(value=None, calculation=calculation)
    proposal = ExtractionProposal.model_validate({"fields": fields})
    actual = prepare_fields(proposal, case.messages, label["document_type"]).model_dump(mode="json")
    for name, field in label["fields"].items():
        assert actual[name]["status"] == field["status"], (case_id, name)
        assert actual[name]["value"] == field["value"], (case_id, name)
        assert set(actual[name]["evidence"]) == set(field["evidence"]), (case_id, name)
        assert bool(actual[name]["derivation"]) == bool(field["derivation"]), (case_id, name)
