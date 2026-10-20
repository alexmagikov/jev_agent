"""Разметка следует общему контракту, а не особенностям ответов одной модели."""

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def labels():
    path = Path(__file__).resolve().parents[1] / "data/labels.jsonl"
    return {label["case_id"]: label for line in path.read_text().splitlines()
            if line.strip() for label in [json.loads(line)]}


@pytest.mark.parametrize("case_id,descriptions", [
    ("C001", ["консультационная услуга"]),
    ("C006", ["пирожок"]),
    ("C010", ["услуга"]),
    ("C012", ["услуга"]),
    ("C025", ["бумага", "ручка", "папка"]),
])
def test_ordinary_item_labels_use_singular_nominative(labels, case_id, descriptions):
    items = labels[case_id]["fields"]["items"]["value"]
    assert [item["description"] for item in items] == descriptions


@pytest.mark.parametrize("case_id,descriptions,quantities", [
    ("C014", ["стул", "стол"], [3, 2]),
    ("C026", ["Абрикосы"], [5]),
    ("C027", ["Краска", "Лента"], [1.5, 2.25]),
    ("C028", ["Красная краска", "Зелёная краска"], [None, None]),
])
def test_quoted_names_and_known_canonical_labels_are_preserved(
    labels, case_id, descriptions, quantities,
):
    items = labels[case_id]["fields"]["items"]["value"]
    assert [item["description"] for item in items] == descriptions
    assert [item["quantity"] for item in items] == quantities


def test_description_normalization_does_not_change_subject_or_physical_quantity(labels):
    for case_id, subject in (
        ("C001", "консультационные услуги"), ("C006", "пирожки"),
        ("C010", "услуги"), ("C012", "услуги"),
    ):
        fields = labels[case_id]["fields"]
        assert fields["subject"]["value"] == subject
        assert fields["items"]["value"][0]["quantity"] is None
    fields = labels["C025"]["fields"]
    assert fields["subject"]["status"] == "missing"
    assert [item["quantity"] for item in fields["items"]["value"]] == [2, 10, 5]
    assert [item["unit_price_rub"] for item in fields["items"]["value"]] == [500, 100, 200]
