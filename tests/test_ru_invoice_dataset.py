"""Преобразование внешнего ru-invoice-extraction-benchmark в формат пайплайна; без API."""

import hashlib
import json

import pytest

from scripts import build_ru_invoice_dataset as builder
from src.main import load_cases, load_labels


def source_rows():
    rows = []
    for split in ("development", "validation", "test"):
        path = builder.SOURCE / "data" / f"{split}.jsonl"
        rows += [{**json.loads(line), "source_split": split} for line in path.read_text(encoding="utf-8").splitlines()]
    return {row["id"]: row for row in rows}


@pytest.fixture(scope="module")
def dataset():
    cases = load_cases(builder.DATA / "cases_ru_invoice.jsonl", "all")
    labels = load_labels(builder.DATA / "labels_ru_invoice.jsonl", {case.case_id for case in cases})
    manifest = json.loads((builder.DATA / "ru_invoice.manifest.json").read_text(encoding="utf-8"))
    return {case.case_id: case for case in cases}, {row["case_id"]: row for row in labels}, manifest


def test_data_files_match_the_generator():
    for name, contents in builder.build_contents().items():
        assert (builder.DATA / name).read_text(encoding="utf-8") == contents, name


def test_source_files_are_the_pinned_revision():
    for name, digest in builder.SOURCE_SHA256.items():
        assert hashlib.sha256((builder.SOURCE / name).read_bytes()).hexdigest() == digest, name
    assert "Apache License" in (builder.SOURCE / "LICENSE").read_text(encoding="utf-8")


def test_every_source_invoice_becomes_one_user_message_with_the_document_verbatim(dataset):
    cases, _, manifest = dataset
    source = source_rows()
    assert list(cases) == list(source)
    for case_id, case in cases.items():
        assert case.split == ("test" if source[case_id]["source_split"] == "test" else "dev")
        assert [(m.id, m.role) for m in case.messages] == [("m1", "user")]
        assert case.messages[0].text == builder.INPUT_TEMPLATE.format(text=source[case_id]["text"])
    assert manifest["input_template"] == builder.INPUT_TEMPLATE
    for line in (builder.DATA / "cases_ru_invoice.jsonl").read_text(encoding="utf-8").splitlines():
        assert set(json.loads(line)) == {"case_id", "split", "messages"}


def test_external_fields_come_from_the_benchmark_labels(dataset):
    _, labels, _ = dataset
    for case_id, row in source_rows().items():
        expected, fields = row["expected"], labels[case_id]["fields"]
        assert labels[case_id]["document_type"] == "invoice_draft"
        assert fields["recipient_name"] == {"status": "fillable", "value": expected["buyer"]["name"],
                                            "evidence": ["m1"], "derivation": None}
        assert fields["recipient_inn"]["value"] == expected["buyer"]["tax_id"]
        rub = row["currency"] == "RUB"
        assert [(i["description"], i["quantity"], i["unit"], i["unit_price_rub"]) for i in fields["items"]["value"]] == [
            (i["description"], i["quantity"], i["unit"], i["unit_price"] if rub else None)
            for i in expected["line_items"] if i["unit_price"] >= 0]
        if rub:
            assert (fields["amount_rub"]["status"], fields["amount_rub"]["value"]) == ("fillable", expected["total_amount"])
        else:
            assert fields["amount_rub"] == {"status": "missing", "value": None, "evidence": [], "derivation": None}


def test_discount_lines_are_not_items_and_stay_only_in_the_total(dataset):
    """Как в пайплайне: скидка не товар и не услуга, она уже учтена в итоге; цена позиции не бывает отрицательной."""
    _, labels, manifest = dataset
    discounted = {case_id: row for case_id, row in source_rows().items()
                  if any(i["unit_price"] < 0 for i in row["expected"]["line_items"])}
    assert len(discounted) == 12
    for case_id, row in discounted.items():
        items = labels[case_id]["fields"]["items"]["value"]
        assert len(items) == sum(i["unit_price"] >= 0 for i in row["expected"]["line_items"])
        assert all(i["unit_price_rub"] is None or i["unit_price_rub"] >= 0 for i in items)
        if row["currency"] == "RUB":
            assert labels[case_id]["fields"]["amount_rub"]["value"] == row["expected"]["total_amount"]
    assert "discount_lines" in manifest["rules"]


@pytest.mark.parametrize("vat_mode, mode, rate", [
    ("standard_20", ("fillable", "rate"), ("fillable", 20)),
    ("reduced_10", ("fillable", "rate"), ("fillable", 10)),
    ("zero_rate", ("fillable", "rate"), ("fillable", 0)),
    ("exempt", ("fillable", "no_vat"), ("missing", None)),
    ("mixed_10_20", ("fillable", "rate"), ("ambiguous", None)),
])
def test_vat_mode_follows_the_output_contract(dataset, vat_mode, mode, rate):
    _, labels, _ = dataset
    ids = [case_id for case_id, row in source_rows().items() if row["vat_mode"] == vat_mode]
    assert ids
    for case_id in ids:
        fields = labels[case_id]["fields"]
        assert (fields["vat_mode"]["status"], fields["vat_mode"]["value"]) == mode
        assert (fields["vat_rate_percent"]["status"], fields["vat_rate_percent"]["value"]) == rate


def test_fields_absent_from_the_benchmark_are_placeholders_and_marked_unscored(dataset):
    _, labels, manifest = dataset
    assert set(manifest["external_fields"]) | set(manifest["placeholder_fields"]) == set(
        next(iter(labels.values()))["fields"])
    assert not set(manifest["external_fields"]) & set(manifest["placeholder_fields"])
    for label in labels.values():
        for name in manifest["placeholder_fields"]:
            assert label["fields"][name] == {"status": "missing", "value": None, "evidence": [], "derivation": None}


def test_manifest_keeps_source_metadata_for_separate_reporting(dataset):
    cases, _, manifest = dataset
    source = source_rows()
    assert manifest["source"]["revision"] == builder.SOURCE_REVISION
    assert manifest["source"]["license"] == "Apache-2.0"
    assert manifest["input_sha256"] == hashlib.sha256((builder.DATA / "cases_ru_invoice.jsonl").read_bytes()).hexdigest()
    assert manifest["labels_sha256"] == hashlib.sha256((builder.DATA / "labels_ru_invoice.jsonl").read_bytes()).hexdigest()
    meta = {row["case_id"]: row for row in manifest["cases"]}
    assert list(meta) == list(cases)
    for case_id, row in source.items():
        assert {key: meta[case_id][key] for key in ("source_split", "currency", "vat_mode", "template_family")} == {
            key: row[key] for key in ("source_split", "currency", "vat_mode", "template_family")}
    assert sum(row["currency"] != "RUB" for row in manifest["cases"]) == 5
