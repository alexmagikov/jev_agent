"""Стресс-выборка — точное подмножество v2, без новых ответов и вызовов моделей."""

import json
from collections import Counter

import pytest

from scripts import build_hard30_dataset as builder
from src.extraction import validate_messages
from src.main import load_cases, load_labels
from src.metrics import calculate_metrics


EXPECTED_IDS = (
    "C004", "C026", "C027", "C042", "C043", "C044", "C046", "C047", "C049", "C051",
    "C052", "C053", "C055", "C057", "C059", "C060", "C063", "C064", "C065", "C066",
    "C067", "C068", "C070", "C072", "C073", "C074", "C075", "C076", "C078", "C081",
)


def rows(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_selection_is_frozen_unique_and_explains_every_case():
    assert tuple(builder.SELECTION) == EXPECTED_IDS
    assert len(builder.SELECTION) == 30
    assert all(reason.strip() for reason in builder.SELECTION.values())


def test_generated_files_are_exact_source_rows_with_matching_labels_and_no_leakage():
    contents = builder.build_contents()
    assert set(contents) == {"cases_hard30.jsonl", "labels_hard30.jsonl"}
    for target, source in (("cases_hard30.jsonl", "cases_v2.jsonl"),
                           ("labels_hard30.jsonl", "labels_v2.jsonl")):
        original = (builder.DATA / source).read_text()
        expected = "".join(line for line in original.splitlines(keepends=True)
                           if line.strip() and json.loads(line)["case_id"] in EXPECTED_IDS)
        assert contents[target] == expected
        assert (builder.DATA / target).read_text() == expected
        records = rows(contents[target])
        assert len(records) == 30
        assert tuple(row["case_id"] for row in records) == EXPECTED_IDS
    for case in rows(contents["cases_hard30.jsonl"]):
        assert set(case) == {"case_id", "split", "messages"}
        assert all(set(message) == {"id", "role", "text"} for message in case["messages"])


def test_real_loaders_sources_splits_and_metrics_work_for_all_thirty():
    cases = load_cases(builder.DATA / "cases_hard30.jsonl", "all")
    labels = load_labels(builder.DATA / "labels_hard30.jsonl", {case.case_id for case in cases})
    assert Counter(case.split for case in cases) == {"dev": 17, "test": 13}
    assert Counter(label["document_type"] for label in labels) == {
        "invoice_draft": 19, "transfer_draft": 8, "none": 3,
    }
    by_id = {label["case_id"]: label for label in labels}
    for case in cases:
        validate_messages(case.messages)
        user_ids = {message.id for message in case.messages if message.role == "user"}
        for field in by_id[case.case_id]["fields"].values():
            assert set(field["evidence"]) <= user_ids
    report = calculate_metrics(labels, labels)
    assert report.total_cases == report.document_type_correct == 30
    assert report.fields.value_accuracy == report.fields.evidence_id_exact_match == 1


def test_unknown_selected_id_is_rejected_instead_of_silently_producing_fewer_cases(monkeypatch):
    selection = dict(builder.SELECTION)
    selection["UNKNOWN"] = selection.pop("C004")
    monkeypatch.setattr(builder, "SELECTION", selection)
    with pytest.raises(ValueError, match="UNKNOWN"):
        builder.build_contents()


def test_selection_must_have_exactly_thirty_cases(monkeypatch):
    monkeypatch.setattr(builder, "SELECTION", {"C004": "Исправление"})
    with pytest.raises(ValueError, match="30"):
        builder.build_contents()
