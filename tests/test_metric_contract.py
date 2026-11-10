"""Контракт оценки значений, всех статусов и документов, включая потерянные результаты."""

from copy import deepcopy

import pytest

from src.main import print_metrics
from src.metrics import calculate_metrics
from tests.test_metrics import field


def row(fields, case_id="C1", kind="invoice_draft"):
    return {"case_id": case_id, "document_type": kind, "fields": fields}


def test_wrong_value_is_not_an_overfill_but_reduces_value_and_document_scores():
    target = row({"amount_rub": field("fillable", 1000, ["m1"])})
    prediction = row({"amount_rub": field("fillable", 9000, ["m1"])})
    report = calculate_metrics([target], [prediction])
    scores = report.fields
    assert scores.precision == scores.recall == 1
    assert scores.overfill_rate == 0
    assert scores.value_precision == scores.value_recall == 0
    assert scores.correct_values == 0 and scores.expected_values == scores.predicted_values == 1
    assert scores.status_accuracy == 1 and scores.field_accuracy == 0
    assert report.exact_document_correct == 0 and report.exact_document_accuracy == 0
    assert report.prediction_coverage == 1
    serialized = scores.model_dump()
    assert "unsupported_fill_rate" not in serialized and "evidence_accuracy" not in serialized
    assert serialized["evidence_id_exact_match"] == 1  # ID совпали, несмотря на неверное число.


def test_value_recall_includes_omitted_values_but_conditional_accuracy_does_not():
    target = row({"amount_rub": field("fillable", 100),
                  "subject": field("fillable", "работа"), "basis_number": field("fillable", "B-7")})
    prediction = deepcopy(target)
    prediction["fields"]["subject"] = field("missing")
    prediction["fields"]["basis_number"]["value"] = "B-8"
    scores = calculate_metrics([target], [prediction]).fields
    assert scores.correct_values == 1 and scores.expected_values == 3 and scores.predicted_values == 2
    assert scores.value_accuracy == scores.value_precision == .5
    assert scores.value_recall == pytest.approx(1 / 3)
    assert scores.recall == pytest.approx(2 / 3)


def test_all_expected_values_can_be_right_while_extra_filling_makes_document_wrong():
    target = row({"amount_rub": field("fillable", 100), "subject": field("missing")})
    prediction = row({"amount_rub": field("fillable", 100), "subject": field("fillable", "работа")})
    report = calculate_metrics([target], [prediction])
    assert report.fields.correct_values == report.fields.expected_values == 1
    assert report.fields.value_accuracy == report.fields.value_recall == 1
    assert report.fields.value_precision == report.fields.overfill_rate == .5
    assert report.fields.status_accuracy == report.fields.field_accuracy == .5
    assert report.exact_document_accuracy == 0


def test_same_number_in_wrong_template_is_not_a_correct_value():
    target = row({"amount_rub": field("fillable", 100)})
    prediction = row(target["fields"], kind="transfer_draft")
    report = calculate_metrics([target], [prediction])
    assert report.fields.value_precision == report.fields.value_recall == 0
    assert report.fields.status_confusion["fillable"]["wrong_document_type"] == 1
    assert report.by_document_type["invoice_draft"].false_negatives == 1
    assert report.by_document_type["transfer_draft"].false_positives == 1
    assert report.prediction_coverage == 1 and report.exact_document_accuracy == 0


STATUSES = ("fillable", "missing", "invalid", "ambiguous")


@pytest.mark.parametrize("expected", STATUSES)
@pytest.mark.parametrize("actual", STATUSES)
def test_all_four_statuses_are_scored_and_counted_in_the_matrix(expected, actual):
    target = row({"recipient_inn": field(expected, "1234567890" if expected == "fillable" else None)})
    prediction = row({"recipient_inn": field(actual, "1234567890" if actual == "fillable" else None)})
    report = calculate_metrics([target], [prediction])
    scores = report.fields
    assert scores.expected_fields == 1
    assert scores.correct_statuses == scores.correct_status_values == int(expected == actual)
    assert scores.status_accuracy == scores.field_accuracy == float(expected == actual)
    assert scores.status_confusion[expected][actual] == 1
    assert sum(sum(columns.values()) for columns in scores.status_confusion.values()) == 1
    assert report.exact_document_correct == int(expected == actual)


@pytest.mark.parametrize("failure", ["no_prediction", "wrong_document_type", "absent_field"])
def test_unavailable_fields_are_not_misreported_as_missing_status(failure):
    target = row({"recipient_inn": field("invalid")})
    predictions = [] if failure == "no_prediction" else [row({}, kind="transfer_draft" if failure == "wrong_document_type" else "invoice_draft")]
    report = calculate_metrics([target], predictions)
    assert report.fields.status_confusion["invalid"][failure] == 1
    assert report.fields.status_confusion["invalid"]["missing"] == 0
    assert report.fields.status_accuracy == report.fields.field_accuracy == 0
    assert report.exact_document_accuracy == 0


def test_lost_result_reduces_coverage_document_accuracy_and_value_recall():
    targets = [row({"amount_rub": field("fillable", 100)}, case_id=cid) for cid in ("C1", "C2")]
    report = calculate_metrics(targets, targets[:1])
    assert report.prediction_coverage == report.exact_document_accuracy == .5
    assert report.exact_document_correct == report.predicted_cases == 1
    assert report.fields.value_accuracy == report.fields.value_precision == 1
    assert report.fields.value_recall == report.fields.status_accuracy == report.fields.field_accuracy == .5
    assert report.fields.status_confusion["fillable"]["no_prediction"] == 1
    assert report.by_document_type["invoice_draft"].value_recall == .5


def test_extra_nonfillable_field_makes_document_inexact():
    target = row({"recipient_inn": field("missing")})
    prediction = row({**target["fields"], "subject": field("missing")})
    report = calculate_metrics([target], [prediction])
    assert report.fields.field_accuracy == 1
    assert report.fields.false_positives == 0
    assert report.exact_document_correct == 0  # Требуется тот же набор полей.


def test_missing_value_key_is_not_a_correct_null_value():
    target = row({"recipient_inn": field("missing")})
    prediction = row({"recipient_inn": {"status": "missing"}})
    report = calculate_metrics([target], [prediction])
    assert report.fields.status_accuracy == 1
    assert report.fields.field_accuracy == report.exact_document_accuracy == 0


def test_exact_document_ignores_evidence_derivation_and_confidence():
    target = row({"amount_rub": field("fillable", 100, ["m1"])})
    prediction = row({"amount_rub": {**field("fillable", 100, ["m2"]), "derivation": "10 * 10 = 100"}})
    prediction["confidence"] = .1
    report = calculate_metrics([target], [prediction])
    assert report.exact_document_accuracy == 1
    assert report.fields.evidence_id_exact_match == 0


def test_no_document_is_correct_only_when_an_actual_none_result_exists():
    target = row({}, kind="none")
    assert calculate_metrics([target], [target]).exact_document_accuracy == 1
    assert calculate_metrics([target], []).exact_document_accuracy == 0


def test_aliases_do_not_change_strict_value_or_document_scores():
    target = row({"destination_bank": field("fillable", "Сбер")}, kind="transfer_draft")
    prediction = row({"destination_bank": field("fillable", "Сбербанк")}, kind="transfer_draft")
    report = calculate_metrics([target], [prediction])
    assert report.exact_document_correct == report.fields.correct_values == 0
    assert report.fields.alias_aware_correct_values == 1
    assert report.fields.alias_aware_value_recall == 1 and report.fields.value_recall == 0


def test_zero_denominators_have_no_claimed_value_or_status_accuracy():
    report = calculate_metrics([], [])
    assert report.prediction_coverage is None and report.exact_document_accuracy is None
    assert report.fields.value_precision is report.fields.value_recall is None
    assert report.fields.status_accuracy is report.fields.field_accuracy is None
    assert report.fields.expected_fields == report.fields.correct_values == 0


def test_classification_does_not_claim_to_evaluate_field_extraction(capsys):
    target = row({"recipient_inn": field("invalid")})
    report = calculate_metrics([target], [row({})], evaluate_fields=False)
    assert report.document_type_accuracy == report.prediction_coverage == 1
    assert not report.field_metrics_evaluated
    assert report.exact_document_correct is report.exact_document_accuracy is None
    assert report.fields.expected_fields == 0 and report.fields.status_accuracy is None
    print_metrics(report)
    printed = capsys.readouterr().out
    assert "Покрытие результатами: 1/1" in printed
    assert "Метрики полей:" not in printed
    assert "Полностью правильные документы:" not in printed


def test_cli_explains_status_errors_and_names_rates_without_claiming_support(capsys):
    target = row({"recipient_inn": field("invalid"), "amount_rub": field("fillable", 1000)})
    prediction = row({"recipient_inn": field("missing"), "amount_rub": field("fillable", 9000)})
    print_metrics(calculate_metrics([target], [prediction]))
    printed = capsys.readouterr().out
    assert "Полностью правильные документы: 0/1" in printed
    assert "invalid → missing: 1" in printed
    assert "value_recall=0.0%" in printed and "status_accuracy=50.0%" in printed
    assert "лишние заполнения=" in printed and "совпадение наборов ID источников=" in printed
    assert "необоснованные заполнения" not in printed


def test_audit_counters_and_loss_accounting_use_the_shared_metrics():
    from scripts.audit_jev import build_report

    target = row({"recipient_inn": field("invalid"), "amount_rub": field("fillable", 1000)})
    before = row({"recipient_inn": field("missing"), "amount_rub": field("fillable", 1000)})
    record = {"case_id": "C1", "before": before, "after": None, "status": "rejected", "jev_rounds": []}
    report = build_report([record], [target])
    for side in ("before", "after"):
        summary = report[side]
        metrics = summary["strict_metrics"]["fields"]
        assert summary["correct_expected_values"] == metrics["correct_values"]
        assert summary["expected_fillable_values"] == metrics["expected_values"]
        assert summary["total_expected_fields"] == metrics["expected_fields"]
        assert summary["correct_field_status_values"] == metrics["correct_status_values"]
    quality = report["verification_quality"]
    assert quality["before"]["strict_metrics"]["fields"]["value_recall"] == 1
    assert quality["after"]["strict_metrics"]["fields"]["value_recall"] == 0
    assert quality["after"]["strict_metrics"]["prediction_coverage"] == 0


def test_rates_without_fillable_fields_are_undefined_not_zero(capsys):
    target = row({"amount_rub": field("missing")})
    report = calculate_metrics([target], [deepcopy(target)])
    scores = report.fields
    assert scores.precision is None and scores.recall is None and scores.overfill_rate is None
    assert report.model_dump(mode="json")["metrics_version"] == 6
    print_metrics(report)
    output = capsys.readouterr().out
    assert "precision=н/д" in output and "recall=н/д" in output and "лишние заполнения=н/д" in output
