"""Правильность содержимого и автоматическое принятие — независимые измерения."""

import asyncio
from copy import deepcopy
import json

import pytest

from scripts.audit_jev import build_report, recover_report
from src.cases.document_type import DocumentMessage
from src.main import print_metrics
from src.metrics import calculate_metrics
from src.pipeline import DocumentTypeDecision, process_selected_document
from src.schemas.document_schemas import InvoiceFields
from tests.test_metric_contract import row
from tests.test_metrics import field


def matrix():
    targets = [row({name: field("fillable", 100) if name == "amount_rub" else field("missing")
                    for name in InvoiceFields.model_fields}, case_id=f"C{i}") for i in range(5)]
    predictions = deepcopy(targets[:4])
    for index, prediction in enumerate(predictions):
        prediction["validation"] = {"status": "passed" if index < 2 else "needs_review"}
        if index % 2:
            prediction["fields"]["amount_rub"]["value"] = 200
    return targets, predictions


def test_four_outcomes_and_missing_result_have_separate_counts_and_denominators():
    targets, predictions = matrix()
    report = calculate_metrics(targets, predictions)
    scores = report.acceptance
    assert report.exact_document_correct == 2
    assert report.exact_document_accuracy == .4  # Правильный needs_review остаётся правильным.
    assert report.missing_predictions == ["C4"]
    assert scores.accepted_cases == scores.needs_review_cases == 2
    assert scores.accepted_correct == scores.accepted_incorrect == 1
    assert scores.needs_review_correct == scores.needs_review_incorrect == 1
    assert scores.unknown_validation_cases == 0
    assert scores.automatic_acceptance_rate == scores.needs_review_rate == .4
    assert scores.accepted_error_rate == .5
    assert scores.correct_automatic_acceptance_rate == .2
    serialized = report.model_dump(mode="json")
    assert serialized["metrics_version"] == 6
    assert serialized["acceptance"] == scores.model_dump(mode="json")


@pytest.mark.parametrize("metadata", [None, {}, {"status": "unknown"}])
def test_missing_or_unknown_validation_is_not_invented_as_passed(metadata):
    target = row({"amount_rub": field("fillable", 100)})
    prediction = {**target, "validation": metadata}
    for result in (prediction, target):  # Включая старый JSON без ключа validation.
        report = calculate_metrics([target], [result])
        assert report.exact_document_accuracy == 1
        assert report.acceptance.unknown_validation_cases == 1
        assert report.acceptance.accepted_cases == report.acceptance.needs_review_cases == 0
        assert report.acceptance.accepted_error_rate is None


@pytest.mark.parametrize("case", ["empty", "missing", "review"])
def test_no_accepted_results_do_not_claim_zero_error_risk(case):
    target = row({"amount_rub": field("fillable", 100)})
    labels = [] if case == "empty" else [target]
    predictions = [{**target, "validation": {"status": "needs_review"}}] if case == "review" else []
    report = calculate_metrics(labels, predictions)
    assert report.acceptance.accepted_error_rate is None
    assert report.acceptance.accepted_cases == report.acceptance.accepted_correct == 0
    assert report.acceptance.automatic_acceptance_rate == (None if case == "empty" else 0)
    assert report.acceptance.correct_automatic_acceptance_rate == (None if case == "empty" else 0)


def test_classification_only_does_not_claim_document_acceptance(capsys):
    target = row({})
    report = calculate_metrics([target], [{**target, "validation": {"status": "passed"}}], evaluate_fields=False)
    assert report.acceptance is None
    print_metrics(report)
    assert "Автоматически приняты:" not in capsys.readouterr().out


@pytest.mark.parametrize("change", ["type", "status", "extra", "alias"])
def test_accepted_error_uses_the_same_strict_content_contract(change):
    target = row({"destination_bank": field("fillable", "Сбер")}, kind="transfer_draft")
    prediction = deepcopy(target)
    prediction["validation"] = {"status": "passed"}
    if change == "type":
        prediction["document_type"] = "invoice_draft"
    elif change == "status":
        prediction["fields"]["destination_bank"] = field("missing")
    elif change == "extra":
        prediction["fields"]["amount_rub"] = field("missing")
    else:
        prediction["fields"]["destination_bank"]["value"] = "Сбербанк"
    report = calculate_metrics([target], [prediction])
    assert report.exact_document_correct == report.acceptance.accepted_correct == 0
    assert report.acceptance.accepted_incorrect == 1
    assert report.acceptance.accepted_error_rate == 1


def test_cli_prints_content_acceptance_risk_and_review_counts(capsys):
    print_metrics(calculate_metrics(*matrix()))
    text = capsys.readouterr().out
    assert "Полностью правильные документы: 2/5" in text
    assert "Автоматически приняты: 2/5 (40.0%)" in text
    assert "Ошибки среди принятых: 1/2 (50.0%)" in text
    assert "Правильные автоматически принятые: 1/5 (20.0%)" in text
    assert "Требуют проверки: 2/5 (40.0%); правильных: 1; ошибочных: 1" in text


def test_unknown_validation_and_undefined_risk_are_visible_in_cli(capsys):
    target = row({})
    print_metrics(calculate_metrics([target], [target]))
    text = capsys.readouterr().out
    assert "Ошибки среди принятых: 0/0 (н/д)" in text
    assert "Результаты без статуса проверки: 1" in text


@pytest.mark.parametrize("use_verifier", [True, False])
def test_none_decision_is_an_explicit_accepted_outcome_without_field_verification(use_verifier):
    result = asyncio.run(process_selected_document(
        "no-document", [DocumentMessage(id="m1", role="user", text="Отмени подготовку")],
        DocumentTypeDecision(document_type="none", confidence=1, reason="Запрос отменён"),
        object() if use_verifier else None, object(), use_verifier=use_verifier,
    ))
    assert result.validation.status == result.validation.python_status == "passed"
    assert result.validation.semantic_status == "not_run"
    target = row({}, case_id="no-document", kind="none")
    scores = calculate_metrics([target], [result.model_dump(mode="json")]).acceptance
    assert scores.accepted_correct == 1
    assert scores.correct_automatic_acceptance_rate == 1


def test_paired_report_and_offline_recovery_reuse_the_same_acceptance_metrics(tmp_path):
    targets, predictions = matrix()
    records = [{"case_id": target["case_id"], "before": predictions[index] if index < 4 else None,
                "after": predictions[3 - index] if index < 4 else None,
                "status": "complete" if index < 4 else "baseline_rejected", "jev_rounds": []}
               for index, target in enumerate(targets)]
    # Сохраняем идентификаторы кейсов при изменении содержимого/статуса между сторонами.
    for record in records:
        if record["after"] is not None:
            record["after"] = {**record["after"], "case_id": record["case_id"]}
    expected = build_report(records, targets)
    for side in ("before", "after"):
        assert expected[side]["strict_metrics"]["acceptance"] == calculate_metrics(
            targets, [record[side] for record in records if record[side] is not None],
        ).acceptance.model_dump(mode="json")
    trace, labels = tmp_path / "trace.jsonl", tmp_path / "labels.jsonl"
    trace.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    labels.write_text("\n".join(json.dumps(target) for target in targets) + "\n")
    recovered = json.loads(recover_report(trace, labels, tmp_path / "recovered").read_text())
    for side in ("before", "after"):
        assert recovered[side]["strict_metrics"]["acceptance"] == expected[side]["strict_metrics"]["acceptance"]
