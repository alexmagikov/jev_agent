"""Первый вердикт и итог одного корневого поля, включая строгий отказ аудита."""

import asyncio
from copy import deepcopy
import json

import pytest

from scripts.audit_jev import audit_cases, build_report, recover_report
from tests.test_detection_metrics import observed
from tests.test_metrics import field
from tests.test_paired_audit import case, Extractor, Jev, label


def finished(*, initial_wrong, final_wrong, flagged=True, review=None):
    target, record = observed(wrong=initial_wrong, flagged=flagged)
    after = deepcopy(target)
    if final_wrong:
        after["fields"]["amount_rub"]["value"] = 300
    after["validation"] = {"status": "needs_review" if review else "passed", "issues": []}
    if review:
        after["fields"]["amount_rub"]["validation_status"] = "needs_review"
        after["validation"]["issues"] = [{"path": "amount_rub", "stage": review, "code": "fixture"}]
    record.update(after=after, status="complete")
    return target, record


@pytest.mark.parametrize("initial_wrong,final_wrong,flagged,review,expected", [
    (True, False, True, None, {"corrected": 1}),
    (True, True, True, "semantic", {"detected_and_review": 1}),
    (True, True, False, None, {"missed": 1}),
    (False, True, True, None, {"regressed_by_repair": 1, "missed": 1}),
    (False, False, True, "semantic", {"false_review": 1}),
    (True, False, True, "semantic", {"corrected": 1, "false_review": 1}),
    (True, True, False, "semantic", {"detected_and_review": 1}),
    (True, True, True, None, {"missed": 1}),  # Ошибка обнаружена сначала, но осталась без отметки в итоге.
])
def test_five_counters_link_the_same_field_and_allow_content_routing_overlap(
    initial_wrong, final_wrong, flagged, review, expected,
):
    target, record = finished(initial_wrong=initial_wrong, final_wrong=final_wrong, flagged=flagged, review=review)
    report = build_report([record], [target])
    outcome = report["field_outcomes"]
    primary = {key: outcome[key] for key in ("corrected", "detected_and_review", "missed", "regressed_by_repair", "false_review")}
    assert primary == {key: expected.get(key, 0) for key in primary}
    detail, = outcome["details"]
    assert detail["case_id"] == "C1" and detail["field"] == "amount_rub"
    assert detail["initial_flagged"] == flagged
    assert detail["initial_error"] == initial_wrong
    assert detail["final_correct"] == (not final_wrong)
    assert detail["initial_outcome"] == report["error_detection"]["details"][0]["outcome"]
    assert detail["document_disposition"] == ("needs_review" if review else "passed")


def test_review_for_another_field_is_not_attributed_to_this_field():
    target, record = finished(initial_wrong=False, final_wrong=False)
    target["fields"]["recipient_relation"] = field("fillable", "self")
    record["before"]["fields"]["recipient_relation"] = field("fillable", "self")
    record["candidates"][0]["checked_fields"] = deepcopy(record["before"]["fields"])
    record["after"]["fields"]["recipient_relation"] = {**field("fillable", "self"), "validation_status": "needs_review"}
    record["after"]["validation"] = {"status": "needs_review", "issues": [
        {"path": "recipient_relation", "stage": "semantic", "code": "fixture"}]}
    score = build_report([record], [target])["field_outcomes"]
    assert score["false_review"] == score["detected_and_review"] == 0
    detail, = score["details"]
    assert detail["final_disposition"] == "passed"
    assert detail["document_disposition"] == "needs_review"


@pytest.mark.parametrize("correct", [True, False])
def test_python_review_does_not_become_a_jev_review(correct):
    target, record = finished(initial_wrong=not correct, final_wrong=not correct, review="python")
    score = build_report([record], [target])["field_outcomes"]
    assert score["false_review"] == score["detected_and_review"] == score["missed"] == 0
    assert score["other_review"] == 1
    assert score["details"][0]["review_origin"] == "python"


def test_nested_issue_is_linked_to_its_root_field():
    target, record = finished(initial_wrong=True, final_wrong=True, review="semantic")
    record["after"]["validation"]["issues"][0]["path"] = "amount_rub.calculation.operands[0]"
    assert build_report([record], [target])["field_outcomes"]["detected_and_review"] == 1


@pytest.mark.parametrize("gold,detected,false_rejection", [(200, 1, 0), (100, 0, 1)])
def test_actual_strict_rejection_is_separate_from_a_review_and_unaffected_fields(gold, detected, false_rejection):
    records = asyncio.run(audit_cases([case()], Extractor(), Jev(), max_attempts=1, wrong_threshold=.8))
    assert records[0]["after"] is None
    score = build_report(records, [label(gold)])["field_outcomes"]
    assert score["detected_and_rejected"] == detected
    assert score["false_rejection"] == false_rejection
    assert score["detected_and_review"] == score["false_review"] == 0
    assert score["withheld_fields"] == 4  # У соседних полей не было замечаний.
    amount = next(detail for detail in score["details"] if detail["field"] == "amount_rub")
    assert amount["final_correct"] is None
    assert amount["last_candidate_correct"] == (gold == 100)
    assert amount["final_disposition"] == "rejected"


@pytest.mark.parametrize("status,reason", [("api_error", None), ("budget_exhausted", None), ("rejected", "invalid_extraction")])
def test_unavailable_final_result_is_not_invented_as_manual_review(status, reason):
    target, record = observed()
    record.update(status=status, failure_reason=reason)
    record["jev_rounds"][0]["answers"]["amount_rub::superseded"]["noul"] = .99
    score = build_report([record], [target])["field_outcomes"]
    assert score["unknown_final"] == 1
    assert score["detected_and_rejected"] == score["detected_and_review"] == score["missed"] == 0
    assert score["details"][0]["final_correct"] is None


def test_missing_final_validation_keeps_content_transition_but_does_not_claim_acceptance():
    target, record = finished(initial_wrong=True, final_wrong=False)
    del record["after"]["validation"]
    score = build_report([record], [target])["field_outcomes"]
    assert score["corrected"] == score["unknown_final"] == 1
    assert score["details"][0]["final_disposition"] == "unknown"


def test_failed_python_repair_does_not_reuse_stale_semantic_rejection_as_final():
    records = asyncio.run(audit_cases([case()], Extractor(), Jev(), max_attempts=1, wrong_threshold=.8))
    records[0]["failure_reason"] = "invalid_extraction"
    score = build_report(records, [label()])["field_outcomes"]
    assert score["unknown_final"] == 5
    assert score["detected_and_rejected"] == 0


def test_offline_recovery_preserves_the_link_and_no_raw_values_are_added(tmp_path):
    records = asyncio.run(audit_cases([case()], Extractor(), Jev(), wrong_threshold=.8))
    target = label()
    target.pop("scoring_only")
    score = build_report(records, [target])["field_outcomes"]
    assert score["corrected"] == 1
    trace, labels = tmp_path / "trace.jsonl", tmp_path / "labels.jsonl"
    trace.write_text(json.dumps(records[0]) + "\n")
    labels.write_text(json.dumps(target) + "\n")
    recovered = json.loads(recover_report(trace, labels, tmp_path / "recovered").read_text())
    assert recovered["field_outcomes"] == score
    assert all("value" not in detail and "quote" not in detail for detail in score["details"])


def test_unassessed_initial_fields_are_not_silently_linked_to_final_results():
    target, record = finished(initial_wrong=True, final_wrong=False)
    del record["jev_rounds"][0]["candidate_index"]
    score = build_report([record], [target])["field_outcomes"]
    assert score["linked_fields"] == score["corrected"] == 0
    assert score["unassessed_fields"] == 1


@pytest.mark.parametrize("gold,counter", [(200, "detected_and_review"), (100, "false_review")])
def test_link_accepts_the_actual_pipeline_review_result(gold, counter):
    from src.pipeline import DocumentTypeDecision, process_selected_document
    from tests.test_paired_audit import proposal

    records = asyncio.run(audit_cases([case()], Extractor(), Jev(), max_attempts=1, wrong_threshold=.8))
    after = asyncio.run(process_selected_document(
        "C1", case().messages, DocumentTypeDecision(document_type="transfer_draft", confidence=1),
        Jev(), Extractor(), initial_proposal=proposal(100), max_attempts=1, wrong_threshold=.8,
    ))
    assert after.validation.status == "needs_review"
    records[0].update(after=after.model_dump(mode="json"), status="complete")
    score = build_report(records, [label(gold)])["field_outcomes"]
    assert score[counter] == 1
    assert score["unknown_final"] == score["withheld_fields"] == 0


def test_global_response_issue_is_not_assumed_to_accept_each_field():
    target, record = finished(initial_wrong=True, final_wrong=True)
    record["after"]["validation"] = {"status": "needs_review", "issues": [
        {"path": "", "stage": "llm_response", "code": "invalid_json"}]}
    score = build_report([record], [target])["field_outcomes"]
    assert score["unknown_final"] == 1
    assert score["missed"] == score["detected_and_review"] == 0
