"""Обнаружение ошибки оценивается по первому кандидату, независимо от ремонта."""

import asyncio
from copy import deepcopy
import json

import pytest

from scripts.audit_jev import audit_cases, build_report, recover_report
from tests.test_metrics import field
from tests.test_paired_audit import case, Extractor, Jev, label


def observed(case_id="C1", *, wrong=True, flagged=True):
    target = {"case_id": case_id, "document_type": "transfer_draft",
              "fields": {"amount_rub": field("fillable", 200)}}
    before = deepcopy(target)
    if wrong:
        before["fields"]["amount_rub"]["value"] = 100
    entry = {"candidate_index": 0, "wrong_threshold": .8,
             "questions": {"amount_rub::superseded": {}},
             "answers": {"amount_rub::superseded": {"noul": .8 if flagged else .79}}}
    record = {"case_id": case_id, "before": before, "after": None, "status": "rejected",
              "candidates": [{"checked_fields": deepcopy(before["fields"])}], "jev_rounds": [entry]}
    return target, record


def test_four_detection_outcomes_use_root_fields_and_threshold_boundary():
    samples = [observed(f"C{i}", wrong=wrong, flagged=flagged)
               for i, (wrong, flagged) in enumerate(((True, True), (False, True), (True, False), (False, False)))]
    report = build_report([r for _, r in samples], [t for t, _ in samples])
    score = report["error_detection"]
    assert {key: score[key] for key in ("tp", "fp", "fn", "tn")} == dict.fromkeys(("tp", "fp", "fn", "tn"), 1)
    assert score["precision"] == score["recall"] == .5
    assert score["evaluated_cases"] == score["evaluated_fields"] == 4
    assert score["field_coverage"] == 1
    assert score["unassessed_fields"] == 0
    assert report["after"]["strict_metrics"]["prediction_coverage"] == 0


def test_later_repairs_and_failures_do_not_change_initial_detection():
    target, record = observed()
    original = build_report([record], [target])["error_detection"]
    assert original["tp"] == 1 and original["fn"] == 0
    record["after"] = deepcopy(target)
    record["status"] = "complete"
    record["candidates"].append({"checked_fields": target["fields"]})
    repaired = deepcopy(record["jev_rounds"][0])
    repaired["candidate_index"] = 1
    repaired["answers"]["amount_rub::superseded"]["noul"] = .01
    record["jev_rounds"].append(repaired)
    assert build_report([record], [target])["error_detection"] == original
    record["after"] = None
    record["status"] = "api_error"  # API сбой ремонта после уже полученного решения.
    assert build_report([record], [target])["error_detection"] == original


def test_attribute_and_root_flags_count_as_one_items_detection():
    target, record = observed()
    target["document_type"] = record["before"]["document_type"] = "invoice_draft"
    target["fields"] = {"items": field("fillable", [{"description": "ручка", "quantity": 2,
                                                    "unit": "шт.", "unit_price_rub": 100}])}
    record["before"]["fields"] = deepcopy(target["fields"])
    record["before"]["fields"]["items"]["value"][0]["quantity"] = 3
    record["candidates"][0]["checked_fields"] = deepcopy(record["before"]["fields"])
    keys = ["items::superseded", "items[0].quantity::value_wrong", "items[0].unit::value_wrong"]
    record["jev_rounds"][0].update(questions=dict.fromkeys(keys, {}),
                                  answers={key: {"noul": .9} for key in keys})
    result = build_report([record], [target])["error_detection"]
    assert result["tp"] == result["evaluated_fields"] == 1
    assert len(result["details"]) == 1
    assert result["details"][0]["flagged_checks"] == keys


@pytest.mark.parametrize("name,expected,actual", [
    ("destination_bank", "Сбер", "Сбербанк"),
    ("source_account_kind", "расчётный счёт", "расчётный"),
])
def test_permitted_alias_alone_is_not_a_real_error(name, expected, actual):
    target, record = observed(wrong=False, flagged=True)
    target["fields"] = {name: field("fillable", expected)}
    record["before"]["fields"] = {name: field("fillable", actual)}
    record["candidates"][0]["checked_fields"] = deepcopy(record["before"]["fields"])
    key = f"{name}::unsupported"
    record["jev_rounds"][0].update(questions={key: {}}, answers={key: {"noul": .9}})
    report = build_report([record], [target])
    assert report["before"]["strict_metrics"]["exact_document_correct"] == 0
    assert report["error_detection"]["fp"] == 1
    assert report["error_detection"]["tp"] == 0


@pytest.mark.parametrize("problem,reason", [
    ("missing_threshold", "unknown_threshold"),
    ("missing_link", "unlinked_initial_verdict"),
    ("later_candidate", "unlinked_initial_verdict"),
    ("no_response", "no_initial_verdict"),
    ("incomplete", "invalid_initial_verdict"),
    ("bad_score", "invalid_initial_verdict"),
    ("different_candidate", "candidate_mismatch"),
    ("wrong_type", "wrong_document_type"),
    ("no_baseline", "no_baseline"),
])
def test_unavailable_or_unbound_answers_do_not_become_tn_or_fn(problem, reason):
    target, record = observed()
    entry = record["jev_rounds"][0]
    if problem == "missing_threshold":
        del entry["wrong_threshold"]
    elif problem == "missing_link":
        del entry["candidate_index"]
    elif problem == "later_candidate":
        entry["candidate_index"] = 1
    elif problem == "no_response":
        del entry["answers"]
    elif problem == "incomplete":
        entry["answers"] = {}
    elif problem == "bad_score":
        entry["answers"]["amount_rub::superseded"]["noul"] = float("nan")
    elif problem == "different_candidate":
        record["candidates"][0]["checked_fields"]["amount_rub"]["value"] = 999
    elif problem == "wrong_type":
        record["before"]["document_type"] = "invoice_draft"
    else:
        record["before"] = None
    score = build_report([record], [target])["error_detection"]
    assert score["tp"] == score["fp"] == score["fn"] == score["tn"] == 0
    assert score["precision"] is score["recall"] is None
    assert score["unassessed_fields"] == 1
    assert score["skipped_cases"] == [{"case_id": "C1", "reason": reason}]


def test_missing_field_questions_are_visible_as_unassessed_not_correctly_accepted():
    target, record = observed(wrong=False, flagged=False)
    target["fields"]["recipient_relation"] = field("fillable", "self")
    record["before"]["fields"]["recipient_relation"] = field("missing")
    record["candidates"][0]["checked_fields"] = deepcopy(record["before"]["fields"])
    score = build_report([record], [target])["error_detection"]
    assert score["tn"] == 1 and score["fn"] == 0
    assert score["field_coverage"] == .5
    assert score["unassessed_fields"] == 1


def test_transport_retry_is_not_a_second_detection_observation():
    target, record = observed()
    retry = deepcopy(record["jev_rounds"][0])
    del retry["answers"]
    retry["error_type"] = "TypeSafeAPIConnectionError"
    record["jev_rounds"].insert(0, retry)
    score = build_report([record], [target])["error_detection"]
    assert score["tp"] == score["evaluated_fields"] == 1


def test_first_complete_response_is_used_instead_of_best_score_across_rounds():
    target, record = observed(flagged=False)
    later = deepcopy(record["jev_rounds"][0])
    later["answers"]["amount_rub::superseded"]["noul"] = .99
    record["jev_rounds"].append(later)
    score = build_report([record], [target])["error_detection"]
    assert score["fn"] == 1 and score["tp"] == 0


@pytest.mark.parametrize("flagged,precision", [(False, None), (True, 0)])
def test_correct_candidates_have_no_claimed_error_recall(flagged, precision):
    target, record = observed(wrong=False, flagged=flagged)
    score = build_report([record], [target])["error_detection"]
    assert score["recall"] is None
    assert score["precision"] == precision


def test_audit_captures_candidate_binding_and_recovery_preserves_detection(tmp_path):
    records = asyncio.run(audit_cases([case()], Extractor(), Jev(), wrong_threshold=.8))
    entries = records[0]["jev_rounds"]
    assert [entry["candidate_index"] for entry in entries] == [0, 1]
    assert all(entry["wrong_threshold"] == .8 for entry in entries)
    target = label()
    target.pop("scoring_only")
    score = build_report(records, [target])["error_detection"]
    assert score["tp"] == 1 and score["tn"] == 4
    trace, labels = tmp_path / "trace.jsonl", tmp_path / "labels.jsonl"
    trace.write_text(json.dumps(records[0]) + "\n")
    labels.write_text(json.dumps(target) + "\n")
    recovered = json.loads(recover_report(trace, labels, tmp_path / "recovered").read_text())
    assert recovered["error_detection"] == score


def test_real_audit_credits_detection_when_no_repair_attempt_remains():
    records = asyncio.run(audit_cases([case()], Extractor(), Jev(), max_attempts=1, wrong_threshold=.8))
    assert records[0]["status"] == "rejected" and records[0]["after"] is None
    report = build_report(records, [label()])
    assert report["field_changes"]["corrected"] == 0
    assert report["error_detection"]["tp"] == 1
    assert report["error_detection"]["recall"] == 1


def test_permitted_alias_list_does_not_allow_a_different_bank():
    target, record = observed()
    target["fields"] = {"destination_bank": field("fillable", "ВТБ")}
    record["before"]["fields"] = {"destination_bank": field("fillable", "Сбербанк")}
    record["candidates"][0]["checked_fields"] = deepcopy(record["before"]["fields"])
    key = "destination_bank::unsupported"
    record["jev_rounds"][0].update(questions={key: {}}, answers={key: {"noul": .9}})
    assert build_report([record], [target])["error_detection"]["tp"] == 1


def test_empty_detection_has_undefined_rates():
    score = build_report([], [])["error_detection"]
    assert score["evaluated_cases"] == score["evaluated_fields"] == 0
    assert score["precision"] is score["recall"] is score["field_coverage"] is None
