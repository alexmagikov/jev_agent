"""Восстановление метрик из контрольных записей без новых запросов к моделям."""

import hashlib
import json
import subprocess
import sys

import pytest

from scripts.audit_jev import recover_report
from tests.test_paired_audit import label


def inputs(tmp_path):
    gold = label(100)
    gold.pop("scoring_only")
    second = {**gold, "case_id": "C2"}
    complete = {"case_id": "C1", "before": gold, "after": gold,
                "status": "complete", "jev_rounds": []}
    rejected = {"case_id": "C2", "before": second, "after": None,
                "status": "rejected", "error_type": "PipelineRejectionError",
                "failure_reason": "semantic_rejection", "jev_rounds": []}
    trace, labels = tmp_path / "trace.jsonl", tmp_path / "labels.jsonl"
    trace.write_text("\n".join(json.dumps(row) for row in (
        {**complete, "before": None, "after": None, "status": "started"}, complete, rejected)))
    labels.write_text("\n".join(json.dumps(row) for row in (gold, second)))
    return trace, labels


def test_recovery_keeps_last_checkpoint_and_counts_rejected_results(tmp_path, monkeypatch):
    import scripts.audit_jev as audit
    def forbidden(*args, **kwargs):
        pytest.fail("Offline recovery must not configure or call a model")
    monkeypatch.setattr(audit, "LLMSettings", forbidden)
    monkeypatch.setattr(audit, "DecisionModelSettings", forbidden)
    monkeypatch.setattr(audit, "AsyncTypeSafeClient", forbidden)
    trace, labels = inputs(tmp_path)
    source_hash = hashlib.sha256(trace.read_bytes()).hexdigest()
    destination = recover_report(trace, labels, tmp_path / "recovered")
    report = json.loads(destination.read_text())
    assert report["attempted_cases"] == 2 and report["complete_pairs"] == 1
    assert report["before"]["correct_expected_values"] == 2
    assert report["after"]["correct_expected_values"] == 1
    assert report["conclusion"] == "incomplete"
    assert report["result_counts"] == {"before": 2, "after": 1, "lost_results": ["C2"], "gained_results": []}
    assert report["recovered_offline"] is True
    assert report["source_trace_sha256"] == source_hash == hashlib.sha256(trace.read_bytes()).hexdigest()
    assert report["llm_usage"] is None and report["verifier_usage"] is None


def test_recovery_never_overwrites_an_existing_directory(tmp_path):
    trace, labels = inputs(tmp_path)
    destination = tmp_path / "existing"
    destination.mkdir()
    old_report = destination / "report.json"
    old_report.write_text("preserve")
    with pytest.raises(FileExistsError):
        recover_report(trace, labels, destination)
    assert old_report.read_text() == "preserve"


def test_recovery_rejects_empty_trace_without_writing_report(tmp_path):
    trace, labels = inputs(tmp_path)
    trace.write_text("")
    with pytest.raises(ValueError, match="контрольных записей"):
        recover_report(trace, labels, tmp_path / "recovered")
    assert not (tmp_path / "recovered").exists()


def test_parser_separates_live_runs_and_offline_recovery():
    from scripts.audit_jev import build_parser
    parser = build_parser()
    args = parser.parse_args(["--report-trace", "trace.jsonl"])
    assert not args.live
    with pytest.raises(SystemExit):
        parser.parse_args(["--live", "--report-trace", "trace.jsonl"])


def test_offline_cli_prints_acceptance_without_inventing_old_validation_statuses(tmp_path):
    trace, labels = inputs(tmp_path)
    completed = subprocess.run([
        sys.executable, "-B", "-m", "scripts.audit_jev", "--report-trace", str(trace),
        "--labels", str(labels), "--output-dir", str(tmp_path / "recovered"),
    ], capture_output=True, text=True, check=True)
    assert completed.stdout.count("Ошибки среди принятых: 0/0 (н/д)") == 2
    assert "Результаты без статуса проверки: 2" in completed.stdout
    assert "Результаты без статуса проверки: 1" in completed.stdout
    assert "TP=0; FP=0; FN=0; TN=0" in completed.stdout
    assert "Оценено полей: 0/10 (0.0%)" in completed.stdout
    assert "Итог по полям, связанным с первой проверкой" in completed.stdout
    assert "полей без исходной оценки: 10" in completed.stdout
