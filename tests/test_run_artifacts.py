"""Один запуск связывает конфигурацию, результаты, ошибки и метрики."""

import asyncio
import hashlib
import json

import pytest

import src.main as runner
from src.llm import LLMUsage
from src.pipeline import PipelineError, PipelineRateLimitError
from src.run_artifacts import load_run
from src.settings import DecisionModelSettings, LLMSettings, PipelineSettings
from tests.test_main_full import batch_files, result


@pytest.fixture
def offline(monkeypatch):
    class Extractor:
        usage = LLMUsage()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(runner, "DecisionModelSettings", lambda: DecisionModelSettings(
        _env_file=None, DECISION_API_KEY="private-verifier-fixture", DECISION_MODEL="verifier-fixture"))
    monkeypatch.setattr(runner, "DecisionModelClient", lambda **kwargs: object())
    monkeypatch.setattr(runner, "LLMSettings", lambda: LLMSettings(
        _env_file=None, LLM_API_KEY="private-llm-fixture", LLM_MODEL="llm-fixture", LLM_TEMPERATURE=0))
    monkeypatch.setattr(runner, "PipelineSettings", lambda: PipelineSettings(_env_file=None))
    monkeypatch.setattr(runner, "AsyncLLMExtractor", lambda *args, **kwargs: Extractor())


def read_json(path):
    return json.loads(path.read_text())


@pytest.mark.parametrize("mode", ["full", "llm-only"])
def test_saved_metrics_separate_content_acceptance_review_and_processing_failure(
    offline, monkeypatch, tmp_path, capsys, mode,
):
    from src.schemas.document_schemas import TransferResult

    inputs, labels, output = batch_files(tmp_path, count=5)

    async def process(case_id, *args, **kwargs):
        if case_id == "C5":
            raise PipelineError("fixture failure")
        payload = result(case_id).model_dump(mode="json")
        review = case_id in {"C3", "C4"}
        payload["validation"] = {"status": "needs_review" if review else "passed"}
        if case_id in {"C2", "C4"}:
            payload["fields"]["amount_rub"].update(status="fillable", value=100, evidence=["m1"])
        if review:
            payload["fields"]["amount_rub"]["validation_status"] = "needs_review"
            payload["validation"]["issues"] = [{"path": "amount_rub", "stage": "semantic",
                                                  "code": "unsupported", "message": "fixture review"}]
        return TransferResult.model_validate(payload)

    monkeypatch.setattr(runner, "process_document", process)
    actual = asyncio.run(runner.main(inputs, output, labels, mode=mode, verifier="decision"))
    saved = load_run(actual.parent)
    metrics = saved["metrics"]
    assert metrics["metrics_version"] == 6
    assert metrics["exact_document_correct"] == 2
    assert metrics["missing_predictions"] == ["C5"]
    assert len(saved["predictions"]) == 4 and len(saved["errors"]) == 1
    acceptance = metrics["acceptance"]
    assert acceptance["accepted_correct"] == acceptance["accepted_incorrect"] == 1
    assert acceptance["needs_review_correct"] == acceptance["needs_review_incorrect"] == 1
    assert acceptance["automatic_acceptance_rate"] == .4
    assert acceptance["accepted_error_rate"] == .5
    assert acceptance["correct_automatic_acceptance_rate"] == .2
    text = capsys.readouterr().out
    assert "Ошибки среди принятых: 1/2 (50.0%)" in text
    assert "Правильные автоматически принятые: 1/5 (20.0%)" in text


@pytest.mark.parametrize("mode", ["classification", "full", "llm-only"])
@pytest.mark.parametrize("failure", [PipelineError, PipelineRateLimitError])
def test_first_case_failure_keeps_two_runs_separate(offline, monkeypatch, tmp_path, capsys, mode, failure):
    inputs, labels, output = batch_files(tmp_path, count=2)

    async def succeed(case_id, *args, **kwargs):
        return result(case_id)

    async def classify(cases, client):
        return [runner.DocumentTypeResult(case_id=case.case_id, document_type="transfer_draft", confidence=1)
                for case in cases]

    monkeypatch.setattr(runner, "process_document", succeed)
    monkeypatch.setattr(runner, "classify_cases", classify)
    first = asyncio.run(runner.main(inputs, output, labels, mode=mode, verifier="decision"))

    async def fail(*args, **kwargs):
        raise failure("C1: безопасная ошибка")

    monkeypatch.setattr(runner, "process_document", fail)
    monkeypatch.setattr(runner, "classify_cases", fail)
    with pytest.raises(failure):
        asyncio.run(runner.main(inputs, output, labels, mode=mode, verifier="decision", fail_fast=True))

    runs = sorted((tmp_path / "predictions.runs").iterdir())
    assert len(runs) == 2
    old, new = runs
    assert first == old / output.name
    assert len(first.read_text().splitlines()) == 2
    assert (old / "predictions.errors.jsonl").read_text() == ""
    assert (new / output.name).read_text() == ""
    assert read_json(new / "predictions.errors.jsonl")["case_id"] == "C1"
    assert read_json(old / "metrics.json")["document_type_correct"] == 2
    assert read_json(new / "metrics.json")["total_cases"] == 1
    assert read_json(new / "metrics.json")["missing_predictions"] == ["C1"]
    assert read_json(old / "run.json")["status"] == "completed"
    manifest = read_json(new / "run.json")
    assert manifest["status"] == "stopped"
    assert manifest["unprocessed_case_ids"] == ["C2"]
    assert not output.exists() and not output.with_suffix(".errors.jsonl").exists()
    printed = capsys.readouterr().out
    assert str(new / output.name) in printed and str(new / "metrics.json") in printed
    assert "не изменялся" not in printed


@pytest.mark.parametrize("mode", ["classification", "full", "llm-only"])
def test_configuration_and_empty_checkpoint_exist_before_inference(offline, monkeypatch, tmp_path, mode):
    inputs, labels, output = batch_files(tmp_path, count=1)
    previous = output.with_name("predictions.errors.jsonl")
    output.write_text("old predictions")
    previous.write_text("old errors")

    def check():
        directory, = (tmp_path / "predictions.runs").iterdir()
        config = read_json(directory / "config.json")
        assert config["mode"] == mode and config["case_ids"] == ["C1"]
        assert config["input"]["sha256"] == hashlib.sha256(inputs.read_bytes()).hexdigest()
        assert config["labels"]["sha256"] == hashlib.sha256(labels.read_bytes()).hexdigest()
        if mode == "classification":
            assert config["llm"] is None
        else:
            assert config["llm"]["model"] == "llm-fixture" and config["llm"]["temperature"] == 0
            assert config["max_attempts"] == 2
        if mode == "llm-only":
            assert config["decision"] is None
        else:
            assert config["decision"]["model"] == "verifier-fixture"
        assert "private-" not in (directory / "config.json").read_text()
        assert read_json(directory / "run.json")["status"] == "running"
        assert read_json(directory / "metrics.json")["total_cases"] == 0
        assert (directory / output.name).read_text() == ""
        assert (directory / previous.name).read_text() == ""

    async def process(case_id, *args, **kwargs):
        check()
        return result(case_id)

    async def classify(cases, client):
        check()
        return [runner.DocumentTypeResult(case_id=case.case_id, document_type="transfer_draft", confidence=1)
                for case in cases]

    monkeypatch.setattr(runner, "process_document", process)
    monkeypatch.setattr(runner, "classify_cases", classify)
    actual = asyncio.run(runner.main(inputs, output, labels, mode=mode, verifier="decision"))
    assert output.read_text() == "old predictions" and previous.read_text() == "old errors"
    assert read_json(actual.parent / "run.json")["status"] == "completed"


def test_programming_error_leaves_a_marked_partial_run(offline, monkeypatch, tmp_path):
    inputs, labels, output = batch_files(tmp_path)

    async def process(case_id, *args, **kwargs):
        if case_id == "C2":
            raise RuntimeError("PRIVATE_DETAILS")
        return result(case_id)

    monkeypatch.setattr(runner, "process_document", process)
    with pytest.raises(RuntimeError, match="PRIVATE_DETAILS"):
        asyncio.run(runner.main(inputs, output, labels, mode="full", verifier="decision"))
    directory, = (tmp_path / "predictions.runs").iterdir()
    manifest = read_json(directory / "run.json")
    assert manifest["status"] == "interrupted" and manifest["failure_kind"] == "RuntimeError"
    assert manifest["processed_case_ids"] == ["C1"]
    assert manifest["unprocessed_case_ids"] == ["C2", "C3"]
    assert read_json(directory / "metrics.json")["total_cases"] == 1
    assert "PRIVATE_DETAILS" not in (directory / "run.json").read_text()


@pytest.mark.parametrize("interruption", [asyncio.CancelledError, KeyboardInterrupt])
def test_cancellation_is_recorded_and_reraised(offline, monkeypatch, tmp_path, interruption):
    inputs, labels, output = batch_files(tmp_path, count=1)

    async def process(*args, **kwargs):
        raise interruption()

    monkeypatch.setattr(runner, "process_document", process)
    with pytest.raises(interruption):
        asyncio.run(runner.main(inputs, output, labels, mode="llm-only"))
    directory, = (tmp_path / "predictions.runs").iterdir()
    snapshot = load_run(directory)
    assert snapshot["run"]["status"] == "interrupted"
    assert snapshot["run"]["failure_kind"] == interruption.__name__
    assert snapshot["predictions"] == snapshot["errors"] == []
    assert snapshot["metrics"]["total_cases"] == 0


def test_completed_run_can_be_read_and_changed_artifacts_are_rejected(offline, monkeypatch, tmp_path):
    inputs, labels, output = batch_files(tmp_path, count=1)

    async def process(case_id, *args, **kwargs):
        return result(case_id)

    monkeypatch.setattr(runner, "process_document", process)
    actual = asyncio.run(runner.main(inputs, output, labels, mode="llm-only"))
    snapshot = load_run(actual.parent)
    assert snapshot["predictions"][0]["case_id"] == "C1"
    assert snapshot["errors"] == [] and snapshot["metrics"]["predicted_cases"] == 1
    assert snapshot["config"]["run_id"] == snapshot["run"]["run_id"]
    actual.write_text(result("FOREIGN_RUN").model_dump_json())
    with pytest.raises(ValueError, match="снимку"):
        load_run(actual.parent)


def test_disk_failure_does_not_publish_a_mixed_snapshot(offline, monkeypatch, tmp_path):
    import src.run_artifacts as artifacts

    inputs, labels, output = batch_files(tmp_path, count=1)
    write = artifacts.write_errors

    def fail(errors, path):
        if errors:
            raise OSError("disk fixture")
        write(errors, path)

    async def process(*args, **kwargs):
        raise PipelineError("C1: безопасная ошибка")

    monkeypatch.setattr(artifacts, "write_errors", fail)
    monkeypatch.setattr(runner, "process_document", process)
    with pytest.raises(OSError, match="disk fixture"):
        asyncio.run(runner.main(inputs, output, labels, mode="llm-only"))
    directory, = (tmp_path / "predictions.runs").iterdir()
    assert read_json(directory / "run.json")["status"] == "writing"
    with pytest.raises(ValueError, match="не дописан"):
        load_run(directory)


@pytest.mark.parametrize("name", ["config.json", "metrics.json", "run.json"])
def test_reserved_artifact_names_fail_before_configuration(monkeypatch, tmp_path, name):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid artifact names must fail before configuring clients")

    monkeypatch.setattr(runner, "LLMSettings", forbidden)
    with pytest.raises(ValueError, match="имя файла"):
        asyncio.run(runner.main(output_path=tmp_path / name, mode="llm-only"))
    assert list(tmp_path.iterdir()) == []


def test_cli_saves_review_document_and_counts_it_separately(offline, monkeypatch, tmp_path, capsys):
    from src.extraction import ExtractionProposal
    from src.schemas.document_schemas import TransferFields
    from src.schemas.extraction_schemas import DocumentClassification

    inputs, labels, requested = batch_files(tmp_path, count=1)

    class Extractor:
        usage = LLMUsage()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def classify(self, *args, **kwargs):
            return DocumentClassification(document_type="transfer_draft", confidence=1, none_reason=None)

        async def extract(self, *args, **kwargs):
            fields = {name: {"status": "missing", "value": None, "evidence": []}
                      for name in TransferFields.model_fields}
            fields["amount_rub"] = {"status": "fillable", "value": 100,
                "evidence": [{"message_id": "m1", "quote": "PRIVATE_BAD_QUOTE"}]}
            fields["recipient_relation"] = {"status": "fillable", "value": "self",
                "evidence": [{"message_id": "m1", "quote": "себе"}]}
            return ExtractionProposal.model_validate({"fields": fields})

    monkeypatch.setattr(runner, "AsyncLLMExtractor", lambda *args, **kwargs: Extractor())
    output = asyncio.run(runner.main(inputs, requested, labels, mode="llm-only", fail_fast=True))
    saved = json.loads(output.read_text())
    assert saved["validation"]["status"] == "needs_review"
    assert saved["fields"]["amount_rub"]["value"] == 100
    assert saved["fields"]["recipient_relation"]["value"] == "self"
    assert output.with_name("predictions.errors.jsonl").read_text() == ""
    snapshot = load_run(output.parent)
    assert snapshot["run"]["successful_cases"] == 0
    assert snapshot["run"]["needs_review_cases"] == 1
    assert snapshot["run"]["failed_cases"] == 0
    assert snapshot["run"]["status"] == "completed"
    assert snapshot["metrics"]["predicted_cases"] == 1
    printed = capsys.readouterr().out
    assert "Требуют проверки: 1" in printed
    assert "amount_rub.evidence[0].quote" in printed
    assert "PRIVATE_BAD_QUOTE" not in printed
