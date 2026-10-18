import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

import src.main as runner
from src.schemas.document_schemas import TransferFields, TransferResult
from src.llm import LLMUsage
from src.settings import PipelineSettings


class FakeExtractor:
    def __init__(self, settings):
        self.usage = LLMUsage(requests=1, responses=1, usage_reports=1, input_tokens=100, output_tokens=20)
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True


def result(case_id="C1"):
    return TransferResult(
        case_id=case_id, document_type="transfer_draft", confidence=0.9,
        fields={name: {"status": "missing", "value": None, "evidence": [], "derivation": None}
                for name in TransferFields.model_fields},
    )


def write_input(path, duplicate=False):
    row = {"case_id": "C1", "split": "dev", "messages": [
        {"id": "m1", "role": "user", "text": "Переведи себе"},
    ]}
    path.write_text(json.dumps(row) + "\n" + (json.dumps(row) if duplicate else ""))


def test_writing_full_results_preserves_null_and_creates_directory(tmp_path):
    output = tmp_path / "outputs" / "predictions.jsonl"
    runner.write_results([result()], output)
    row = json.loads(output.read_text())
    assert row["fields"]["amount_rub"] == {
        "status": "missing", "value": None, "evidence": [], "derivation": None,
    }


def test_duplicate_case_ids_rejected_before_inference(tmp_path):
    path = tmp_path / "cases.jsonl"
    write_input(path, duplicate=True)
    with pytest.raises(ValueError, match="case_id"):
        runner.load_cases(path)


@pytest.mark.parametrize("debug", [False, True])
def test_main_full_writes_predictions_and_prints_metrics(monkeypatch, tmp_path, capsys, debug):
    input_path, labels_path = tmp_path / "cases.jsonl", tmp_path / "labels.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    write_input(input_path)
    labels_path.write_text(result().model_dump_json() + "\n")
    calls = []

    async def process(case_id, messages, client, extractor, **kwargs):
        calls.append((case_id, messages, kwargs))
        return result(case_id)

    monkeypatch.setattr(runner, "process_document", process)
    monkeypatch.setattr(runner, "JevClientSettings", lambda: SimpleNamespace(api_key=SecretStr("fake")))
    monkeypatch.setattr(runner, "AsyncJev", lambda **kwargs: object())
    monkeypatch.setattr(runner, "LLMSettings", lambda: object())
    extractor = FakeExtractor(None)
    monkeypatch.setattr(runner, "AsyncLLMExtractor", lambda settings, **kwargs: extractor)
    monkeypatch.setattr(runner, "PipelineSettings", lambda: PipelineSettings(_env_file=None))
    asyncio.run(runner.main(input_path, output_path, labels_path, mode="full", limit=1, debug=debug))
    assert len(calls) == 1
    assert json.loads(output_path.read_text())["fields"]["amount_rub"]["value"] is None
    printed = capsys.readouterr().out
    assert "1/1 (100.0%)" in printed
    assert "вход=100" in printed and "выход=20" in printed
    assert ("Ожидаемый результат" in printed) is debug
    assert ("Итог после проверок" in printed) is debug
    assert calls[0][2]["debug"] is debug
    assert extractor.closed


def test_full_failure_preserves_previous_output(monkeypatch, tmp_path):
    from src.pipeline import PipelineError

    path, labels, output = tmp_path / "cases.jsonl", tmp_path / "labels.jsonl", tmp_path / "out.jsonl"
    write_input(path)
    labels.write_text(result().model_dump_json())
    output.write_text("previous run\n")

    async def fail(*args, **kwargs):
        raise PipelineError("C1: проверка не пройдена")

    monkeypatch.setattr(runner, "process_document", fail)
    monkeypatch.setattr(runner, "JevClientSettings", lambda: SimpleNamespace(api_key=SecretStr("fake")))
    monkeypatch.setattr(runner, "AsyncJev", lambda **kwargs: object())
    monkeypatch.setattr(runner, "LLMSettings", lambda: object())
    extractor = FakeExtractor(None)
    monkeypatch.setattr(runner, "AsyncLLMExtractor", lambda settings, **kwargs: extractor)
    with pytest.raises(PipelineError):
        asyncio.run(runner.main(path, output, labels, mode="full", fail_fast=True))
    assert output.read_text() == "previous run\n"
    assert extractor.closed


def test_output_cannot_overwrite_dataset(tmp_path):
    path = tmp_path / "cases.jsonl"
    write_input(path)
    with pytest.raises(ValueError, match="совпадать"):
        asyncio.run(runner.main(input_path=path, output_path=path, mode="full"))


def test_cli_supports_full_mode_and_limit():
    args = runner.build_parser().parse_args(["--mode", "full", "--limit", "1"])
    assert args.mode == "full"
    assert args.limit == 1


def test_debug_cli_is_explicitly_opt_in():
    parser = runner.build_parser()
    assert parser.parse_args([]).debug is False
    assert parser.parse_args(["--mode", "full", "--debug"]).debug is True


@pytest.mark.parametrize("debug", [False, True])
def test_debug_prints_expected_before_inference_even_on_failure(monkeypatch, tmp_path, capsys, debug):
    from src.pipeline import PipelineError

    path, labels = tmp_path / "cases.jsonl", tmp_path / "labels.jsonl"
    write_input(path)
    label = result().model_dump(mode="json")
    label["fields"]["destination_bank"]["value"] = "EXPECTED_ONLY"
    labels.write_text(json.dumps(label))

    async def fail(*args, **kwargs):
        # Эталон доступен только консоли, но не модели и не проверяющему.
        assert "EXPECTED_ONLY" not in repr((args, kwargs))
        assert kwargs["debug"] is debug
        raise PipelineError("C1: проверка не пройдена")

    monkeypatch.setattr(runner, "process_document", fail)
    monkeypatch.setattr(runner, "JevClientSettings", lambda: SimpleNamespace(api_key=SecretStr("fake")))
    monkeypatch.setattr(runner, "AsyncJev", lambda **kwargs: object())
    monkeypatch.setattr(runner, "LLMSettings", lambda: object())
    monkeypatch.setattr(runner, "AsyncLLMExtractor", lambda settings, **kwargs: FakeExtractor(settings))
    with pytest.raises(PipelineError):
        asyncio.run(runner.main(path, tmp_path / "out.jsonl", labels, mode="full", debug=debug, fail_fast=True))
    printed = capsys.readouterr().out
    assert ("EXPECTED_ONLY" in printed) is debug


def mock_batch_runtime(monkeypatch, process):
    monkeypatch.setattr(runner, "process_document", process)
    monkeypatch.setattr(runner, "JevClientSettings", lambda: SimpleNamespace(api_key=SecretStr("fake")))
    monkeypatch.setattr(runner, "AsyncJev", lambda **kwargs: object())
    monkeypatch.setattr(runner, "LLMSettings", lambda: object())
    extractor = FakeExtractor(None)
    monkeypatch.setattr(runner, "AsyncLLMExtractor", lambda settings, **kwargs: extractor)
    return extractor


def batch_files(tmp_path, count=3):
    path, labels = tmp_path / "cases.jsonl", tmp_path / "labels.jsonl"
    rows = [{"case_id": f"C{i}", "split": "dev", "messages": [
        {"id": "m1", "role": "user", "text": "Переведи себе"},
    ]} for i in range(1, count + 1)]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    labels.write_text("\n".join(result(row["case_id"]).model_dump_json() for row in rows))
    return path, labels, tmp_path / "predictions.jsonl"


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("failed_id", ["C1", "C2", "C3"])
def test_batch_continues_after_failure_and_saves_each_case(monkeypatch, tmp_path, capsys, failed_id):
    from src.pipeline import PipelineError

    path, labels, output = batch_files(tmp_path)
    errors = tmp_path / "predictions.errors.jsonl"
    calls = []
    snapshots = []

    async def process(case_id, *args, **kwargs):
        if calls:
            snapshots.append((jsonl(output), jsonl(errors)))
        calls.append(case_id)
        if case_id == failed_id:
            raise PipelineError(f"{case_id}: попытки исчерпаны")
        return result(case_id)

    extractor = mock_batch_runtime(monkeypatch, process)
    asyncio.run(runner.main(path, output, labels, mode="full"))
    assert calls == ["C1", "C2", "C3"]
    assert [row["case_id"] for row in jsonl(output)] == [i for i in calls if i != failed_id]
    assert jsonl(errors) == [{"case_id": failed_id, "error": f"{failed_id}: попытки исчерпаны"}]
    assert len(snapshots) == 2
    assert sum(len(rows) for rows in snapshots[0]) == 1
    assert sum(len(rows) for rows in snapshots[1]) == 2
    printed = capsys.readouterr().out
    assert "2/3 (66.7%)" in printed  # Ошибка не исчезает из знаменателя метрики.
    assert "Успешно: 2; с ошибкой: 1" in printed
    assert extractor.closed


def test_all_failures_produce_empty_predictions_and_complete_error_report(monkeypatch, tmp_path, capsys):
    from src.pipeline import PipelineError

    path, labels, output = batch_files(tmp_path, count=2)
    output.write_text(result("OLD").model_dump_json())

    async def process(case_id, *args, **kwargs):
        raise PipelineError(f"{case_id}: проверки не пройдены")

    mock_batch_runtime(monkeypatch, process)
    asyncio.run(runner.main(path, output, labels, mode="full"))
    assert jsonl(output) == []
    assert [row["case_id"] for row in jsonl(tmp_path / "predictions.errors.jsonl")] == ["C1", "C2"]
    printed = capsys.readouterr().out
    assert "0/2 (0.0%)" in printed
    assert "Успешно: 0; с ошибкой: 2" in printed


def test_successful_rerun_clears_old_failures(monkeypatch, tmp_path):
    path, labels, output = batch_files(tmp_path, count=1)
    errors = tmp_path / "predictions.errors.jsonl"
    errors.write_text('{"case_id":"OLD","error":"old"}\n')

    async def process(case_id, *args, **kwargs):
        return result(case_id)

    mock_batch_runtime(monkeypatch, process)
    asyncio.run(runner.main(path, output, labels, mode="full"))
    assert jsonl(errors) == []


def test_unexpected_bug_is_not_swallowed_and_previous_checkpoint_survives(monkeypatch, tmp_path):
    path, labels, output = batch_files(tmp_path)

    async def process(case_id, *args, **kwargs):
        if case_id == "C2":
            raise RuntimeError("unexpected bug")
        return result(case_id)

    extractor = mock_batch_runtime(monkeypatch, process)
    with pytest.raises(RuntimeError, match="unexpected bug"):
        asyncio.run(runner.main(path, output, labels, mode="full"))
    assert [row["case_id"] for row in jsonl(output)] == ["C1"]
    assert extractor.closed


def test_fail_fast_stops_before_next_case(monkeypatch, tmp_path):
    from src.pipeline import PipelineError

    path, labels, output = batch_files(tmp_path)
    calls = []

    async def process(case_id, *args, **kwargs):
        calls.append(case_id)
        raise PipelineError(f"{case_id}: проверки не пройдены")

    mock_batch_runtime(monkeypatch, process)
    with pytest.raises(PipelineError):
        asyncio.run(runner.main(path, output, labels, mode="full", fail_fast=True))
    assert calls == ["C1"]
    assert jsonl(tmp_path / "predictions.errors.jsonl")[0]["case_id"] == "C1"


def test_fail_fast_cli_defaults_to_continuing():
    parser = runner.build_parser()
    assert parser.parse_args([]).fail_fast is False
    assert parser.parse_args(["--fail-fast"]).fail_fast is True


def test_errors_path_cannot_overwrite_labels(tmp_path):
    path, _, output = batch_files(tmp_path)
    labels = tmp_path / "predictions.errors.jsonl"
    labels.write_text(result().model_dump_json())
    with pytest.raises(ValueError, match="совпадать"):
        asyncio.run(runner.main(path, output, labels, mode="full"))


def test_failed_checkpoint_replacement_keeps_previous_complete_file(monkeypatch, tmp_path):
    from pathlib import Path

    output = tmp_path / "predictions.jsonl"
    previous = result("OLD").model_dump_json() + "\n"
    output.write_text(previous)

    def fail(*args, **kwargs):
        raise OSError("cannot replace")

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError, match="cannot replace"):
        runner.write_results([result("NEW")], output)
    assert output.read_text() == previous
    assert list(tmp_path.iterdir()) == [output]

