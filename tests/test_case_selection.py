"""Выбор отдельных кейсов до создания клиентов и платных запросов."""

import asyncio
import json
import subprocess
import sys

import pytest

import src.main as runner
from src.llm import LLMUsage
from src.schemas.document_schemas import NoDocumentResult
from src.settings import DecisionModelSettings, PipelineSettings


def sample_result(case_id):
    return NoDocumentResult(
        case_id=case_id, document_type="none", confidence=1,
        fields={}, reason="Информационный вопрос",
    )


@pytest.fixture
def case_files(tmp_path):
    cases = tmp_path / "cases.jsonl"
    rows = [
        {"case_id": case_id, "split": split, "messages": [
            {"id": "m1", "role": "user", "text": f"Вопрос {case_id}"},
        ]}
        for case_id, split in [("C1", "dev"), ("C2", "test"), ("C3", "dev"), ("C4", "dev")]
    ]
    cases.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return cases, tmp_path / "labels.jsonl", tmp_path / "selected.jsonl"


def test_cli_accepts_repeated_case_ids():
    parser = runner.build_parser()
    assert parser.parse_args([]).case_ids is None
    args = parser.parse_args(["--case-id", "C3", "--case-id", "C1"])
    assert args.case_ids == ["C3", "C1"]


@pytest.mark.parametrize("mode", ["classification", "full", "llm-only"])
@pytest.mark.parametrize("limit,expected", [(None, ["C1", "C3"]), (1, ["C1"])])
def test_only_selected_cases_reach_inference_and_metrics(
    monkeypatch, case_files, capsys, mode, limit, expected,
):
    cases, labels, output = case_files
    # No labels for unselected cases: metrics must use the same subset as inference.
    labels.write_text("\n".join(sample_result(cid).model_dump_json() for cid in expected))
    seen = []

    async def process(case_id, messages, client, extractor, **kwargs):
        assert messages[0].text == f"Вопрос {case_id}"
        seen.append(case_id)
        return sample_result(case_id)

    async def classify(selected, client):
        seen.extend(case.case_id for case in selected)
        return [runner.DocumentTypeResult(
            case_id=case.case_id, document_type="none", confidence=1,
        ) for case in selected]

    class Extractor:
        usage = LLMUsage()

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(runner, "process_document", process)
    monkeypatch.setattr(runner, "classify_cases", classify)
    monkeypatch.setattr(runner, "DecisionModelSettings", lambda: DecisionModelSettings(
        _env_file=None, DECISION_API_KEY="offline-fixture",
    ))
    monkeypatch.setattr(runner, "DecisionModelClient", lambda **kwargs: object())
    monkeypatch.setattr(runner, "LLMSettings", lambda: object())
    monkeypatch.setattr(runner, "AsyncLLMExtractor", Extractor)
    monkeypatch.setattr(runner, "PipelineSettings", lambda: PipelineSettings(_env_file=None))

    output = asyncio.run(runner.main(
        cases, output, labels, mode=mode, verifier="decision", split="all",
        case_ids=["C3", "C1", "C3"], limit=limit,
    ))
    assert seen == expected  # Dataset order; repeated ids do not repeat API calls.
    assert [json.loads(line)["case_id"] for line in output.read_text().splitlines()] == expected
    assert f"Обработано кейсов: {len(expected)}" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["classification", "full", "llm-only"])
@pytest.mark.parametrize("split,ids,missing", [
    ("all", ["C1", "UNKNOWN"], "UNKNOWN"),
    ("dev", ["C2"], "C2"),
])
def test_unavailable_ids_fail_before_labels_settings_or_inference(
    monkeypatch, case_files, mode, split, ids, missing,
):
    cases, labels, output = case_files
    output.write_text("previous results\n")

    def forbidden(*args, **kwargs):
        pytest.fail("Selection errors must fail before labels or clients are loaded")

    for name in ("load_labels", "DecisionModelSettings", "LLMSettings"):
        monkeypatch.setattr(runner, name, forbidden)
    with pytest.raises(ValueError, match=missing):
        asyncio.run(runner.main(
            cases, output, labels, mode=mode, split=split, case_ids=ids, limit=1,
        ))
    assert output.read_text() == "previous results\n"


def test_cli_passes_case_ids_to_main_before_opening_labels(case_files):
    cases, labels, output = case_files
    completed = subprocess.run([
        sys.executable, "-m", "src.main", "--mode", "full",
        "--input", str(cases), "--labels", str(labels), "--output", str(output),
        "--split", "all", "--case-id", "UNKNOWN",
    ], capture_output=True, text=True, check=False)
    assert completed.returncode == 1
    assert "UNKNOWN" in completed.stderr
    assert not output.exists()
