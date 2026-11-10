"""Единый эксперимент: реальные SDK и пайплайн, HTTP полностью подменён."""

import asyncio
import json
from pathlib import Path

import httpx2
import pytest

import scripts.run_experiment as experiment
from src.settings import DecisionModelSettings, LLMSettings, PipelineSettings
from tests.test_paired_audit import case, label, proposal


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    inputs, labels = tmp_path / "cases.jsonl", tmp_path / "labels.jsonl"
    inputs.write_text(case().model_dump_json() + "\n")
    gold = label()
    gold.pop("scoring_only")
    labels.write_text(json.dumps(gold) + "\n")
    monkeypatch.setattr(experiment, "LLMSettings", lambda: LLMSettings(
        _env_file=None, LLM_API_KEY="PRIVATE_LLM_KEY", LLM_MODEL="fixture-llm",
        LLM_API_PROTOCOL="openai", LLM_RESPONSE_FORMAT="text", LLM_TEMPERATURE=0,
        LLM_INPUT_PRICE_PER_MILLION=.15, LLM_OUTPUT_PRICE_PER_MILLION=1.875))
    monkeypatch.setattr(experiment, "DecisionModelSettings", lambda: DecisionModelSettings(
        _env_file=None, DECISION_API_KEY="PRIVATE_DECISION_KEY", DECISION_MODEL="fixture-jev",
        DECISION_INPUT_PRICE_PER_MILLION=.042, DECISION_OUTPUT_PRICE_PER_MILLION=0))
    monkeypatch.setattr(experiment, "PipelineSettings", lambda: PipelineSettings(_env_file=None))
    return experiment.build_parser().parse_args([
        "--input", str(inputs), "--labels", str(labels), "--split", "all", "--limit", "1",
        "--output-dir", str(tmp_path / "experiment"), "--api-retries", "0",
    ])


@pytest.fixture
def http_models(monkeypatch, request):
    import src.client as client_module

    calls = {"llm": [], "verifier": []}
    behavior = getattr(request, "param", {})

    def llm_http(request):
        body = json.loads(request.content)
        calls["llm"].append(body)
        user = json.loads(body["messages"][-1]["content"])
        assert "scoring_only" not in json.dumps(user) and "GOLD_ONLY_MARKER" not in json.dumps(user)
        if "document_type" not in user:
            reply = {"document_type": "transfer_draft", "confidence": 1, "none_reason": None}
        else:
            result = proposal(200 if user.get("repair_fields") else 100)
            reply = result.model_dump(mode="json")
            if user.get("repair_fields"):
                reply["fields"] = {name: reply["fields"][name] for name in user["repair_fields"]}
        return httpx2.Response(200, json={"usage": {"prompt_tokens": 1000, "completion_tokens": 100},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(reply)}}]})

    def verifier_http(request):
        body = json.loads(request.content)
        calls["verifier"].append(body)
        if behavior.get("verifier_status") and len(calls["verifier"]) <= behavior.get("fail_count", 1):
            return httpx2.Response(behavior["verifier_status"], json={"error": "PRIVATE_SERVER_BODY"})
        if "document_type" in body["questions"]:
            answers = {
                "document_type": {"type": "choice", "choice": "transfer_draft", "confidence": 1,
                                  "probabilities": {"invoice_draft": 0, "transfer_draft": 1, "none": 0}},
                "none_reason": {"type": "choice", "choice": "no_request", "confidence": 1,
                                "probabilities": {"no_request": 1, "other_document": 0, "cancelled": 0}},
            }
        else:
            answers = {key: {"type": "noul", "noul": .99 if key == "amount_rub::superseded"
                and question["instructions"]["field"]["value"] == 100 else .01}
                for key, question in body["questions"].items()}
        return httpx2.Response(200, json={"model": "fixture", "usage": {"input_tokens": 2000, "output_tokens": 100},
                                         "answers": answers})

    real_extractor = experiment.BudgetedExtractor
    real_sdk = client_module.AsyncTypeSafeClient

    def extractor(settings, budget):
        return real_extractor(settings, budget, client=httpx2.AsyncClient(transport=httpx2.MockTransport(llm_http)))

    def sdk(**kwargs):
        return real_sdk(**kwargs, transport=httpx2.MockTransport(verifier_http))

    monkeypatch.setattr(experiment, "BudgetedExtractor", extractor)
    monkeypatch.setattr(client_module, "AsyncTypeSafeClient", sdk)
    monkeypatch.setattr(experiment, "AsyncTypeSafeClient", sdk)
    return calls


def test_default_is_a_plan_and_never_calls_models(prepared, http_models, capsys):
    assert asyncio.run(experiment.run_experiment(prepared)) is None
    assert http_models == {"llm": [], "verifier": []}
    assert not prepared.output_dir.exists()
    assert "--live" in capsys.readouterr().out


def test_two_rounds_compare_whole_modes_and_paired_seed_separately(prepared, http_models):
    prepared.live, prepared.rounds = True, 2
    report_path = asyncio.run(experiment.run_experiment(prepared))
    report = json.loads(report_path.read_text())
    assert report["status"] == "completed"
    assert [phase["mode"] for phase in report["phases"]] == ["llm-only", "full", "paired", "full", "llm-only", "paired"]
    assert all(phase["status"] == "completed" for phase in report["phases"])
    assert report["mode_comparison"]["llm-only"]["exact_document_correct"] == 0
    assert report["mode_comparison"]["full"]["exact_document_correct"] == 2
    assert report["mode_comparison"]["full"]["total_cases"] == 2
    assert report["paired_audit"]["field_changes"]["corrected"] == 2
    assert report["paired_audit"]["error_detection"]["tp"] == 2
    assert report["paired_audit"]["field_outcomes"]["corrected"] == 2
    assert len(http_models["llm"]) == 14  # На раунд: 2 llm-only + 2 full + 3 paired.
    assert len(http_models["verifier"]) == 10  # На раунд: 3 full + 2 paired.
    assert report["calls"] == {"llm": 14, "verifier": 10}
    assert report["costs"]["actual_total_usd"] == pytest.approx(.005565)
    assert report["costs"]["llm-only_usd"] == pytest.approx(.00135)
    assert report["costs"]["full_usd"] == pytest.approx(.001854)
    assert report["costs"]["paired_audit_usd"] == pytest.approx(.002361)
    config = json.loads((prepared.output_dir / "config.json").read_text())
    assert config["settings"]["llm"]["model"] == "fixture-llm"
    assert config["settings"]["pipeline"]["wrong_threshold"] == .8
    assert (prepared.output_dir / "snapshot/src/pipeline.py").is_file()
    for path in prepared.output_dir.rglob("*.json"):
        text = path.read_text()
        assert "PRIVATE_LLM_KEY" not in text and "PRIVATE_DECISION_KEY" not in text


def test_budget_is_shared_across_modes_and_stops_before_extra_http(prepared, http_models):
    prepared.live, prepared.max_llm_calls = True, 2
    report = json.loads(asyncio.run(experiment.run_experiment(prepared)).read_text())
    assert report["status"] == "stopped"
    assert report["stop_reason"] == "budget_exhausted"
    assert len(http_models["llm"]) == 2
    assert report["mode_comparison"]["full"]["total_cases"] == 1
    assert report["mode_comparison"]["full"]["predicted_cases"] == 0
    assert report["paired_audit"]["not_attempted"] == ["r001/C1"]


def test_existing_directory_and_invalid_selection_fail_before_http(prepared, http_models):
    prepared.live = True
    prepared.output_dir.mkdir()
    sentinel = prepared.output_dir / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        asyncio.run(experiment.run_experiment(prepared))
    assert sentinel.read_text() == "keep" and not http_models["llm"]


def test_source_change_stops_the_next_phase_and_preserves_partial_results(prepared, http_models, monkeypatch):
    prepared.live = True
    original = experiment.run_mode

    async def changed(*args, **kwargs):
        await original(*args, **kwargs)
        prepared.input.write_text(prepared.input.read_text() + "\n")

    monkeypatch.setattr(experiment, "run_mode", changed)
    with pytest.raises(experiment.ExperimentChangedError) as failure:
        asyncio.run(experiment.run_experiment(prepared))
    report = json.loads((prepared.output_dir / "report.json").read_text())
    assert report["status"] == "interrupted"
    assert report["stop_reason"] == "ExperimentChangedError"
    assert report["phases"][0]["status"] == "completed"
    assert report["phases"][1]["status"] == "not_started"
    assert report["changed_files"] == failure.value.changed_files
    assert report["changed_files"] == [str(prepared.input.resolve())]
    assert failure.value.report_path == prepared.output_dir / "report.json"


def test_active_mode_has_a_running_checkpoint_and_a_visible_artifact_path(prepared, http_models, monkeypatch):
    prepared.live = True
    process = experiment.process_document
    observations = []

    async def inspect(*args, **kwargs):
        report = json.loads((prepared.output_dir / "report.json").read_text())
        phase, = [phase for phase in report["phases"] if phase["status"] == "running"]
        manifest = json.loads((Path(phase["run_directory"]) / "run.json").read_text())
        assert manifest["status"] == "running"
        observations.append(phase["mode"])
        return await process(*args, **kwargs)

    monkeypatch.setattr(experiment, "process_document", inspect)
    asyncio.run(experiment.run_experiment(prepared))
    assert observations == ["llm-only", "full"]


def test_unknown_case_is_rejected_before_artifacts_or_http(prepared, http_models):
    prepared.live, prepared.case_ids = True, ["UNKNOWN"]
    with pytest.raises(ValueError):
        asyncio.run(experiment.run_experiment(prepared))
    assert not prepared.output_dir.exists()
    assert http_models == {"llm": [], "verifier": []}


@pytest.mark.parametrize("http_models", [{"verifier_status": 500}], indirect=True)
def test_transport_retry_is_counted_and_server_error_responses_keep_the_cost_exact(prepared, http_models):
    prepared.live, prepared.api_retries = True, 1
    report = json.loads(asyncio.run(experiment.run_experiment(prepared)).read_text())
    assert report["status"] == "completed"
    assert report["calls"] == {"llm": 7, "verifier": 6}
    assert report["mode_comparison"]["full"]["exact_document_correct"] == 1
    # HTTP 500 — ответ сервера с отказом, а не потерянный запрос: цена остаётся точной.
    assert report["costs"]["actual_total_usd"] > 0
    assert report["costs"]["is_lower_bound"] is False


@pytest.mark.parametrize("http_models", [{"verifier_status": 401}], indirect=True)
def test_auth_error_stops_other_phases_without_recording_server_body(prepared, http_models):
    prepared.live = True
    report = json.loads(asyncio.run(experiment.run_experiment(prepared)).read_text())
    assert report["status"] == "stopped" and report["stop_reason"] == "service_error"
    assert len(http_models["llm"]) == 2 and len(http_models["verifier"]) == 1
    assert report["phases"][2]["status"] == "not_started"
    for path in prepared.output_dir.rglob("*.json*"):
        assert "PRIVATE_SERVER_BODY" not in path.read_text()


@pytest.mark.parametrize("http_models", [{"verifier_status": 500}], indirect=True)
def test_verifier_budget_blocks_a_retry_before_sdk_call(prepared, http_models):
    prepared.live, prepared.api_retries, prepared.max_verifier_calls = True, 2, 1
    report = json.loads(asyncio.run(experiment.run_experiment(prepared)).read_text())
    assert report["status"] == "stopped" and report["stop_reason"] == "budget_exhausted"
    assert len(http_models["verifier"]) == report["calls"]["verifier"] == 1


def test_drift_diagnostic_lists_all_changed_files_without_showing_contents(prepared):
    plan = experiment.prepare_plan(prepared)
    prepared.input.write_text(prepared.input.read_text() + "\n")
    prepared.labels.write_text(prepared.labels.read_text() + "\n")
    with pytest.raises(experiment.ExperimentChangedError) as failure:
        experiment.assert_unchanged(plan)
    assert set(failure.value.changed_files) == {str(prepared.input.resolve()), str(prepared.labels.resolve())}
    assert "cases.jsonl" in str(failure.value) and "labels.jsonl" in str(failure.value)
    assert "PRIVATE_LLM_KEY" not in str(failure.value)


def test_cli_explains_local_drift_and_points_to_saved_report(monkeypatch, tmp_path):
    error = experiment.ExperimentChangedError(["src/detection_metrics.py"])
    error.report_path = tmp_path / "report.json"

    async def changed(args):
        raise error

    monkeypatch.setattr(experiment, "run_experiment", changed)
    with pytest.raises(SystemExit) as failure:
        experiment.main([])
    text = str(failure.value)
    assert "src/detection_metrics.py" in text
    assert str(error.report_path) in text
    assert "API" not in text


def test_cli_still_hides_unclassified_server_error_bodies(monkeypatch):
    async def broken(args):
        raise RuntimeError("PRIVATE_SERVER_BODY")

    monkeypatch.setattr(experiment, "run_experiment", broken)
    with pytest.raises(SystemExit) as failure:
        experiment.main([])
    assert "RuntimeError" in str(failure.value)
    assert "PRIVATE_SERVER_BODY" not in str(failure.value)
