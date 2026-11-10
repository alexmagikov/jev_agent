"""Повторы транспорта и продолжение аудита без пересэмплирования исходных ответов."""

import asyncio
import json
from copy import deepcopy

import pytest
from typesafe_sdk import TypeSafeAPITimeoutError

from scripts.audit_jev import APICallBudget, audit_cases, build_report
from src.extraction import ExtractionError
from tests.test_paired_audit import case, proposal, label, Extractor, Jev


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch):
    async def sleep(delay):
        assert 0 <= delay <= 60
    monkeypatch.setattr("scripts.audit_jev.asyncio.sleep", sleep)


def test_temporary_verifier_retry_uses_same_question_and_counts_each_request():
    class Flaky(Jev):
        requests = []
        async def ainvoke(self, state, questions, response_model):
            self.requests.append((state.model_dump(mode="json"), questions.model_dump(mode="json")))
            if len(self.requests) == 1:
                raise TypeSafeAPITimeoutError(120)
            return await super().ainvoke(state, questions, response_model)

    budget, verifier = APICallBudget(3), Flaky()
    records = asyncio.run(audit_cases([case()], Extractor(), verifier, api_retries=1, verifier_budget=budget))
    assert records[0]["status"] == "complete"
    assert verifier.requests[0] == verifier.requests[1]
    assert budget.used == len(verifier.requests) == 3
    assert len(records[0]["jev_rounds"]) == 3
    assert records[0]["jev_rounds"][0]["error_type"] == "TypeSafeAPITimeoutError"


def test_exhausted_temporary_error_does_not_stop_next_case():
    class Flaky(Jev):
        calls = 0
        async def ainvoke(self, *args):
            self.calls += 1
            if self.calls <= 2:
                raise TypeSafeAPITimeoutError(120)
            return await super().ainvoke(*args)

    budget = APICallBudget(5)
    records = asyncio.run(audit_cases([case(), case("C2")], Extractor(), Flaky(),
                                     api_retries=1, verifier_budget=budget))
    assert [r["status"] for r in records] == ["api_error", "complete"]
    assert records[0]["before"] is not None and records[0]["after"] is None
    assert budget.used == 3
    report = build_report(records, [label(), {**label(), "case_id": "C2"}])
    assert report["verification_quality"]["api_error_cases"] == ["C1"]
    assert report["verification_quality"]["cases"] == ["C2"]


def test_retry_cannot_exceed_verifier_budget():
    class Unavailable:
        calls = 0
        async def ainvoke(self, *args):
            self.calls += 1
            raise TypeSafeAPITimeoutError(120)

    verifier, budget = Unavailable(), APICallBudget(1)
    records = asyncio.run(audit_cases([case(), case("C2")], Extractor(), verifier,
                                     api_retries=2, verifier_budget=budget))
    assert verifier.calls == budget.used == 1
    assert len(records) == 1 and records[0]["status"] == "budget_exhausted"


def test_invalid_baseline_is_not_an_api_error_and_does_not_stop_batch():
    class InvalidFirst(Extractor):
        async def extract(self, *args, **kwargs):
            if self.classifications == 1:
                raise ExtractionError("PRIVATE_QUOTE", code="invalid_source")
            return proposal(200)

    records = asyncio.run(audit_cases([case(), case("C2")], InvalidFirst(), Jev()))
    assert [r["status"] for r in records] == ["baseline_rejected", "complete"]
    assert records[0]["failure_reason"] == "invalid_extraction"
    assert len(records[0]["validation_failures"]) == 2
    assert "PRIVATE_QUOTE" not in json.dumps(records[0]["validation_failures"])


def test_resume_keeps_completed_pair_and_rechecks_last_candidate_without_llm():
    complete = asyncio.run(audit_cases([case()], Extractor(), Jev()))[0]
    interrupted = deepcopy(complete)
    interrupted.update(case_id="C2", after=None, status="api_error", enhanced_attempts=2,
                       error_type="TypeSafeAPITimeoutError")
    interrupted["before"]["case_id"] = "C2"
    interrupted["jev_rounds"][-1].pop("answers")

    class NoLLM:
        async def classify(self, *args, **kwargs):
            pytest.fail("Saved classification must be reused")
        async def extract(self, *args, **kwargs):
            pytest.fail("Validated last candidate must be reused")

    budget, verifier = APICallBudget(1), Jev()
    saved = {"C1": complete, "C2": interrupted}
    records = asyncio.run(audit_cases([case(), case("C2")], NoLLM(), verifier,
                                     resume_records=saved, verifier_budget=budget))
    assert len(records) == 2
    assert records[0]["after"] == complete["after"]
    assert records[1]["before"] == interrupted["before"]
    assert records[1]["after"]["fields"]["amount_rub"]["value"] == 200
    assert records[1]["seed_proposal"] == complete["seed_proposal"]
    assert budget.used == len(verifier.states) == 1
    assert interrupted["after"] is None  # Source trace data is not mutated.


def test_resume_replays_pending_repair_with_original_base_and_feedback():
    source = asyncio.run(audit_cases([case()], Extractor(), Jev()))[0]
    source.update(after=None, status="api_error", enhanced_attempts=2, stage="repair")
    source["candidates"] = source["candidates"][:1]
    source.pop("enhanced_proposal", None)
    source["pending_extraction"] = {
        "feedback": ["amount_rub::superseded: saved feedback"],
        "previous_proposal": source["seed_proposal"], "repair_fields": ["amount_rub"],
        "repair_paths": ["amount_rub"], "previous_checked_fields": source["before"]["fields"],
    }
    calls = []

    class Replay:
        async def classify(self, *args, **kwargs):
            pytest.fail("Do not replace saved classification")
        async def extract(self, *args, **kwargs):
            calls.append(kwargs)
            assert kwargs["previous_proposal"] == proposal(100)
            assert kwargs["feedback"] == source["pending_extraction"]["feedback"]
            assert kwargs["repair_paths"] == ["amount_rub"]
            return proposal(200)

    records = asyncio.run(audit_cases([case()], Replay(), Jev(), resume_records={"C1": source}))
    assert len(calls) == 1
    assert records[0]["before"] == source["before"]
    assert records[0]["after"]["fields"]["amount_rub"]["value"] == 200
    assert records[0]["enhanced_attempts"] == 2
    assert records[0]["pending_extraction"] is None


def test_resume_retains_later_completed_pairs_when_budget_is_exhausted():
    complete = asyncio.run(audit_cases([case()], Extractor(), Jev()))[0]
    later = deepcopy(complete)
    later["case_id"] = later["before"]["case_id"] = later["after"]["case_id"] = "C3"
    records = asyncio.run(audit_cases([case(), case("C2"), case("C3")], Extractor(), Jev(),
                                     max_jev_calls=1, resume_records={"C3": later}))
    assert [r["case_id"] for r in records] == ["C1", "C3"]
    assert records[0]["status"] == "budget_exhausted"
    assert records[1] == later


def test_resume_keeps_semantic_refusals_and_counts_them_as_quality_losses():
    source = asyncio.run(audit_cases([case()], Extractor(), Jev(), max_attempts=1))[0]
    assert source["status"] == "rejected"
    records = asyncio.run(audit_cases([case()], None, None, resume_records={"C1": source}))
    assert records == [source]
    quality = build_report(records, [label(100)])["verification_quality"]
    assert quality["cases"] == quality["rejected_cases"] == ["C1"]
    assert quality["before"]["correct_expected_values"] == 1
    assert quality["after"]["correct_expected_values"] == 0


def test_legacy_rejection_is_not_counted_as_api_failure_when_rescoring():
    source = asyncio.run(audit_cases([case()], Extractor(), Jev(), max_attempts=1))[0]
    source["status"] = "api_error"  # Старый формат ошибочно называл отказ API-сбоем.
    quality = build_report([source], [label(100)])["verification_quality"]
    assert quality["api_error_cases"] == []
    assert quality["rejected_cases"] == ["C1"]


def test_resume_cli_skips_completed_pairs_and_counts_only_new_calls(tmp_path, monkeypatch):
    from src.llm import LLMUsage
    from src.settings import LLMSettings, DecisionModelSettings
    import scripts.audit_jev as module

    complete = asyncio.run(audit_cases([case()], Extractor(), Jev()))[0]
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps(complete))
    original = trace.read_bytes()
    input_path, labels_path = tmp_path / "cases.jsonl", tmp_path / "labels.jsonl"
    input_path.write_text(case().model_dump_json())
    target = label()
    target.pop("scoring_only")
    labels_path.write_text(json.dumps(target))

    class SDK:
        def __init__(self, **kwargs):
            assert kwargs["timeout"] == 120
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def system_one(self, **kwargs):
            pytest.fail("Completed cases must not call API")

    class NoGeneration:
        usage = LLMUsage()
        def __init__(self, *args):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def classify(self, *args, **kwargs):
            pytest.fail("Completed cases must not classify again")

    monkeypatch.setattr(module, "LLMSettings", lambda: LLMSettings(_env_file=None, LLM_API_KEY="fake", LLM_MODEL="small"))
    monkeypatch.setattr(module, "DecisionModelSettings", lambda: DecisionModelSettings(_env_file=None, DECISION_API_KEY="fake"))
    monkeypatch.setattr(module, "AsyncTypeSafeClient", SDK)
    monkeypatch.setattr(module, "BudgetedExtractor", NoGeneration)
    output = tmp_path / "resumed"
    args = module.build_parser().parse_args([
        "--live", "--verifier", "decision", "--resume-trace", str(trace),
        "--input", str(input_path), "--labels", str(labels_path), "--output-dir", str(output),
    ])
    assert asyncio.run(module.run_live(args))
    report = json.loads((output / "report.json").read_text())
    assert report["reused_cases"] == ["C1"]
    assert report["llm_http_calls"] == report["verifier_api_calls"] == 0
    assert report["usage_scope"] == "current_run_only"
    assert trace.read_bytes() == original
    assert not report["resume_config_verified"]

    # Новые запуски сохраняют config до API: следующее продолжение проверяет его целиком.
    args.resume_trace = output / "trace.jsonl"
    args.output_dir = tmp_path / "resumed-again"
    assert asyncio.run(module.run_live(args))
    assert json.loads((args.output_dir / "report.json").read_text())["resume_config_verified"]
    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    config["extraction_prompt_sha256"] = "different-prompt"
    config_path.write_text(json.dumps(config))
    args.output_dir = tmp_path / "must-not-create"
    with pytest.raises(ValueError, match="extraction_prompt_sha256"):
        asyncio.run(module.run_live(args))
    assert not args.output_dir.exists()


@pytest.mark.parametrize("status", [401, 403, 429])
def test_service_wide_failure_stops_new_cases_without_logging_server_body(status):
    import httpx2
    from typesafe_sdk import TypeSafeAPIError

    class Unavailable:
        calls = 0
        async def ainvoke(self, *args):
            self.calls += 1
            raise TypeSafeAPIError(status, {"private": "SERVER_SECRET"}, httpx2.Headers())

    verifier = Unavailable()
    records = asyncio.run(audit_cases([case(), case("C2")], Extractor(), verifier))
    assert len(records) == verifier.calls == 1
    assert records[0]["status"] == "api_error"
    assert "SERVER_SECRET" not in json.dumps(records)


def test_llm_auth_failure_stops_batch():
    import httpx2
    from scripts.audit_jev import BudgetedExtractor
    from src.settings import LLMSettings

    def respond(request):
        return httpx2.Response(401, text="private error")
    budget = APICallBudget(2)
    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="fake", LLM_MODEL="small")
        async with BudgetedExtractor(settings, budget) as extractor:
            extractor._client = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
            return await audit_cases([case(), case("C2")], extractor, Jev())

    records = asyncio.run(run())
    assert len(records) == budget.used == 1
    assert records[0]["status"] == "api_error"
    assert "private error" not in json.dumps(records)


def test_resume_uses_received_verifier_answer_without_resampling():
    source = asyncio.run(audit_cases([case()], Extractor(), Jev()))[0]
    source.update(status="started", after=None)
    records = asyncio.run(audit_cases([case()], None, None, resume_records={"C1": source}))
    assert records[0]["status"] == "complete"
    assert records[0]["after"]["fields"]["amount_rub"]["value"] == 200
    assert len(records[0]["jev_rounds"]) == len(source["jev_rounds"])


def test_llm_transport_retry_reuses_payload_and_respects_http_budget(monkeypatch):
    import httpx2
    from scripts.audit_jev import BudgetedExtractor, audit_llm_settings
    from src.settings import LLMSettings

    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx2.Response(503)
        return httpx2.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "document_type": "none", "confidence": 0.9, "none_reason": "no_request",
        })}}]})

    settings = audit_llm_settings(LLMSettings(_env_file=None, LLM_API_KEY="fake", LLM_MODEL="small"), api_retries=1)
    budget = APICallBudget(2)
    async def run():
        async with BudgetedExtractor(settings, budget) as extractor:
            extractor._client = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
            return await extractor.classify(case().messages)

    assert asyncio.run(run()).document_type == "none"
    assert requests[0] == requests[1]
    assert budget.used == 2


def test_resume_does_not_add_semantic_attempts_after_last_verifier_timeout():
    source = asyncio.run(audit_cases([case()], Extractor(), Jev()))[0]
    source.update(status="api_error", after=None, enhanced_proposal=proposal(100).model_dump(mode="json"),
                  enhanced_attempts=2, candidates=source["candidates"][:1])
    source["jev_rounds"][-1].pop("answers")
    records = asyncio.run(audit_cases([case()], None, Jev(), resume_records={"C1": source}))
    assert records[0]["status"] == "rejected"
    assert records[0]["enhanced_attempts"] == 2


@pytest.mark.parametrize("corruption", ["messages", "baseline", "candidate"])
def test_resume_rejects_changed_source_or_saved_values_before_any_api(corruption):
    source = asyncio.run(audit_cases([case()], Extractor(), Jev()))[0]
    if corruption == "messages":
        source["messages"][0]["text"] += " changed"
    elif corruption == "baseline":
        source["before"]["fields"]["amount_rub"]["value"] = 123
    else:
        source["candidates"][-1]["checked_fields"]["amount_rub"]["value"] = 123
    with pytest.raises(ValueError):
        asyncio.run(audit_cases([case()], None, None, resume_records={"C1": source}))


class Repaired(Extractor):
    async def extract(self, *args, **kwargs):
        return proposal(200)


def test_resume_does_not_record_the_rechecked_candidate_twice():
    class Down(Jev):
        async def ainvoke(self, *args):
            raise TypeSafeAPITimeoutError(120)

    interrupted = asyncio.run(audit_cases([case()], Extractor(), Down(), api_retries=0))[0]
    assert interrupted["status"] == "api_error"
    record = asyncio.run(audit_cases([case()], Repaired(), Jev(), resume_records={"C1": interrupted}))[0]
    assert record["status"] == "complete"
    assert [candidate["proposal"] for candidate in record["candidates"]] == [
        proposal(100).model_dump(mode="json"), proposal(200).model_dump(mode="json")]
    assert [entry["candidate_index"] for entry in record["jev_rounds"]] == [0, 0, 1]
    detection = build_report([record], [label()])["error_detection"]
    assert (detection["evaluated_cases"], detection["tp"], detection["skipped_cases"]) == (1, 1, [])


def test_resume_with_received_final_answer_keeps_the_rejected_field_outcome():
    class StillWrong(Extractor):
        async def extract(self, *args, **kwargs):
            return proposal(100)

    rejected = asyncio.run(audit_cases([case()], StillWrong(), Jev()))[0]
    expected = build_report([rejected], [label()])["field_outcomes"]
    assert expected["detected_and_rejected"] == 1
    crashed = {key: value for key, value in deepcopy(rejected).items()
               if key not in ("error_type", "failure_reason", "validation_failures")}
    crashed["status"] = "started"  # процесс упал после ответа Jev, до записи итога
    record = asyncio.run(audit_cases([case()], None, None, resume_records={"C1": crashed}))[0]
    assert record["status"] == "rejected"
    assert build_report([record], [label()])["field_outcomes"] == expected


def test_unusable_verifier_answer_is_not_saved_as_a_verdict_and_is_asked_again_on_resume():
    class Incomplete(Jev):
        async def ainvoke(self, state, questions, response_model):
            keys = list(questions.root)[1:]
            return response_model.model_validate({"answers": {key: {"noul": 0.01} for key in keys}})

    interrupted = asyncio.run(audit_cases([case()], Extractor(), Incomplete(), api_retries=0))[0]
    assert interrupted["status"] == "api_error"
    assert "answers" not in interrupted["jev_rounds"][-1]
    verifier = Jev()
    record = asyncio.run(audit_cases([case()], Repaired(), verifier, resume_records={"C1": interrupted}))[0]
    assert record["status"] == "complete"
    assert len(verifier.states) == 2
    detection = build_report([record], [label()])["error_detection"]
    assert (detection["evaluated_cases"], detection["tp"], detection["skipped_cases"]) == (1, 1, [])
