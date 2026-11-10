"""Долгий сбой API останавливает прогон, а не расходует лимит вызовов на все кейсы."""

import asyncio
import json

import httpx2
import pytest
from typesafe_sdk import TypeSafeAPIConnectionError, TypeSafeAPIError

from scripts.audit_jev import audit_cases, service_unavailable
from src.llm import AsyncLLMExtractor, LLMError, LLMResponseError, LLMUnavailableError
from src.pipeline import PipelineError
from src.settings import LLMSettings
from tests.test_experiment import experiment, http_models, prepared  # noqa: F401
from tests.test_paired_audit import Extractor, Jev, case, label


def test_llm_without_response_after_retries_is_reported_as_unavailable():
    from src.cases.document_type import DocumentMessage

    def offline(request):
        raise httpx2.ConnectError("fixture")

    settings = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture",
                           LLM_BASE_URL="https://example.com/v1", LLM_MAX_RETRIES=0)

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(offline)) as client:
            await AsyncLLMExtractor(settings, client=client).classify(
                [DocumentMessage(id="m1", role="user", text="Привет")])

    with pytest.raises(LLMUnavailableError) as raised:
        asyncio.run(run())
    assert isinstance(raised.value, LLMError)


def wrapped(error):
    try:
        raise PipelineError("case failed") from error
    except PipelineError as pipeline_error:
        return pipeline_error


@pytest.mark.parametrize("error,expected", [
    (wrapped(LLMUnavailableError("fixture")), True),
    (TypeSafeAPIConnectionError("fixture"), True),
    (wrapped(TypeSafeAPIError(503, None, httpx2.Headers())), True),
    (wrapped(TypeSafeAPIError(400, None, httpx2.Headers())), False),
    (wrapped(LLMResponseError("fixture")), False),
    (RuntimeError("fixture"), False),
])
def test_service_unavailable_detects_only_exhausted_transport_failures(error, expected):
    assert service_unavailable(error) is expected


class FlakyJev(Jev):
    """Недоступен для перечисленных кейсов, для остальных отвечает как обычно."""

    def __init__(self, offline_cases):
        super().__init__()
        self.offline_cases = offline_cases
        self.current = None

    async def ainvoke(self, state, questions, response_model):
        if self.current in self.offline_cases:
            raise TypeSafeAPIConnectionError("fixture")
        return await super().ainvoke(state, questions, response_model)


def run_audit(case_ids, offline_cases):
    jev = FlakyJev(offline_cases)

    def track(record):
        jev.current = record["case_id"]

    return asyncio.run(audit_cases([case(case_id) for case_id in case_ids], Extractor(), jev,
                                   api_retries=0, on_event=track))


def test_audit_stops_after_two_consecutive_outages():
    records = run_audit(["C1", "C2", "C3"], {"C1", "C2", "C3"})
    assert [record["case_id"] for record in records] == ["C1", "C2"]
    assert all(record["status"] == "api_error" for record in records)


def test_successful_case_between_outages_resets_the_counter():
    records = run_audit(["C1", "C2", "C3"], {"C1", "C3"})
    assert [record["status"] for record in records] == ["api_error", "complete", "api_error"]


@pytest.mark.parametrize("http_models", [{"verifier_status": 503, "fail_count": 1000}], indirect=True)
def test_experiment_mode_stops_after_two_consecutive_outages(prepared, http_models, tmp_path):
    cases = [case(case_id) for case_id in ("C1", "C2", "C3")]
    labels = []
    for case_id in ("C1", "C2", "C3"):
        gold = label()
        gold.pop("scoring_only")
        labels.append({**gold, "case_id": case_id})
    prepared.input.write_text("".join(item.model_dump_json() + "\n" for item in cases))
    prepared.labels.write_text("".join(json.dumps(row) + "\n" for row in labels))
    prepared.live, prepared.limit = True, 3
    report = json.loads(asyncio.run(experiment.run_experiment(prepared)).read_text())
    assert report["status"] == "stopped" and report["stop_reason"] == "service_unavailable"
    full, = [phase for phase in report["phases"] if phase["mode"] == "full"]
    assert full["status"] == "stopped"
    assert len(http_models["verifier"]) == 2  # третий кейс уже не отправлялся


def chained(error):
    try:
        raise PipelineError("case failed") from error
    except PipelineError as pipeline_error:
        return pipeline_error


@pytest.mark.parametrize("error,expected", [
    (chained(LLMUnavailableError("fixture")), True),
    (chained(LLMError("HTTP 400", status_code=400)), True),
    (chained(LLMError("LLM отказала в ответе")), False),
    (chained(LLMResponseError("fixture")), False),
    (TypeSafeAPIConnectionError("fixture"), True),
    (PipelineError("классификация не прошла за 2 попытки"), False),
])
def test_api_failure_separates_service_errors_from_model_errors(error, expected):
    from scripts.audit_jev import api_failure

    assert api_failure(error) is expected


def test_api_failure_covers_decision_model_and_verifier_response_errors():
    from scripts.audit_jev import api_failure
    from src.client import DecisionModelError, DecisionModelRateLimitError
    from src.pipeline import PipelineRejectionError
    from src.verification import VerificationError

    assert api_failure(chained(DecisionModelError("fixture")))
    assert api_failure(chained(DecisionModelRateLimitError("fixture")))
    assert api_failure(chained(VerificationError("fixture")))
    assert not api_failure(PipelineRejectionError("fixture", failure_reason="x", validation_failures=[]))


@pytest.mark.parametrize("http_models", [{"verifier_status": 503, "fail_count": 1}], indirect=True)
def test_api_errors_are_recorded_and_excluded_pairwise_from_the_analysis(prepared, http_models):
    import scripts.analyze_experiment as analysis

    cases = [case(case_id) for case_id in ("C1", "C2")]
    labels = []
    for case_id in ("C1", "C2"):
        gold = label()
        gold.pop("scoring_only")
        labels.append({**gold, "case_id": case_id})
    prepared.input.write_text("".join(item.model_dump_json() + "\n" for item in cases))
    prepared.labels.write_text("".join(json.dumps(row) + "\n" for row in labels))
    prepared.live, prepared.limit = True, 2
    report_path = asyncio.run(experiment.run_experiment(prepared))
    report = json.loads(report_path.read_text())
    full, = [phase for phase in report["phases"] if phase["mode"] == "full"]
    errors = [json.loads(line) for line in next(
        __import__("pathlib").Path(full["run_directory"]).glob("*.errors.jsonl")).read_text().splitlines()]
    assert errors == [{"case_id": "C1", "error": "PipelineError", "kind": "api_error"}]
    result = analysis.analyze(report_path.parent)
    assert result["excluded_api_error"] == ["C1"]
    assert result["cases"] == 1
    assert "warning" in result  # исключено 50% > 10%
