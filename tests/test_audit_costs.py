"""Стоимость парного аудита: общий этап оплачивается один раз, usage не выдумывается."""

import asyncio
import json

import httpx2
import pytest

import scripts.audit_jev as audit
from src.client import DecisionUsage
from src.llm import LLMUsage, LLMError
from src.settings import DecisionModelSettings, LLMSettings, PipelineSettings
from src.verification import VerificationResponse
from tests.test_paired_audit import Extractor, Jev, case, label, proposal


def prices():
    return DecisionModelSettings(_env_file=None, DECISION_API_KEY="fixture",
                                 DECISION_INPUT_PRICE_PER_MILLION=.042,
                                 DECISION_OUTPUT_PRICE_PER_MILLION=0)


class MeteredExtractor(Extractor):
    def __init__(self, fail=False):
        super().__init__()
        self.usage = LLMUsage(estimated_cost=0)
        self.fail = fail

    def charge(self, amount):
        self.usage.requests += 1
        if self.fail:
            raise LLMError("fixture")
        self.usage.responses += 1
        self.usage.usage_reports += 1
        self.usage.input_tokens += 100
        self.usage.output_tokens += 10
        self.usage.estimated_cost += amount

    async def classify(self, *args, **kwargs):
        self.charge(.001)
        return await super().classify(*args, **kwargs)

    async def extract(self, *args, **kwargs):
        self.charge(.003 if kwargs.get("repair_fields") else .002)
        result = await super().extract(*args, **kwargs)
        return proposal(200) if kwargs.get("repair_fields") else result


class MeteredJev(Jev):
    def __init__(self):
        super().__init__()
        self.usage = DecisionUsage()

    async def ainvoke(self, *args, **kwargs):
        self.usage.sdk_calls += 1
        self.usage.responses += 1
        self.usage.usage_reports += 1
        self.usage.input_tokens += 1000
        self.usage.output_tokens += 100
        return await super().ainvoke(*args, **kwargs)


def test_actual_requests_are_attributed_to_phases_and_shared_cost_is_not_doubled():
    events = []
    records = asyncio.run(audit.audit_cases([case()], MeteredExtractor(), MeteredJev(), on_usage=events.append))
    assert [event["phase"] for event in events] == ["classification", "baseline", "verification", "enhanced", "verification"]
    assert records[0]["usage_events"] == events
    costs = audit.build_cost_comparison(events, prices())
    assert costs["llm_only_usd"] == pytest.approx(.003)
    assert costs["extra_full_usd"] == pytest.approx(.003084)
    assert costs["full_usd"] == pytest.approx(.006084)
    assert costs["audit_total_usd"] == costs["full_usd"]
    assert costs["audit_total_usd"] != costs["llm_only_usd"] + costs["full_usd"]
    assert costs["full_to_llm_only_ratio"] == pytest.approx(2.028)
    assert costs["phases"]["verification"]["usage"]["input_tokens"] == 2000
    assert costs["phases"]["enhanced"]["usage"]["requests"] == 1


def test_a_failed_api_request_marks_its_cost_as_a_lower_bound_instead_of_free():
    events = []
    records = asyncio.run(audit.audit_cases([case()], MeteredExtractor(fail=True), MeteredJev(), on_usage=events.append))
    assert records[0]["status"] == "api_error"
    costs = audit.build_cost_comparison(events, prices())
    assert costs["llm_only_usd"] == 0 and costs["audit_total_usd"] == 0
    assert costs["is_lower_bound"] and costs["phases"]["classification"]["is_lower_bound"]
    assert costs["phases"]["classification"]["usage"]["requests"] == 1


def test_seed_reuse_keeps_actual_spending_but_does_not_invent_baseline_cost():
    events = []
    from src.pipeline import DocumentTypeDecision
    decision = DocumentTypeDecision(document_type="transfer_draft", confidence=1)
    records = asyncio.run(audit.audit_cases([case()], MeteredExtractor(), MeteredJev(),
                                           initial_seeds={"C1": (decision, proposal(100))}, on_usage=events.append))
    assert not any(event["phase"] == "classification" for event in events)
    costs = audit.build_cost_comparison(events, prices(), reused_baseline=True)
    assert costs["llm_only_usd"] is None and costs["full_usd"] is None
    assert costs["audit_total_usd"] == pytest.approx(.003084)
    previous = json.loads(json.dumps(records[0]))
    new_events = []
    asyncio.run(audit.audit_cases([case()], MeteredExtractor(), MeteredJev(),
                                 resume_records={"C1": previous}, on_usage=new_events.append))
    assert new_events == []
    resumed = audit.build_cost_comparison(new_events, prices(), reused_baseline=True)
    assert resumed["audit_total_usd"] == 0
    assert resumed["full_usd"] is None


@pytest.mark.parametrize("missing", [True, False])
def test_missing_usage_or_unpriced_verifier_does_not_look_free(missing):
    events = []
    verifier = Jev() if missing else MeteredJev()
    asyncio.run(audit.audit_cases([case()], MeteredExtractor(), verifier, on_usage=events.append))
    costs = audit.build_cost_comparison(events, prices() if missing else None)
    assert costs["llm_only_usd"] == pytest.approx(.003)
    assert costs["phases"]["verification"]["cost_usd"] is None
    assert costs["full_usd"] is None and costs["audit_total_usd"] is None


def test_real_sdk_response_with_missing_usage_is_still_a_valid_verdict():
    from typesafe_sdk import SystemOneResponse

    class SDK:
        async def system_one(self, **kwargs):
            return SystemOneResponse.model_validate({"model": "fixture", "usage": {}, "answers": {
                "check": {"type": "noul", "noul": .01},
            }})

    class Data:
        def model_dump(self, **kwargs):
            return {}

    verifier = audit.AuditDecisionModel(SDK())
    result = asyncio.run(verifier.ainvoke(Data(), Data(), VerificationResponse))
    assert result.answers["check"].noul == .01
    assert verifier.usage.sdk_calls == verifier.usage.responses == 1
    assert verifier.usage.usage_reports == 0


def test_a_budget_blocked_before_http_does_not_add_a_phantom_cost():
    from src.schemas.extraction_schemas import DocumentClassification

    settings = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture",
                           LLM_API_PROTOCOL="openai", LLM_RESPONSE_FORMAT="text",
                           LLM_INPUT_PRICE_PER_MILLION=.15, LLM_OUTPUT_PRICE_PER_MILLION=1.875)
    extractor = audit.BudgetedExtractor(settings, audit.APICallBudget(1))
    extractor.budget.spend()
    events = []

    async def run():
        async with extractor:
            return await audit.audit_cases([case()], extractor, MeteredJev(), on_usage=events.append)

    records = asyncio.run(run())
    assert records[0]["status"] == "budget_exhausted"
    assert events[0]["usage"]["requests"] == 0
    assert audit.build_cost_comparison(events, prices())["audit_total_usd"] == 0


def test_verifier_retries_are_counted_even_when_a_later_response_succeeds(monkeypatch):
    from typesafe_sdk import TypeSafeAPIConnectionError, SystemOneResponse

    class SDK:
        count = 0

        async def system_one(self, **kwargs):
            self.count += 1
            if self.count == 1:
                raise TypeSafeAPIConnectionError("fixture")
            return SystemOneResponse.model_validate({"model": "fixture", "usage": {"input_tokens": 1000, "output_tokens": 10},
                "answers": {key: {"type": "noul", "noul": .01} for key in kwargs["questions"]}})

    async def sleep(*args):
        pass

    monkeypatch.setattr(audit.asyncio, "sleep", sleep)
    events = []
    records = asyncio.run(audit.audit_cases([case()], MeteredExtractor(), audit.AuditDecisionModel(SDK()),
                                           on_usage=events.append, api_retries=1))
    assert records[0]["status"] == "complete"
    assert [event["phase"] for event in events].count("verification") == 2
    costs = audit.build_cost_comparison(events, prices())
    assert costs["audit_total_usd"] is not None and costs["is_lower_bound"]
    assert costs["phases"]["verification"]["is_lower_bound"]
    assert costs["phases"]["verification"]["usage"]["sdk_calls"] == 2
    assert costs["phases"]["verification"]["usage"]["input_tokens"] == 1000
    assert costs["phases"]["verification"]["known_subtotal_usd"] == pytest.approx(.000042)


def test_offline_recovery_does_not_turn_unknown_historical_costs_into_zero(tmp_path):
    from tests.test_audit_recovery import inputs

    trace, labels = inputs(tmp_path)
    path = audit.recover_report(trace, labels, tmp_path / "recovered")
    assert json.loads(path.read_text())["cost_comparison"] is None


@pytest.mark.parametrize("repair", [False, True])
def test_live_report_and_console_include_costs_from_mock_http(monkeypatch, tmp_path, capsys, repair):
    inputs, labels = tmp_path / "cases.jsonl", tmp_path / "labels.jsonl"
    inputs.write_text(case().model_dump_json() + "\n")
    gold = label(200 if repair else 100)
    gold.pop("scoring_only")
    labels.write_text(json.dumps(gold) + "\n")
    requests = []

    def llm_http(request):
        requests.append(json.loads(request.content))
        number = len(requests)
        reply = ({"document_type": "transfer_draft", "confidence": 1, "none_reason": None} if number == 1
                 else {"fields": (proposal(200 if number == 3 else 100)).model_dump(mode="json")["fields"]})
        return httpx2.Response(200, json={"usage": {"prompt_tokens": 1000, "completion_tokens": 100},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(reply)}}]})

    def verifier_http(request):
        body = json.loads(request.content)
        return httpx2.Response(200, json={"model": "fixture", "usage": {"input_tokens": 2000, "output_tokens": 100},
            "answers": {key: {"type": "noul", "noul": .99 if repair and key == "amount_rub::superseded"
                and question["instructions"]["field"]["value"] == 100 else .01}
                for key, question in body["questions"].items()}})

    real_extractor, real_sdk = audit.BudgetedExtractor, audit.AsyncTypeSafeClient

    def extractor(settings, budget):
        instance = real_extractor(settings, budget)
        instance._client = httpx2.AsyncClient(transport=httpx2.MockTransport(llm_http))
        return instance

    monkeypatch.setattr(audit, "BudgetedExtractor", extractor)
    monkeypatch.setattr(audit, "AsyncTypeSafeClient", lambda **kwargs: real_sdk(
        **kwargs, transport=httpx2.MockTransport(verifier_http)))
    monkeypatch.setattr(audit, "LLMSettings", lambda: LLMSettings(
        _env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture", LLM_API_PROTOCOL="openai",
        LLM_RESPONSE_FORMAT="text", LLM_INPUT_PRICE_PER_MILLION=.15, LLM_OUTPUT_PRICE_PER_MILLION=1.875))
    monkeypatch.setattr(audit, "DecisionModelSettings", prices)
    monkeypatch.setattr(audit, "PipelineSettings", lambda: PipelineSettings(_env_file=None))
    directory = tmp_path / "audit"
    args = audit.build_parser().parse_args(["--live", "--verifier", "decision", "--input", str(inputs),
                                          "--labels", str(labels), "--output-dir", str(directory)])
    assert asyncio.run(audit.run_live(args))
    report = json.loads((directory / "report.json").read_text())
    costs = report["cost_comparison"]
    assert costs["llm_only_usd"] == pytest.approx(.000675)
    assert costs["full_usd"] == pytest.approx(.0011805 if repair else .000759)
    assert costs["audit_total_usd"] == costs["full_usd"]
    assert len(requests) == (3 if repair else 2)
    assert report["pricing"]["decision_input_per_million_usd"] == .042
    before = report["before"]["strict_metrics"]["acceptance"]
    after = report["after"]["strict_metrics"]["acceptance"]
    assert before["accepted_error_rate"] == float(repair)
    assert after["accepted_error_rate"] == 0
    assert after["accepted_correct"] == 1
    detection = report["error_detection"]
    assert detection["tp"] == int(repair)
    assert detection["tn"] == 5 - int(repair)
    assert detection["fp"] == detection["fn"] == 0
    assert detection["field_coverage"] == 1
    assert report["field_outcomes"]["corrected"] == int(repair)
    assert report["field_outcomes"]["missed"] == report["field_outcomes"]["unknown_final"] == 0
    output = capsys.readouterr().out
    assert "llm-only: $0.000675" in output and "Фактическая оценка расходов аудита:" in output
    assert output.count("Автоматически приняты: 1/1 (100.0%)") == 2
    assert "Правильные автоматически принятые: 1/1 (100.0%)" in output
    assert "Обнаружение ошибок в первой проверке исходного кандидата" in output
    assert f"TP={int(repair)}; FP=0; FN=0; TN={5 - int(repair)}" in output
    assert f"Исправлено: {int(repair)}; обнаружено и отправлено на проверку: 0" in output
