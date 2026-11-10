"""Стоимость по сообщённому usage; без запросов к платным API."""

import asyncio
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import src.main as runner
from src.client import DecisionUsage
from src.llm import LLMUsage
from src.run_artifacts import load_run, public_settings
from src.schemas.document_schemas import TransferFields, TransferResult
from src.settings import DecisionModelSettings, LLMSettings, PipelineSettings


def rates(**changes):
    return DecisionModelSettings(_env_file=None, DECISION_API_KEY="fixture",
                                 DECISION_INPUT_PRICE_PER_MILLION=.042,
                                 DECISION_OUTPUT_PRICE_PER_MILLION=0, **changes)


def llm_usage(**changes):
    return LLMUsage(**{**dict(requests=1, responses=1, usage_reports=1,
                             input_tokens=1_000_000, output_tokens=1_000_000,
                             estimated_cost=2.025), **changes})


def decision_usage(**changes):
    return DecisionUsage(**{**dict(sdk_calls=2, responses=2, usage_reports=2,
                                  input_tokens=1_000_000, output_tokens=50_000), **changes})


def test_prices_are_loaded_from_env_and_saved_without_credentials(monkeypatch):
    monkeypatch.setenv("DECISION_INPUT_PRICE_PER_MILLION", "0.042")
    monkeypatch.setenv("DECISION_OUTPUT_PRICE_PER_MILLION", "0")
    config = DecisionModelSettings(_env_file=None, DECISION_API_KEY="fixture")
    assert config.input_price_per_million == .042
    assert config.output_price_per_million == 0
    safe = public_settings(config)
    assert safe["input_price_per_million"] == .042
    assert safe["output_price_per_million"] == 0
    assert "api_key" not in safe


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
@pytest.mark.parametrize("name", ["DECISION_INPUT_PRICE_PER_MILLION", "DECISION_OUTPUT_PRICE_PER_MILLION"])
def test_invalid_rates_are_rejected(name, value):
    with pytest.raises(ValidationError):
        DecisionModelSettings(_env_file=None, DECISION_API_KEY="fixture", **{name: value})


def test_full_includes_jev_and_llm_only_does_not_mutate_usage():
    llm, decision = llm_usage(), decision_usage()
    before = asdict(llm), asdict(decision)
    full = runner.estimate_usage_costs(llm, decision, decision_settings=rates(), include_decision=True)
    baseline = runner.estimate_usage_costs(llm, None, include_decision=False)
    money = ("llm_usd", "decision_usd", "total_usd")
    assert {key: full[key] for key in money} == pytest.approx({"llm_usd": 2.025, "decision_usd": .042, "total_usd": 2.067})
    assert {key: baseline[key] for key in money} == pytest.approx({"llm_usd": 2.025, "decision_usd": 0, "total_usd": 2.025})
    assert full["is_lower_bound"] is baseline["is_lower_bound"] is False
    assert (asdict(llm), asdict(decision)) == before


# Вызов без ответа даёт нижнюю границу (отдельный тест); здесь ответ без usage.
@pytest.mark.parametrize("changes", [dict(usage_reports=1)])
def test_incomplete_decision_usage_does_not_look_like_full_cost(changes):
    costs = runner.estimate_usage_costs(llm_usage(), decision_usage(**changes),
                                       decision_settings=rates(), include_decision=True)
    assert costs["llm_usd"] == 2.025
    assert costs["decision_usd"] is None and costs["total_usd"] is None


@pytest.mark.parametrize("changes", [dict(estimated_cost=None), dict(usage_reports=0)])
def test_unknown_llm_cost_preserves_decision_cost(changes):
    costs = runner.estimate_usage_costs(llm_usage(**changes), decision_usage(),
                                       decision_settings=rates(), include_decision=True)
    assert costs["llm_usd"] is None and costs["total_usd"] is None
    assert costs["decision_usd"] == pytest.approx(.042)


def test_unused_models_cost_zero_but_an_unpriced_used_verifier_is_unknown():
    assert runner.estimate_usage_costs(LLMUsage(), DecisionUsage(), include_decision=True) == {
        "llm_usd": 0, "decision_usd": 0, "total_usd": 0,
        "is_lower_bound": False, "unanswered_requests": {"llm": 0, "decision": 0},
    }
    costs = runner.estimate_usage_costs(llm_usage(), decision_usage(), include_decision=True)
    assert costs["decision_usd"] is None and costs["total_usd"] is None


@pytest.mark.parametrize("mode,expected", [("llm-only", 2.025), ("full", 2.067)])
def test_main_prints_and_saves_mode_costs_without_network(monkeypatch, tmp_path, capsys, mode, expected):
    case = {"case_id": "price", "split": "dev", "messages": [{"id": "m1", "role": "user", "text": "Переведи себе"}]}
    result = TransferResult(case_id="price", document_type="transfer_draft", confidence=1,
                            fields={name: {"status": "missing", "value": None, "evidence": [], "derivation": None}
                                    for name in TransferFields.model_fields})
    inputs, labels = tmp_path / "cases.jsonl", tmp_path / "labels.jsonl"
    inputs.write_text(json.dumps(case) + "\n")
    labels.write_text(result.model_dump_json() + "\n")
    llm_settings = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="qwen-fixture",
                               LLM_INPUT_PRICE_PER_MILLION=.15, LLM_OUTPUT_PRICE_PER_MILLION=1.875)

    class Extractor:
        usage = llm_usage()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    async def process(*args, **kwargs):
        return result

    monkeypatch.setattr(runner, "LLMSettings", lambda: llm_settings)
    monkeypatch.setattr(runner, "DecisionModelSettings", rates)
    monkeypatch.setattr(runner, "PipelineSettings", lambda: PipelineSettings(_env_file=None))
    monkeypatch.setattr(runner, "DecisionModelClient", lambda **kwargs: SimpleNamespace(usage=decision_usage()))
    monkeypatch.setattr(runner, "AsyncLLMExtractor", lambda *args, **kwargs: Extractor())
    monkeypatch.setattr(runner, "process_document", process)
    output = asyncio.run(runner.main(inputs, tmp_path / "result.jsonl", labels, mode=mode, verifier="decision"))
    assert f"Оценка общей стоимости запуска: ${expected:.6f}" in capsys.readouterr().out
    bundle = load_run(output.parent)
    assert bundle["run"]["usage"]["cost_estimate"]["total_usd"] == pytest.approx(expected)
    assert bundle["config"]["llm"]["input_price_per_million"] == .15
    if mode == "full":
        assert bundle["config"]["decision"]["input_price_per_million"] == .042


def test_unanswered_llm_request_keeps_answered_cost_as_a_lower_bound():
    costs = runner.estimate_usage_costs(llm_usage(requests=2), decision_usage(),
                                       decision_settings=rates(), include_decision=True)
    assert costs["llm_usd"] == pytest.approx(2.025)
    assert costs["total_usd"] == pytest.approx(2.067)
    assert costs["is_lower_bound"] is True
    assert costs["unanswered_requests"] == {"llm": 1, "decision": 0}


def test_rate_limited_requests_are_not_billed_and_keep_the_cost_exact():
    costs = runner.estimate_usage_costs(llm_usage(requests=3, rate_limit_errors=2), None)
    assert costs["llm_usd"] == pytest.approx(2.025)
    assert costs["is_lower_bound"] is False
    assert costs["unanswered_requests"] == {"llm": 0, "decision": 0}


def test_unanswered_decision_call_keeps_answered_cost_as_a_lower_bound():
    costs = runner.estimate_usage_costs(llm_usage(), decision_usage(sdk_calls=3),
                                       decision_settings=rates(), include_decision=True)
    assert costs["decision_usd"] == pytest.approx(.042)
    assert costs["total_usd"] == pytest.approx(2.067)
    assert costs["is_lower_bound"] is True
    assert costs["unanswered_requests"] == {"llm": 0, "decision": 1}


def test_failed_first_llm_call_does_not_hide_later_reported_costs():
    import httpx2
    from src.llm import AsyncLLMExtractor, LLMError

    replies = iter([
        httpx2.Response(400, json={}),
        httpx2.Response(200, json={
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(
                {"document_type": "none", "confidence": .9, "none_reason": "no_request"})}}],
            "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0},
        }),
    ])
    settings = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture",
                           LLM_BASE_URL="https://example.com/v1", LLM_RESPONSE_FORMAT="json_object",
                           LLM_INPUT_PRICE_PER_MILLION=.15, LLM_OUTPUT_PRICE_PER_MILLION=1.875)

    async def run():
        transport = httpx2.MockTransport(lambda request: next(replies))
        async with httpx2.AsyncClient(transport=transport) as client:
            extractor = AsyncLLMExtractor(settings, client=client)
            message = [runner.DocumentMessage(id="m1", role="user", text="Привет")]
            with pytest.raises(LLMError):
                await extractor.classify(message)
            await extractor.classify(message)
            return extractor.usage

    usage = asyncio.run(run())
    assert usage.estimated_cost == pytest.approx(.15)


def _llm_usage_after(status):
    import httpx2
    from src.llm import AsyncLLMExtractor, LLMError

    settings = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture",
                           LLM_BASE_URL="https://example.com/v1", LLM_RESPONSE_FORMAT="json_object",
                           LLM_MAX_RETRIES=0, LLM_INPUT_PRICE_PER_MILLION=.15, LLM_OUTPUT_PRICE_PER_MILLION=1.875)

    async def run():
        transport = httpx2.MockTransport(lambda request: httpx2.Response(status, json={}))
        async with httpx2.AsyncClient(transport=transport) as client:
            extractor = AsyncLLMExtractor(settings, client=client)
            with pytest.raises(LLMError):
                await extractor.classify([runner.DocumentMessage(id="m1", role="user", text="Привет")])
            return extractor.usage

    return asyncio.run(run())


@pytest.mark.parametrize("status", [401, 503])
def test_rejected_llm_request_is_a_server_answer_not_an_unanswered_request(status):
    usage = _llm_usage_after(status)
    assert usage.requests == 1 and usage.responses == 0 and usage.error_responses == 1
    costs = runner.estimate_usage_costs(usage, None)
    assert costs["llm_usd"] == 0 and costs["is_lower_bound"] is False


def _decision_usage_after(error):
    import src.client as client_module
    from src.client import DecisionModelClient, DecisionModelError

    class SDK:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def system_one(self, **kwargs):
            raise error

    client = DecisionModelClient(api_key="fixture")
    original = client_module.AsyncTypeSafeClient
    client_module.AsyncTypeSafeClient = SDK
    try:
        with pytest.raises(DecisionModelError):
            asyncio.run(client.ainvoke(SimpleNamespace(model_dump=lambda **kwargs: {}),
                                       SimpleNamespace(model_dump=lambda **kwargs: {}), dict))
    finally:
        client_module.AsyncTypeSafeClient = original
    return client.usage


def test_rate_limited_decision_call_is_not_an_unanswered_request():
    import httpx2
    from typesafe_sdk import TypeSafeRateLimitError

    usage = _decision_usage_after(TypeSafeRateLimitError(429, None, httpx2.Headers()))
    assert usage.sdk_calls == 1 and usage.responses == 0 and usage.error_responses == 1
    costs = runner.estimate_usage_costs(None, usage, decision_settings=rates(), include_decision=True)
    assert costs["decision_usd"] == 0 and costs["is_lower_bound"] is False


def test_lost_decision_connection_is_still_an_unanswered_request():
    from typesafe_sdk import TypeSafeAPIConnectionError

    usage = _decision_usage_after(TypeSafeAPIConnectionError("fixture"))
    assert usage.error_responses == 0
    costs = runner.estimate_usage_costs(None, usage, decision_settings=rates(), include_decision=True)
    assert costs["is_lower_bound"] is True and costs["unanswered_requests"]["decision"] == 1


def test_unknown_total_is_not_printed_as_a_lower_bound(capsys):
    from scripts import audit_jev

    audit_jev.print_cost_comparison({
        "llm_only_usd": None, "full_usd": None, "extra_full_usd": None, "audit_total_usd": None,
        "full_to_llm_only_ratio": None, "reused_baseline": False, "is_lower_bound": True,
        "known_subtotal_usd": .5,
    })
    output = capsys.readouterr().out
    assert "Это нижняя граница" not in output
    assert "$0.500000" in output and "без ответа" in output
