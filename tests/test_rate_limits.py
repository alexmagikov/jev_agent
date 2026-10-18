import asyncio
import json
from types import SimpleNamespace

import httpx2
import pytest

import src.llm as llm
import src.main as runner
import src.pipeline as pipeline
from src.cases.document_type import DocumentMessage
from src.schemas.document_schemas import TransferFields, TransferResult
from src.settings import LLMSettings, PipelineSettings


def settings(**overrides):
    values = {
        "LLM_API_KEY": "fake-test-key", "LLM_MODEL": "small",
        "LLM_BASE_URL": "https://example.com/v1", "LLM_API_PROTOCOL": "anthropic",
        "LLM_MAX_RETRIES": 0, "LLM_RATE_LIMIT_RETRIES": 2,
        "LLM_RATE_LIMIT_BACKOFF_SECONDS": 30,
    }
    values.update(overrides)
    return LLMSettings(_env_file=None, **values)


def text_response(data):
    return httpx2.Response(200, json={
        "content": [{"type": "text", "text": json.dumps(data)}], "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    })


def classification():
    return {"document_type": "transfer_draft", "confidence": 0.9, "none_reason": None}


def proposal():
    return {"fields": {name: {"status": "missing", "value": None, "evidence": [], "calculation": None}
                       for name in TransferFields.model_fields}}


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(now=0.0, sleeps=[])

    async def sleep(delay):
        state.sleeps.append(delay)
        state.now += delay

    monkeypatch.setattr(llm.time, "monotonic", lambda: state.now)
    monkeypatch.setattr(llm.asyncio, "sleep", sleep)
    return state


def test_pacing_is_shared_by_classification_extraction_and_cases(clock):
    starts = []
    replies = [classification(), proposal(), classification()]

    def respond(request):
        starts.append(clock.now)
        return text_response(replies.pop(0))

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with llm.AsyncLLMExtractor(settings(LLM_MIN_REQUEST_INTERVAL_SECONDS=6), client=client) as extractor:
                await extractor.classify([])
                await extractor.extract([], "transfer_draft")
                await extractor.classify([])

    asyncio.run(run())
    assert starts == [0, 6, 12]
    assert clock.sleeps == [6, 6]


def test_429_honors_retry_after_even_when_other_network_retries_are_disabled(clock, capsys):
    requests = []

    def respond(request):
        requests.append(request.content)
        if len(requests) == 1:
            return httpx2.Response(429, headers={"Retry-After": "20"})
        return text_response(classification())

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with llm.AsyncLLMExtractor(settings(), client=client) as extractor:
                result = await extractor.classify([])
                assert result.document_type == "transfer_draft"
                assert extractor.usage.requests == 2
                assert extractor.usage.retries == 1
                assert extractor.usage.rate_limit_errors == 1
                assert extractor.usage.input_tokens == 10

    asyncio.run(run())
    assert len(requests) == 2 and requests[0] == requests[1]
    assert clock.sleeps == [20]
    printed = capsys.readouterr().out
    assert "429" in printed and "20" in printed
    assert "fake-test-key" not in printed


@pytest.mark.parametrize("retry_after", [None, "nonsense", "nan"])
def test_429_without_valid_header_uses_bounded_slow_backoff(clock, retry_after):
    calls = []

    def respond(request):
        calls.append(clock.now)
        if len(calls) <= 2:
            return httpx2.Response(429, headers={"Retry-After": retry_after} if retry_after else {})
        return text_response(classification())

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with llm.AsyncLLMExtractor(settings(), client=client) as extractor:
                await extractor.classify([])

    asyncio.run(run())
    assert calls == [0, 30, 90]
    assert clock.sleeps == [30, 60]


def test_persistent_429_has_a_distinct_terminal_error_and_finite_budget(clock):
    calls = []

    def respond(request):
        calls.append(True)
        return httpx2.Response(429)

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with llm.AsyncLLMExtractor(settings(), client=client) as extractor:
                with pytest.raises(llm.LLMRateLimitError, match="429"):
                    await extractor.classify([])
                assert extractor.usage.requests == 3
                assert extractor.usage.retries == 2
                assert extractor.usage.rate_limit_errors == 3

    asyncio.run(run())
    assert len(calls) == 3
    assert clock.sleeps == [30, 60]


def test_long_retry_after_is_not_clipped_into_an_early_retry(clock):
    calls = []

    def respond(request):
        calls.append(True)
        return httpx2.Response(429, headers={"Retry-After": "120"})

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with llm.AsyncLLMExtractor(settings(), client=client) as extractor:
                with pytest.raises(llm.LLMRateLimitError, match="Retry-After"):
                    await extractor.classify([])

    asyncio.run(run())
    assert len(calls) == 1 and clock.sleeps == []


def test_429_budget_does_not_enable_other_network_retries(clock):
    calls = []

    def respond(request):
        calls.append(True)
        return httpx2.Response(503)

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with llm.AsyncLLMExtractor(settings(), client=client) as extractor:
                with pytest.raises(llm.LLMError, match="503"):
                    await extractor.classify([])

    asyncio.run(run())
    assert len(calls) == 1 and clock.sleeps == []


@pytest.mark.parametrize("separate_budget", [False, True])
def test_mixed_errors_keep_legacy_shared_budget_unless_429_budget_is_explicit(clock, separate_budget):
    calls = []

    def respond(request):
        calls.append(True)
        if len(calls) == 1:
            return httpx2.Response(429, headers={"Retry-After": "1"})
        if len(calls) == 2:
            return httpx2.Response(503, headers={"Retry-After": "1"})
        return text_response(classification())

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with llm.AsyncLLMExtractor(settings(
                LLM_MAX_RETRIES=1, LLM_RATE_LIMIT_RETRIES=1 if separate_budget else None,
            ), client=client) as extractor:
                if separate_budget:
                    await extractor.classify([])
                else:
                    with pytest.raises(llm.LLMError, match="503"):
                        await extractor.classify([])

    asyncio.run(run())
    assert len(calls) == (3 if separate_budget else 2)


def test_retry_after_and_pacing_are_both_respected_without_double_wait(clock):
    calls = []

    def respond(request):
        calls.append(clock.now)
        if len(calls) == 1:
            return httpx2.Response(429, headers={"Retry-After": "2"})
        return text_response(classification())

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with llm.AsyncLLMExtractor(settings(LLM_MIN_REQUEST_INTERVAL_SECONDS=6), client=client) as extractor:
                await extractor.classify([])

    asyncio.run(run())
    assert calls == [0, 6]
    assert sum(clock.sleeps) == 6


@pytest.mark.parametrize("phase", ["classification", "extraction"])
def test_pipeline_preserves_rate_limit_error_for_batch_stop(clock, phase):
    replies = [] if phase == "classification" else [classification()]

    def respond(request):
        return text_response(replies.pop(0)) if replies else httpx2.Response(429)

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with llm.AsyncLLMExtractor(settings(LLM_RATE_LIMIT_RETRIES=0), client=client) as extractor:
                with pytest.raises(pipeline.PipelineRateLimitError, match="429"):
                    await pipeline.process_document(
                        "C1", [DocumentMessage(id="m1", role="user", text="Переведи себе")],
                        None, extractor, use_jev=False,
                    )

    asyncio.run(run())


def result(case_id):
    return TransferResult(
        case_id=case_id, document_type="transfer_draft", confidence=0.9,
        fields={name: {"status": "missing", "value": None, "evidence": [], "derivation": None}
                for name in TransferFields.model_fields},
    )


@pytest.mark.parametrize("mode", ["full", "llm-only"])
def test_batch_stops_on_rate_limit_and_prints_partial_summary(monkeypatch, tmp_path, capsys, mode):
    path, labels, output = (tmp_path / name for name in ("cases.jsonl", "labels.jsonl", "out.jsonl"))
    path.write_text("\n".join(json.dumps({"case_id": case_id, "split": "dev", "messages": [
        {"id": "m1", "role": "user", "text": "Переведи себе"},
    ]}) for case_id in ("C1", "C2", "C3")))
    labels.write_text("\n".join(result(case_id).model_dump_json() for case_id in ("C1", "C2", "C3")))
    calls = []

    async def process(case_id, *args, **kwargs):
        calls.append(case_id)
        if case_id == "C2":
            raise pipeline.PipelineRateLimitError("C2: HTTP 429")
        return result(case_id)

    class Extractor:
        usage = llm.LLMUsage(requests=4, rate_limit_errors=3)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(runner, "process_document", process)
    monkeypatch.setattr(runner, "JevClientSettings", lambda: SimpleNamespace(api_key=SimpleNamespace(get_secret_value=lambda: "fake")))
    monkeypatch.setattr(runner, "AsyncJev", lambda **kwargs: object())
    monkeypatch.setattr(runner, "LLMSettings", lambda: object())
    monkeypatch.setattr(runner, "PipelineSettings", lambda: PipelineSettings(_env_file=None))
    monkeypatch.setattr(runner, "AsyncLLMExtractor", lambda *args, **kwargs: Extractor())
    with pytest.raises(pipeline.PipelineRateLimitError):
        asyncio.run(runner.main(path, output, labels, mode=mode))
    assert calls == ["C1", "C2"]
    assert json.loads(output.read_text())["case_id"] == "C1"
    errors = [json.loads(line) for line in (tmp_path / "out.errors.jsonl").read_text().splitlines()]
    assert [row["case_id"] for row in errors] == ["C2"]
    printed = capsys.readouterr().out
    assert "Успешно: 1; с ошибкой: 1" in printed
    assert "Не обработано: 1" in printed
    assert "1/2 (50.0%)" in printed
    assert "HTTP 429" in printed
