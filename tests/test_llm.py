import asyncio
import json

import httpx2
import pytest

from src.cases.document_type import DocumentMessage
from src.schemas.document_schemas import TransferFields
from src.llm import AsyncLLMExtractor, LLMError
from src.settings import LLMSettings


def settings(**overrides):
    return LLMSettings(
        _env_file=None, LLM_API_KEY="test", LLM_MODEL="small",
        LLM_BASE_URL="http://localhost:11434/v1", **overrides,
    )


def test_prompt_explicitly_states_inn_length_and_invalid_handling():
    from src.llm import EXTRACTION_PROMPT

    assert "ИНН должен содержать ровно 10 или 12 цифр" in EXTRACTION_PROMPT
    assert "Не дополняй и не исправляй его самостоятельно" in EXTRACTION_PROMPT
    assert "При другой длине установи status='invalid', value=null" in EXTRACTION_PROMPT
    assert "и сохрани исходную цитату в evidence" in EXTRACTION_PROMPT


def dialogue():
    return [DocumentMessage(id="m1", role="user", text="Переведи себе")]


def envelope(content=None, usage=None):
    proposal = {
        "fields": {name: {"status": "missing", "value": None, "evidence": []}
                   for name in TransferFields.model_fields},
    }
    return {
        "choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(proposal) if content is None else content,
        }}],
        "usage": usage,
    }


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_debug_shows_model_text_before_validation_only_when_enabled(debug, finish_reason, capsys):
    from src.llm import LLMResponseError

    response = envelope("not valid JSON")
    response["choices"][0]["finish_reason"] = finish_reason

    async def run():
        transport = httpx2.MockTransport(lambda request: httpx2.Response(200, json=response))
        async with httpx2.AsyncClient(transport=transport) as client:
            async with AsyncLLMExtractor(settings(), client=client, debug=debug) as extractor:
                with pytest.raises(LLMResponseError):
                    await extractor.extract(dialogue(), "transfer_draft")

    asyncio.run(run())
    printed = capsys.readouterr().out
    assert ("not valid JSON" in printed) is debug
    assert ("finish_reason=" in printed) is debug
    assert "Authorization" not in printed
    assert "Bearer" not in printed


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504, 529])
def test_temporary_http_error_retries_with_delay_and_tracks_usage(monkeypatch, status):
    requests, pauses = [], []

    async def sleep(delay):
        pauses.append(delay)

    monkeypatch.setattr("src.llm.asyncio.sleep", sleep)

    def respond(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx2.Response(status, headers={"Retry-After": "2"})
        return httpx2.Response(200, json=envelope(usage={"prompt_tokens": 100, "completion_tokens": 20}))

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings(
                LLM_INPUT_PRICE_PER_MILLION=1, LLM_OUTPUT_PRICE_PER_MILLION=2,
            ), client=client) as extractor:
                result = await extractor.extract(dialogue(), "transfer_draft")
                assert set(result.fields) == set(TransferFields.model_fields)
                assert extractor.usage.requests == 2
                assert extractor.usage.retries == 1
                assert extractor.usage.input_tokens == 100
                assert extractor.usage.output_tokens == 20
                assert extractor.usage.estimated_cost == pytest.approx(0.00014)
                assert extractor.usage.elapsed_seconds >= 0

    asyncio.run(run())
    assert requests[0] == requests[1]
    assert pauses == [2]


@pytest.mark.parametrize("status", [401, 403, 400, 302])
def test_permanent_http_errors_are_not_retried(status):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx2.Response(status, headers={"Location": "https://example.com"})

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings(), client=client) as extractor:
                with pytest.raises(LLMError, match=str(status)):
                    await extractor.extract(dialogue(), "transfer_draft")

    asyncio.run(run())
    assert len(calls) == 1


def test_transport_retries_are_bounded(monkeypatch):
    calls = []

    async def sleep(delay):
        pass

    monkeypatch.setattr("src.llm.asyncio.sleep", sleep)

    def respond(request):
        calls.append(request)
        raise httpx2.ReadTimeout("test", request=request)

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings(LLM_MAX_RETRIES=1), client=client) as extractor:
                with pytest.raises(LLMError):
                    await extractor.extract(dialogue(), "transfer_draft")
                assert extractor.usage.requests == 2
                assert extractor.usage.retries == 1

    asyncio.run(run())
    assert len(calls) == 2


def test_malformed_model_output_still_counts_tokens():
    from src.llm import LLMResponseError

    async def run():
        transport = httpx2.MockTransport(lambda request: httpx2.Response(
            200, json=envelope("not json", {"prompt_tokens": 50, "completion_tokens": 5}),
        ))
        async with httpx2.AsyncClient(transport=transport) as client:
            async with AsyncLLMExtractor(settings(), client=client) as extractor:
                with pytest.raises(LLMResponseError):
                    await extractor.extract(dialogue(), "transfer_draft")
                assert extractor.usage.input_tokens == 50
                assert extractor.usage.output_tokens == 5
                assert extractor.usage.estimated_cost is None

    asyncio.run(run())


def test_missing_usage_does_not_claim_zero_cost():
    async def run():
        transport = httpx2.MockTransport(lambda request: httpx2.Response(200, json=envelope()))
        async with httpx2.AsyncClient(transport=transport) as client:
            async with AsyncLLMExtractor(settings(
                LLM_INPUT_PRICE_PER_MILLION=1, LLM_OUTPUT_PRICE_PER_MILLION=2,
            ), client=client) as extractor:
                await extractor.extract(dialogue(), "transfer_draft")
                assert extractor.usage.usage_reports == 0
                assert extractor.usage.estimated_cost is None

    asyncio.run(run())


def test_owned_http_client_is_closed_even_after_failure():
    async def run():
        extractor = AsyncLLMExtractor(settings())
        with pytest.raises(RuntimeError):
            async with extractor:
                client = extractor._client
                raise RuntimeError("test")
        assert client.is_closed

    asyncio.run(run())


def test_response_size_is_bounded():
    async def run():
        transport = httpx2.MockTransport(lambda request: httpx2.Response(200, content=b"x" * 2_000_001))
        async with httpx2.AsyncClient(transport=transport) as client:
            async with AsyncLLMExtractor(settings(), client=client) as extractor:
                with pytest.raises(LLMError, match="Слишком большой"):
                    await extractor.extract(dialogue(), "transfer_draft")

    asyncio.run(run())


def test_invalid_api_envelope_does_not_claim_zero_cost():
    async def run():
        transport = httpx2.MockTransport(lambda request: httpx2.Response(200, content=b"not json"))
        async with httpx2.AsyncClient(transport=transport) as client:
            async with AsyncLLMExtractor(settings(
                LLM_INPUT_PRICE_PER_MILLION=1, LLM_OUTPUT_PRICE_PER_MILLION=2,
            ), client=client) as extractor:
                with pytest.raises(LLMError, match="формат ответа"):
                    await extractor.extract(dialogue(), "transfer_draft")
                assert extractor.usage.estimated_cost is None

    asyncio.run(run())


def test_retry_after_too_long_stops_instead_of_retrying_early():
    calls = []

    def respond(request):
        calls.append(request)
        return httpx2.Response(429, headers={"Retry-After": "120"})

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings(), client=client) as extractor:
                with pytest.raises(LLMError, match="Retry-After"):
                    await extractor.extract(dialogue(), "transfer_draft")

    asyncio.run(run())
    assert len(calls) == 1


def test_retry_after_http_date_and_backoff_cap(monkeypatch):
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime
    from src.llm import _retry_delay

    header = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=10), usegmt=True)
    assert 8 <= _retry_delay(0, header) <= 10
    monkeypatch.setattr("src.llm.random.uniform", lambda *args: 1.5)
    assert _retry_delay(20, None) == 60


def test_timeout_bounds_whole_attempt():
    async def respond(request):
        await asyncio.sleep(1)
        return httpx2.Response(200, json=envelope())

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings(
                LLM_TIMEOUT_SECONDS=0.01, LLM_MAX_RETRIES=0,
            ), client=client) as extractor:
                with pytest.raises(LLMError, match="timeout"):
                    await extractor.extract(dialogue(), "transfer_draft")
                assert extractor.usage.requests == 1

    asyncio.run(run())
