import asyncio
import json

import httpx2
import pytest
from pydantic import ValidationError

from src.cases.document_type import DocumentMessage
from src.extraction import get_proposal_schema
from src.llm import AsyncLLMExtractor, EXTRACTION_PROMPT, LLMError, LLMResponseError
from src.schemas.document_schemas import InvoiceFields, TransferFields
from src.settings import LLMSettings


def settings(**overrides):
    values = {
        "LLM_API_KEY": "test-gateway-key",
        "LLM_API_PROTOCOL": "anthropic",
        "LLM_BASE_URL": "https://claude.gigaparsers.ru/v1",
        "LLM_MODEL": "deepseek-v4.1-flash",
        "LLM_RESPONSE_FORMAT": "text",
        "LLM_OPENCODE_SESSION": "test",
        "LLM_MAX_RETRIES": 0,
    }
    values.update(overrides)
    return LLMSettings(_env_file=None, **values)


def envelope(document_type="transfer_draft"):
    model = InvoiceFields if document_type == "invoice_draft" else TransferFields
    proposal = {"fields": {
        name: {"status": "missing", "value": None, "evidence": [], "calculation": None}
        for name in model.model_fields
    }}
    text = json.dumps(proposal)
    midpoint = len(text) // 2
    return {
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "internal reasoning"},
            {"type": "text", "text": text[:midpoint]},
            {"type": "text", "text": text[midpoint:]},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


@pytest.mark.parametrize("document_type", ["invoice_draft", "transfer_draft"])
def test_gateway_uses_messages_protocol_and_counts_tokens(document_type):
    requests = []
    messages = [DocumentMessage(id="m1", role="user", text="Подготовь черновик")]
    feedback = ["items::unsupported: проверь источники"]

    def respond(request):
        requests.append(request)
        return httpx2.Response(200, json=envelope(document_type))

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings(
                LLM_INPUT_PRICE_PER_MILLION=1, LLM_OUTPUT_PRICE_PER_MILLION=2,
            ), client=client) as extractor:
                result = await extractor.extract(messages, document_type, feedback=feedback)
                model = InvoiceFields if document_type == "invoice_draft" else TransferFields
                assert set(result.fields) == set(model.model_fields)
                assert extractor.usage.input_tokens == 100
                assert extractor.usage.output_tokens == 20
                assert extractor.usage.usage_reports == 1
                assert extractor.usage.estimated_cost == pytest.approx(0.00014)

    asyncio.run(run())
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://claude.gigaparsers.ru/v1/messages"
    assert request.headers["Authorization"] == "Bearer test-gateway-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert request.headers["x-opencode-session"] == "test"
    body = json.loads(request.content)
    assert body["model"] == "deepseek-v4.1-flash"
    assert body["system"] == EXTRACTION_PROMPT
    assert body["max_tokens"] == 4096
    assert body["stream"] is False
    assert "max_completion_tokens" not in body
    assert "response_format" not in body
    assert len(body["messages"]) == 1
    assert body["messages"][0]["role"] == "user"
    user = json.loads(body["messages"][0]["content"])
    assert user["messages"] == [message.model_dump() for message in messages]
    assert user["feedback"] == feedback
    assert user["output_schema"] == get_proposal_schema(document_type)


@pytest.mark.parametrize("reason,error_type", [
    ("max_tokens", LLMResponseError), ("refusal", LLMError), ("tool_use", LLMError),
])
def test_gateway_rejects_incomplete_answers_without_hiding_usage(reason, error_type):
    data = envelope()
    data["stop_reason"] = reason
    calls = []

    def respond(request):
        calls.append(request)
        return httpx2.Response(200, json=data)

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings(), client=client) as extractor:
                with pytest.raises(error_type):
                    await extractor.extract([], "transfer_draft")
                assert extractor.usage.output_tokens == 20

    asyncio.run(run())
    assert len(calls) == 1


@pytest.mark.parametrize("content", [[], "not a block list", [123], [{"type": "text", "text": 123}],
                                     [{"type": "tool_use", "id": "unexpected"}]])
def test_gateway_rejects_missing_or_malformed_text_blocks(content):
    data = envelope()
    data["content"] = content

    async def run():
        transport = httpx2.MockTransport(lambda request: httpx2.Response(200, json=data))
        async with httpx2.AsyncClient(transport=transport) as client:
            async with AsyncLLMExtractor(settings(), client=client) as extractor:
                with pytest.raises(LLMError):
                    await extractor.extract([], "transfer_draft")

    asyncio.run(run())


def test_gateway_counts_cached_tokens_without_inventing_cache_price():
    data = envelope()
    data["usage"].update(cache_creation_input_tokens=30, cache_read_input_tokens=40)

    async def run():
        transport = httpx2.MockTransport(lambda request: httpx2.Response(200, json=data))
        async with httpx2.AsyncClient(transport=transport) as client:
            async with AsyncLLMExtractor(settings(
                LLM_INPUT_PRICE_PER_MILLION=1, LLM_OUTPUT_PRICE_PER_MILLION=2,
            ), client=client) as extractor:
                await extractor.extract([], "transfer_draft")
                assert extractor.usage.input_tokens == 170
                assert extractor.usage.output_tokens == 20
                assert extractor.usage.estimated_cost is None

    asyncio.run(run())


def test_gateway_invalid_json_still_counts_usage():
    data = envelope()
    data["content"] = [{"type": "text", "text": "not json"}]

    async def run():
        transport = httpx2.MockTransport(lambda request: httpx2.Response(200, json=data))
        async with httpx2.AsyncClient(transport=transport) as client:
            async with AsyncLLMExtractor(settings(), client=client) as extractor:
                with pytest.raises(LLMResponseError):
                    await extractor.extract([], "transfer_draft")
                assert extractor.usage.output_tokens == 20

    asyncio.run(run())


@pytest.mark.parametrize("response_format", ["json_object", "json_schema"])
def test_gateway_incompatible_format_is_rejected_before_requests(response_format):
    with pytest.raises(ValidationError):
        settings(LLM_RESPONSE_FORMAT=response_format)


def test_gateway_session_cannot_inject_http_headers():
    with pytest.raises(ValidationError):
        settings(LLM_OPENCODE_SESSION="test\r\nAuthorization: other")


@pytest.mark.parametrize("status", [401, 429, 503])
def test_gateway_zero_retries_surfaces_http_error_without_another_request(status):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx2.Response(status, headers={"Retry-After": "1"})

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings(), client=client) as extractor:
                with pytest.raises(LLMError, match=str(status)):
                    await extractor.extract([], "transfer_draft")
                assert extractor.usage.requests == 1
                assert extractor.usage.retries == 0

    asyncio.run(run())
    assert len(requests) == 1


def test_gateway_without_optional_settings_keeps_local_schema_validation(capsys):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx2.Response(200, json=envelope())

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings(
                LLM_RESPONSE_FORMAT=None, LLM_OPENCODE_SESSION=None,
            ), client=client, debug=True) as extractor:
                await extractor.extract([], "transfer_draft")

    asyncio.run(run())
    assert "x-opencode-session" not in requests[0].headers
    assert "response_format" not in json.loads(requests[0].content)
    printed = capsys.readouterr().out
    assert "internal reasoning" not in printed
    assert "test-gateway-key" not in printed
