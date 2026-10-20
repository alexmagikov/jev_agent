import asyncio
from decimal import Decimal

import pytest
from pydantic import BaseModel, Field

from src.client import AsyncJev


class State(BaseModel):
    request_id: str = Field(alias="requestId")
    amount: Decimal


class Questions(BaseModel):
    should_check: str = Field(alias="shouldCheck")


class Answer(BaseModel):
    accepted: bool


class FakeTypeSafeClient:
    def __init__(self, *, api_key):
        self.api_key = api_key
        self.entered = False
        self.exit_args = None
        self.calls = []
        self.result = Answer(accepted=True)
        self.error = None

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self.exit_args = (exc_type, exc, traceback)
        return False

    async def system_one(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


@pytest.mark.parametrize("fails", [False, True])
def test_async_jev_serializes_calls_returns_response_and_closes_sdk(monkeypatch, fails):
    import src.client as client_module

    clients = []

    def make_client(*, api_key):
        sdk_client = FakeTypeSafeClient(api_key=api_key)
        clients.append(sdk_client)
        return sdk_client

    monkeypatch.setattr(client_module, "AsyncTypeSafeClient", make_client)
    jev = AsyncJev(api_key="test-key")
    state = State(requestId="case-7", amount=Decimal("12.50"))
    questions = Questions(shouldCheck="Is this supported?")
    expected = Answer(accepted=True)

    if fails:
        error = RuntimeError("SDK request failed")
        # The fake SDK is configured when the context manager is constructed.
        original_factory = make_client

        def failing_client(*, api_key):
            sdk_client = original_factory(api_key=api_key)
            sdk_client.error = error
            return sdk_client

        monkeypatch.setattr(client_module, "AsyncTypeSafeClient", failing_client)
        with pytest.raises(RuntimeError) as raised:
            asyncio.run(jev.ainvoke(state, questions, Answer))
        assert raised.value is error
    else:
        result = asyncio.run(jev.ainvoke(state, questions, Answer))
        assert result == expected

    sdk_client = clients[-1]
    assert sdk_client.api_key == "test-key"
    assert sdk_client.entered
    assert len(sdk_client.calls) == 1
    assert sdk_client.calls[0] == {
        "state": {"requestId": "case-7", "amount": "12.50"},
        "questions": {"shouldCheck": "Is this supported?"},
    }
    assert sdk_client.exit_args is not None
    if fails:
        assert sdk_client.exit_args[0] is RuntimeError
        assert sdk_client.exit_args[1] is error
    else:
        assert sdk_client.exit_args == (None, None, None)


@pytest.mark.parametrize("metadata", [True, False])
def test_jev_usage_is_retained_while_returning_the_callers_typed_model(monkeypatch, metadata):
    import src.client as module
    from typesafe_sdk import SystemOneResponse
    from src.verification import VerificationResponse

    class SDK(FakeTypeSafeClient):
        async def system_one(self, **kwargs):
            self.calls.append(kwargs)
            return SystemOneResponse.model_validate({
                "model": "test-jev", "usage": {"input_tokens": 12, "output_tokens": 3} if metadata else {},
                "answers": {"supported": {"type": "noul", "noul": 0.01}},
            })

    clients = []
    def create(**kwargs):
        sdk = SDK(**kwargs)
        clients.append(sdk)
        return sdk
    monkeypatch.setattr(module, "AsyncTypeSafeClient", create)
    client = AsyncJev("fake")
    async def run():
        for _ in range(2):
            result = await client.ainvoke(State(requestId="id", amount=1), Questions(shouldCheck="check"), VerificationResponse)
            assert result.answers["supported"].noul == .01
    asyncio.run(run())
    assert client.usage.sdk_calls == client.usage.responses == 2
    assert client.usage.usage_reports == (2 if metadata else 0)
    assert client.usage.input_tokens == (24 if metadata else 0)
    assert client.usage.output_tokens == (6 if metadata else 0)
    assert client.usage.elapsed_seconds >= 0
    assert len(clients) == 2 and all(len(sdk.calls) == 1 for sdk in clients)


def test_jev_failed_request_is_counted_without_inventing_token_usage(monkeypatch):
    import src.client as module
    from typesafe_sdk import TypeSafeAPITimeoutError
    from src.client import JevError

    def create(**kwargs):
        sdk = FakeTypeSafeClient(**kwargs)
        sdk.error = TypeSafeAPITimeoutError(30)
        return sdk
    monkeypatch.setattr(module, "AsyncTypeSafeClient", create)
    client = AsyncJev("fake")
    with pytest.raises(JevError):
        asyncio.run(client.ainvoke(State(requestId="id", amount=1), Questions(shouldCheck="check"), Answer))
    assert client.usage.sdk_calls == 1 and client.usage.responses == 0
    assert client.usage.usage_reports == 0


def test_real_sdk_response_preserves_classification_and_usage_over_mock_http(monkeypatch):
    import httpx2
    import src.client as module
    from src.cases.document_type import DocumentTypeCase, DocumentTypeState, DocumentMessage

    calls = []
    def respond(request):
        calls.append(request)
        return httpx2.Response(200, json={"model": "test-jev", "usage": {"input_tokens": 17, "output_tokens": 5},
            "answers": {
                "document_type": {"type": "choice", "choice": "transfer_draft", "confidence": .9,
                                  "probabilities": {"transfer_draft": .9, "invoice_draft": .05, "none": .05}},
                "none_reason": {"type": "choice", "choice": "no_request", "confidence": .9,
                                "probabilities": {"no_request": .9, "cancelled": .05, "other_document": .05}},
            }})
    real_sdk = module.AsyncTypeSafeClient
    monkeypatch.setattr(module, "AsyncTypeSafeClient", lambda **kwargs: real_sdk(
        **kwargs, transport=httpx2.MockTransport(respond),
    ))
    client = AsyncJev("fake")
    case = DocumentTypeCase(state=DocumentTypeState(messages=[
        DocumentMessage(id="m1", role="user", text="Переведи себе"),
    ]))
    result = asyncio.run(case.run(client))
    assert result.answers.document_type.choice == "transfer_draft"
    assert result.answers.document_type.confidence == .9
    assert len(calls) == 1
    assert client.usage.usage_reports == 1
    assert (client.usage.input_tokens, client.usage.output_tokens) == (17, 5)
