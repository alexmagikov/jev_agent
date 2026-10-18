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
        "response_model": Answer,
    }
    assert sdk_client.exit_args is not None
    if fails:
        assert sdk_client.exit_args[0] is RuntimeError
        assert sdk_client.exit_args[1] is error
    else:
        assert sdk_client.exit_args == (None, None, None)
