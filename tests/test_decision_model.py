"""Подключение универсального клиента: без реального API и секретов."""
import asyncio
import json

import httpx2
import pytest
from pydantic import ValidationError

from src.settings import DecisionModelSettings
from src.client import DecisionModelClient, decision_sdk_options
from src.cases.document_type import DocumentTypeCase, DocumentTypeState, DocumentMessage


def test_polza_configuration_reaches_real_sdk(monkeypatch):
    import src.client as module
    calls = []
    def respond(request):
        calls.append(request)
        return httpx2.Response(200, json={
            "model": "typesafe/jev", "usage": {"input_tokens": 20, "output_tokens": 0},
            "answers": {
                "document_type": {"type": "choice", "choice": "invoice_draft", "confidence": .9,
                                  "probabilities": {"invoice_draft": .9, "transfer_draft": .05, "none": .05}},
                "none_reason": {"type": "choice", "choice": "no_request", "confidence": .9,
                                "probabilities": {"no_request": .9, "cancelled": .05, "other_document": .05}},
            },
        })
    real_sdk = module.AsyncTypeSafeClient
    monkeypatch.setattr(module, "AsyncTypeSafeClient", lambda **kwargs: real_sdk(
        **kwargs, transport=httpx2.MockTransport(respond),
    ))
    settings = DecisionModelSettings(_env_file=None, DECISION_API_KEY="fake-polza",
        DECISION_BASE_URL="https://polza.ai/api", DECISION_MODEL="typesafe/jev")
    options = decision_sdk_options(settings)
    client = DecisionModelClient(api_key=options.pop("api_key"), sdk_options=options)
    case = DocumentTypeCase(state=DocumentTypeState(messages=[
        DocumentMessage(id="m1", role="user", text="Выставь счёт"),
    ]))
    result = asyncio.run(case.run(client))
    assert result.answers.document_type.choice == "invoice_draft"
    assert len(calls) == 1
    assert str(calls[0].url) == "https://polza.ai/api/v1/systemone"
    assert calls[0].headers["Authorization"] == "Bearer fake-polza"
    assert json.loads(calls[0].content)["model"] == "typesafe/jev"
    assert client.usage.input_tokens == 20


def test_old_key_name_still_works_and_new_key_wins():
    settings = DecisionModelSettings(_env_file=None, JEV_API_KEY="old", DECISION_API_KEY="new")
    assert settings.api_key.get_secret_value() == "new"
    assert DecisionModelSettings(_env_file=None, JEV_API_KEY="old").api_key.get_secret_value() == "old"


@pytest.mark.parametrize("url", ["http://example.com", "https://user:pass@example.com", "https://example.com?key=secret"])
def test_decision_url_rejects_insecure_or_embedded_credentials(url):
    with pytest.raises(ValidationError):
        DecisionModelSettings(_env_file=None, DECISION_API_KEY="fake", DECISION_BASE_URL=url)
