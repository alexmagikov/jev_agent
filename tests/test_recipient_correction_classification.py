"""Контракт классификации после исправлений; ответы моделей заданы фикстурами.

Проверяется реальная сериализация и маршрутизация, не качество живых моделей.
"""

import asyncio
import json

import httpx2
import pytest

from src.cases.document_type import DocumentMessage, DocumentTypeCase, DocumentTypeState, NONE_REASON_TEXT
from src.client import DecisionModelClient
from src.llm import AsyncLLMExtractor, CLASSIFICATION_PROMPT
from src.pipeline import process_document
from src.schemas.document_schemas import InvoiceFields, TransferFields
from src.settings import LLMSettings


def test_both_independent_questions_apply_corrections_before_support_checks():
    questions = DocumentTypeCase(state=DocumentTypeState(messages=[
        DocumentMessage(id="m1", role="user", text="Переведи себе"),
        DocumentMessage(id="m2", role="user", text="Нет, переведи маме"),
    ])).questions.model_dump(mode="json")
    instructions = questions["document_type"]["instructions"]
    assert "before checking the supported-draft criteria" in instructions["precedence_rule"]
    assert "only if the updated request still satisfies" in instructions["precedence_rule"]
    assert "changes details, not the document type" not in json.dumps(questions)
    assert "support evaluated after those updates" in instructions["request_rule"]
    transfer = questions["document_type"]["criteria"]["transfer_draft"]
    assert "latest applicable request after corrections" in transfer
    reason = questions["none_reason"]
    assert "Apply all later corrections before checking" in reason["instructions"]
    assert "another person's card" in reason["criteria"]["other_document"]
    assert "recipient/card correction" in reason["criteria"]["other_document"]
    assert instructions["precedence_rule"] in CLASSIFICATION_PROMPT
    assert reason["criteria"]["other_document"] in CLASSIFICATION_PROMPT


SCENARIOS = [
    (["Переведи себе", "Нет, переведи маме"], "none", "other_document", None),
    (["Переведи маме", "Нет, переведи себе, на мою карту"], "transfer_draft", None, 1),
    (["Переведи себе", "Нет, на другую мою карту"], "transfer_draft", None, 1),
    (["Переведи себе 5000 рублей", "Сумму исправь на 6000 рублей"], "transfer_draft", None, 0),
    (["Переведи себе", "Номер карты и банк сообщу позже"], "transfer_draft", None, 0),
    (["Выставь счёт покупателю А", "Покупателя исправь на покупателя Б"], "invoice_draft", None, None),
    (["Переведи себе", "Отмени подготовку перевода"], "none", "cancelled", None),
    (["Переведи себе", "Нет, маме", "Всё-таки себе, на мою карту"], "transfer_draft", None, 2),
]


@pytest.mark.parametrize("mode", ["full", "llm-only"])
@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("texts,kind,reason,owner_index", SCENARIOS)
def test_mock_classification_routes_updated_dialogue_without_an_extra_classification(
    monkeypatch, mode, protocol, texts, kind, reason, owner_index,
):
    import src.client as client_module

    messages = [DocumentMessage(id=f"m{i+1}", role="user", text=text) for i, text in enumerate(texts)]
    serialized = [m.model_dump(mode="json") for m in messages]
    llm_calls, jev_calls = [], []

    def jev_response(request):
        body = json.loads(request.content)
        jev_calls.append(body)
        assert body["state"]["messages"] == serialized
        if "document_type" in body["questions"]:
            assert "only if the updated request still satisfies" in body["questions"]["document_type"]["instructions"]["precedence_rule"]
            assert "Apply all later corrections before checking" in body["questions"]["none_reason"]["instructions"]
            options = {"document_type": (["invoice_draft", "transfer_draft", "none"], kind),
                       "none_reason": (["no_request", "other_document", "cancelled"], reason or "no_request")}
            answers = {name: {"type": "choice", "choice": chosen, "confidence": 1,
                              "probabilities": {value: int(value == chosen) for value in values}}
                       for name, (values, chosen) in options.items()}
        else:
            answers = {key: {"type": "noul", "noul": .01} for key in body["questions"]}
        return httpx2.Response(200, json={"model": "fixture", "usage": {"input_tokens": 10, "output_tokens": 10},
                                         "answers": answers})

    def llm_response(request):
        body = json.loads(request.content)
        llm_calls.append(body)
        user = json.loads(body["messages"][-1]["content"])
        assert user["messages"] == serialized
        if "document_type" not in user:
            system = body["system"] if protocol == "anthropic" else body["messages"][0]["content"]
            assert "only if the updated request still satisfies" in system
            assert "recipient/card correction" in system
            reply = {"document_type": kind, "confidence": 1, "none_reason": reason}
        else:
            assert kind != "none" and user["document_type"] == kind
            model = InvoiceFields if kind == "invoice_draft" else TransferFields
            fields = {name: {"status": "missing", "value": None, "evidence": [], "calculation": None}
                      for name in model.model_fields}
            if kind == "transfer_draft":
                fields["recipient_relation"] = {"status": "fillable", "value": "self", "calculation": None,
                    "evidence": [{"message_id": messages[owner_index].id, "quote": messages[owner_index].text}]}
            reply = {"fields": fields}
        content = json.dumps(reply)
        if protocol == "anthropic":
            return httpx2.Response(200, json={"content": [{"type": "text", "text": content}], "stop_reason": "end_turn"})
        return httpx2.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}]})

    real_sdk = client_module.AsyncTypeSafeClient
    monkeypatch.setattr(client_module, "AsyncTypeSafeClient", lambda **kwargs: real_sdk(
        **kwargs, transport=httpx2.MockTransport(jev_response)))

    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture",
                               LLM_API_PROTOCOL=protocol, LLM_RESPONSE_FORMAT="text")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(llm_response)) as http:
            async with AsyncLLMExtractor(settings, client=http) as extractor:
                return await process_document("recipient-change", messages,
                    DecisionModelClient("fixture") if mode == "full" else None,
                    extractor, use_verifier=mode == "full")

    result = asyncio.run(run())
    assert result.document_type == kind
    assert len(llm_calls) == (int(kind != "none") + int(mode == "llm-only"))
    assert len(jev_calls) == (1 + int(kind != "none") if mode == "full" else 0)
    if kind == "none":
        assert result.fields == {} and result.reason == NONE_REASON_TEXT[reason]
    elif kind == "transfer_draft":
        assert result.fields.recipient_relation.value == "self"
        assert result.fields.recipient_relation.evidence == [messages[owner_index].id]
        assert result.fields.destination_card_masked.value is None
        assert result.fields.destination_bank.value is None
