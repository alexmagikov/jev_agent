"""Проверяем реальные тела запросов: правила только выбранного документа."""
import asyncio
import json

import httpx2
import pytest

from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionProposal, get_proposal_schema, prepare_fields
from src.field_rules import INVOICE_AMOUNT_RULE, TRANSFER_ALIAS_RULES
from src.llm import AsyncLLMExtractor
from src.schemas.document_schemas import InvoiceFields, TransferFields
from src.settings import LLMSettings


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("document_type", ["invoice_draft", "transfer_draft"])
def test_request_contains_only_selected_document_rules(protocol, repair, document_type):
    model = InvoiceFields if document_type == "invoice_draft" else TransferFields
    fields = {name: {"status": "missing", "value": None, "evidence": [], "calculation": None}
              for name in model.model_fields}
    initial = ExtractionProposal.model_validate({"fields": fields})
    messages = [DocumentMessage(id="m1", role="user", text="Подготовь документ")]
    captured = []

    def respond(request):
        captured.append(json.loads(request.content))
        reply = {"fields": {"amount_rub": fields["amount_rub"]}} if repair else {"fields": fields}
        content = json.dumps(reply)
        return httpx2.Response(200, json=(
            {"content": [{"type": "text", "text": content}], "stop_reason": "end_turn"}
            if protocol == "anthropic" else
            {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        ))

    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="fake", LLM_MODEL="small",
                               LLM_API_PROTOCOL=protocol, LLM_RESPONSE_FORMAT="text")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                return await extractor.extract(messages, document_type, **({
                    "previous_proposal": initial, "repair_fields": ["amount_rub"],
                    "repair_paths": ["amount_rub"], "feedback": ["Проверь сумму"],
                    "previous_checked_fields": prepare_fields(initial, messages, document_type),
                } if repair else {}))

    result = asyncio.run(run())
    assert set(result.fields) == set(model.model_fields)
    assert len(captured) == 1
    body = captured[0]
    system = body["system"] if protocol == "anthropic" else body["messages"][0]["content"]
    user = json.loads(body["messages"][-1]["content"])
    assert user["document_type"] == document_type
    assert user["messages"] == [message.model_dump() for message in messages]
    expected_fields = {"amount_rub"} if repair else set(model.model_fields)
    assert set(user["template"]["fields"]) == expected_fields
    assert user["output_schema"] == get_proposal_schema(
        document_type, field_names=["amount_rub"] if repair else None,
    )
    assert "Используй только сведения пользователя" in system
    assert "позднее явное исправление" in system
    assert "ambiguous" in system and "invalid" in system
    assert "expression" in system and "operands" in system
    assert "repair_paths" in system
    if repair:
        assert user["repair_paths"] == ["amount_rub"]
        assert user["previous_proposal"] == initial.model_dump(mode="json")
        assert user["previous_checked_fields"] == prepare_fields(initial, messages, document_type).model_dump(mode="json")
        assert user["python_changes"] == {}
    else:
        assert "previous_checked_fields" not in user and "python_changes" not in user
    if document_type == "transfer_draft":
        for unrelated in ("items", "subject", "ИНН", "НДС", "basis_number", "quantity_calculation",
                          "unit_price_rub", "<example_calculation>", "<example_output>"):
            assert unrelated not in system
        assert "recipient_relation='self'" in system
        assert "Не восстанавливай скрытые цифры карты" in system
        assert all(rule in system for rule in TRANSFER_ALIAS_RULES.values())
    else:
        assert INVOICE_AMOUNT_RULE in system
        assert "ИНН — ровно 10 или 12 цифр" in system
        assert "НДС" in system and "quantity_calculation" in system
        assert "items[i].attribute" in system
        assert "recipient_relation" not in system
        assert all(rule not in system for rule in TRANSFER_ALIAS_RULES.values())
