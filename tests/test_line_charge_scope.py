"""Отдельные платы суммируются; общая плата за обе позиции не удваивается.

Вердикты Jev заданы явно: проверяем контракт запроса, вычисления и область ремонта.
"""

import asyncio
import json

import httpx2
import pytest

from scripts.measure_request_sizes import jev_body
from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionProposal, prepare_fields
from src.field_rules import INVOICE_AMOUNT_RULE, LINE_CHARGES_RULE
from src.llm import AsyncLLMExtractor, build_extraction_prompt
from src.pipeline import DocumentTypeDecision, process_selected_document
from src.schemas.document_schemas import InvoiceFields
from src.settings import LLMSettings
from src.verification import VerificationResponse, VerificationState, _RULES, build_verification_questions


def inputs(shared, *, missing=False):
    statement = "Стоимость обеих работ вместе 100 рублей." if shared else "Стоимость каждой из двух работ 100 рублей."
    messages = [DocumentMessage(id="m1", role="user", text="Счёт за настройку сервера и настройку роутера. " + statement)]
    source = [{"message_id": "m1", "quote": messages[0].text}]
    fields = {name: {"status": "missing", "value": None, "evidence": []} for name in InvoiceFields.model_fields}
    fields["items"] = {"status": "fillable", "value": [
        {"description": name, "quantity": None, "unit": None, "unit_price_rub": None}
        for name in ("настройка сервера", "настройка роутера")
    ], "evidence": source}
    if not missing:
        fields["amount_rub"] = {"status": "fillable", "value": None, "evidence": source,
                                "calculation": {"expression": "fee + fee", "operands": [
                                    {"name": "fee", "source": {"message_id": "m1", "quote": "100"}},
                                ]}}
    return messages, ExtractionProposal.model_validate({"fields": fields})


@pytest.mark.parametrize("shared", [False, True])
def test_wire_and_extraction_distinguish_per_line_fees_from_one_shared_total(shared):
    assert "shared fee for two lines: fee + fee" not in LINE_CHARGES_RULE
    assert "100 for each of two lines: 100+100=200" in LINE_CHARGES_RULE
    assert "100 for both lines together: total=100" in LINE_CHARGES_RULE
    assert LINE_CHARGES_RULE in INVOICE_AMOUNT_RULE
    assert build_extraction_prompt("invoice_draft").count(LINE_CHARGES_RULE) == 1
    messages, initial = inputs(shared)
    checked = prepare_fields(initial, messages, "invoice_draft")
    assert checked.amount_rub.value == 200  # Python проверяет арифметику, не смысл повторного сложения.
    questions = build_verification_questions(checked, proposal=initial, messages=messages)
    payload = json.loads(jev_body(VerificationState(messages=messages, document_type="invoice_draft", rules=_RULES), questions, VerificationResponse))
    question = payload["questions"]["amount_rub::calculation_operation_wrong"]
    assert INVOICE_AMOUNT_RULE in question["instructions"]["rules"]
    assert question["instructions"]["field"]["calculation"]["expression"] == "fee + fee"
    assert payload["state"]["messages"] == [message.model_dump(mode="json") for message in messages]


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_mock_semantic_verdict_repairs_only_total_and_keeps_unknown_item_attributes(protocol, shared, missing):
    messages, initial = inputs(shared, missing=missing)
    before = initial.model_dump(mode="json")
    _, corrected = inputs(shared)
    if shared:
        corrected.fields["amount_rub"].value = 100
        corrected.fields["amount_rub"].calculation = None
    target = 100 if shared else 200
    calls = []

    def respond(request):
        body = json.loads(request.content)
        system = body["system"] if protocol == "anthropic" else body["messages"][0]["content"]
        assert LINE_CHARGES_RULE in system
        user = json.loads(body["messages"][-1]["content"])
        calls.append(user)
        assert user["repair_fields"] == user["repair_paths"] == ["amount_rub"]
        text = json.dumps({"fields": {"amount_rub": corrected.fields["amount_rub"].model_dump(mode="json")}})
        if protocol == "anthropic":
            return httpx2.Response(200, json={"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"})
        return httpx2.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": text}}]})

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            claim = questions.root["amount_rub::status_wrong"].instructions["field"]
            error = "missing_supported" if claim["status"] == "missing" else "calculation_operation_wrong"
            return response_model.model_validate({"answers": {
                key: {"noul": .95 if key == f"amount_rub::{error}" and claim["value"] != target else .01}
                for key in questions.root
            }})

    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture", LLM_API_PROTOCOL=protocol,
                               LLM_RESPONSE_FORMAT="text")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                return await process_selected_document(
                    "line-fees", messages, DocumentTypeDecision(document_type="invoice_draft", confidence=1),
                    Verifier(), extractor, initial_proposal=initial,
                )

    result = asyncio.run(run())
    assert len(calls) == int(missing or shared)
    assert result.fields == prepare_fields(corrected, messages, "invoice_draft")
    assert all(item.quantity is None and item.unit is None and item.unit_price_rub is None for item in result.fields.items.value)
    assert initial.model_dump(mode="json") == before
