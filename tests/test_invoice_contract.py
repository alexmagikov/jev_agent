"""Проверки передачи общего контракта; не оценка качества реальных моделей."""

import asyncio
import json

import httpx2
import pytest

from src.extraction import ExtractionProposal, prepare_fields
from src.llm import AsyncLLMExtractor, build_extraction_prompt
from src.cases.document_type import DocumentMessage
from src.schemas.document_schemas import InvoiceFields
from src.settings import LLMSettings
from src.verification import build_verification_questions

EXTRACTION_PROMPT = build_extraction_prompt("invoice_draft")


def proposal(*, amount_status="missing"):
    fields = {name: {"status": "missing", "value": None, "evidence": []}
              for name in InvoiceFields.model_fields}
    fields["items"] = {"status": "fillable", "value": [{
        "description": "кабели", "quantity": None, "unit": None, "unit_price_rub": 280,
    }, {
        "description": "адаптеры", "quantity": None, "unit": None, "unit_price_rub": 280,
    }], "evidence": [{"message_id": "m1", "quote": dialogue()[0].text}]}
    fields["amount_rub"] = {"status": amount_status, "value": None,
                            "evidence": [] if amount_status == "missing" else [
                                {"message_id": "m1", "quote": dialogue()[0].text},
                            ]}
    return ExtractionProposal.model_validate({"fields": fields})


def dialogue():
    return [DocumentMessage(id="m1", role="user", text=(
        "Счёт за товары кабель и адаптер, по 280 рублей каждый."
    ))]


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("repair", [False, True])
def test_initial_and_repair_requests_include_the_same_approved_contract(protocol, repair):
    from src.field_rules import ITEM_DESCRIPTION_RULE, LINE_CHARGES_RULE

    requests = []
    initial = proposal()

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        reply = initial.model_dump(mode="json")
        if repair:
            reply = {"fields": {"items": reply["fields"]["items"]}}
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
                await extractor.extract(dialogue(), "invoice_draft", **(
                    {"previous_proposal": initial, "repair_fields": ["items"]} if repair else {}
                ))

    asyncio.run(run())
    system = requests[0]["system"] if protocol == "anthropic" else requests[0]["messages"][0]["content"]
    assert ITEM_DESCRIPTION_RULE in system
    assert LINE_CHARGES_RULE in system
    assert "singular nominative" in ITEM_DESCRIPTION_RULE
    assert "quotation marks" in ITEM_DESCRIPTION_RULE and "plural-only" in ITEM_DESCRIPTION_RULE
    assert "Unknown quantities" in LINE_CHARGES_RULE
    assert "за килограмм" in LINE_CHARGES_RULE


@pytest.mark.parametrize("amount_status", ["missing", "ambiguous"])
def test_jev_reports_semantic_description_error_not_grammar_in_repair_feedback(amount_status):
    from src.field_rules import ITEM_DESCRIPTION_RULE, ITEM_DESCRIPTION_SEMANTIC_RULE, LINE_CHARGES_RULE
    from src.verification import verify_fields

    initial = proposal(amount_status=amount_status)
    initial.fields["items"].value[0].description = "розетка"
    prepared = prepare_fields(initial, dialogue(), "invoice_draft")
    questions = build_verification_questions(prepared).root
    assert LINE_CHARGES_RULE in questions["amount_rub::missing_supported"].model_dump_json()
    description = questions["items[0].description::value_wrong"].model_dump_json()
    assert ITEM_DESCRIPTION_SEMANTIC_RULE in description
    assert ITEM_DESCRIPTION_RULE not in description
    assert "Singular nominative normalization preserves identity" in description
    # Проверяющий должен передать правило в ремонт, а не только видеть его сам.
    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            return response_model.model_validate({"answers": {
                key: {"noul": 0.9 if key == "items[0].description::value_wrong" else 0.01}
                for key in questions.root
            }})
    feedback = [str(remark) for remark in asyncio.run(verify_fields(dialogue(), "invoice_draft", initial, prepared, Verifier()))]
    assert len(feedback) == 1
    assert "different product/service" in feedback[0]
    assert "Singular nominative normalization preserves identity" in feedback[0]
    assert "description=консультационные услуги" in EXTRACTION_PROMPT
    assert "description 'консультационные услуги'" not in description


@pytest.mark.parametrize("known_quantity", [None, 2])
def test_partial_repair_removes_invented_total_and_preserves_known_item_data(known_quantity):
    from src.pipeline import DocumentTypeDecision, process_selected_document
    from src.field_rules import ITEM_DESCRIPTION_SEMANTIC_RULE, LINE_CHARGES_RULE

    messages = dialogue()
    initial = proposal()
    initial.fields["items"].value[0].description = "розетка"
    initial.fields["items"].value[1].description = "выключатель"
    if known_quantity is not None:
        messages[0].text += " Кабелей — 2 штуки."
        initial.fields["items"].value[0].quantity = known_quantity
        initial.fields["items"].value[0].unit = "штуки"
        initial.fields["items"].evidence[0].quote = messages[0].text
    # Арифметика верна, но складывать цены единиц вместо сумм строк нельзя.
    data = initial.model_dump(mode="json")
    data["fields"]["amount_rub"] = {
        "status": "fillable", "value": None,
        "evidence": [{"message_id": "m1", "quote": messages[0].text}],
        "calculation": {"expression": "fee + fee", "operands": [
            {"name": "fee", "source": {"message_id": "m1", "quote": "280"}},
        ]},
    }
    initial = ExtractionProposal.model_validate(data)
    before = prepare_fields(initial, messages, "invoice_draft")
    requests, verified = [], []
    corrected_items = initial.fields["items"].model_dump(mode="json")
    for item, name in zip(corrected_items["value"], ("кабель", "адаптер")):
        item["description"] = name
    reply = {"fields": {
        "items": corrected_items,
        "amount_rub": {"status": "missing", "value": None, "evidence": [], "calculation": None},
    }}

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(reply),
        }}]})

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            verified.append(state)
            amount_key = "amount_rub::calculation_operation_wrong" if len(verified) == 1 else "amount_rub::missing_supported"
            assert LINE_CHARGES_RULE in questions.root[amount_key].model_dump_json()
            assert ITEM_DESCRIPTION_SEMANTIC_RULE in questions.root["items[0].description::value_wrong"].model_dump_json()
            return response_model.model_validate({"answers": {
                key: {"noul": 0.9 if len(verified) == 1 and key in (
                    "items[0].description::value_wrong", "items[1].description::value_wrong",
                    "amount_rub::calculation_operation_wrong",
                ) else 0.01}
                for key in questions.root
            }})

    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="fake", LLM_MODEL="small")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                return await process_selected_document(
                    "offline-contract", messages,
                    DocumentTypeDecision(document_type="invoice_draft", confidence=0.9),
                    Verifier(), extractor, initial_proposal=initial,
                )

    result = asyncio.run(run())
    assert before.amount_rub.value == 560
    assert len(requests) == 1 and len(verified) == 2
    request_user = json.loads(requests[0]["messages"][-1]["content"])
    assert set(request_user["repair_fields"]) == {"items", "amount_rub"}
    assert result.fields.amount_rub.model_dump() == {
        "status": "missing", "value": None, "evidence": [], "derivation": None,
    }
    assert [item.description for item in result.fields.items.value] == ["кабель", "адаптер"]
    assert [item.quantity for item in result.fields.items.value] == [known_quantity, None]
    assert [item.unit for item in result.fields.items.value] == ["штуки" if known_quantity else None, None]
    assert all(item.unit_price_rub == 280 for item in result.fields.items.value)
    for name in InvoiceFields.model_fields:
        if name not in ("items", "amount_rub"):
            assert getattr(result.fields, name) == getattr(before, name)
