"""Приоритет формы единицы количества перед формой в цене: контракт обоих промптов."""

import asyncio
import json
from pathlib import Path

import httpx2
import pytest

from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionProposal, prepare_fields
from src.field_rules import ITEM_UNIT_RULE
from src.llm import AsyncLLMExtractor, build_extraction_prompt
from src.schemas.document_schemas import InvoiceFields
from src.settings import LLMSettings
from src.verification import build_verification_questions


def candidate(messages, description, quantity, unit, price):
    fields = {name: {"status": "missing", "value": None, "evidence": []}
              for name in InvoiceFields.model_fields}
    fields["items"] = {
        "status": "fillable", "value": [{
            "description": description, "quantity": quantity, "unit": unit, "unit_price_rub": price,
        }], "evidence": [{"message_id": messages[-1].id, "quote": messages[-1].text}],
    }
    return ExtractionProposal.model_validate({"fields": fields})


@pytest.mark.parametrize("scenario", ["t001_quantity_and_price", "price_unit_only"])
def test_unit_source_priority_is_shared_by_actual_extraction_payload_and_jev_question(scenario):
    if scenario == "t001_quantity_and_price":
        path = Path(__file__).resolve().parents[1] / "data/cases_tricky35.jsonl"
        case = next(row for row in map(json.loads, path.read_text().splitlines()) if row["case_id"] == "T001")
        messages = [DocumentMessage.model_validate(m) for m in case["messages"] if m["id"] == "m5"]
        assert "4 часа по 2400 рублей за час" in messages[0].text
        expected = candidate(messages, "аудит", 4, "часа", 2400)
    else:
        messages = [DocumentMessage(id="m1", role="user", text="Кабель по 115 рублей за м, длина пока неизвестна.")]
        expected = candidate(messages, "кабель", None, "м", 115)
    requests = []
    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        system = body["messages"][0]["content"]
        assert ITEM_UNIT_RULE in system
        assert "Prefer the quantity's unit" in system
        assert "price's unit only" in system
        assert "шт. stays шт." in system
        return httpx2.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": expected.model_dump_json(),
        }}]})

    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture",
                               LLM_API_PROTOCOL="openai", LLM_RESPONSE_FORMAT="text")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                proposal = await extractor.extract(messages, "invoice_draft")
        prepared = prepare_fields(proposal, messages, "invoice_draft")
        return prepared, build_verification_questions(prepared, proposal=proposal, messages=messages)
    prepared, questions = asyncio.run(run())
    question = questions.root["items[0].unit::value_wrong"]
    assert ITEM_UNIT_RULE in question.instructions["rules"]
    assert "Prefer the quantity's unit" in question.model_dump_json()
    assert "price's unit only" in question.model_dump_json()
    assert prepared.items.value[0].unit == expected.fields["items"].value[0].unit
    assert len(requests) == 1


def test_price_form_does_not_become_a_quantity_unit_just_by_appearing_in_same_source():
    messages = [DocumentMessage(id="m1", role="user", text="Аудит: 4 часа по 2400 рублей за час.")]
    proposal = candidate(messages, "аудит", 4, "час", 2400)
    prepared = prepare_fields(proposal, messages, "invoice_draft")
    question = build_verification_questions(prepared, proposal=proposal, messages=messages).root["items[0].unit::value_wrong"]
    # Python проверяет дословность, а приоритет смысловой роли остаётся явной задачей Jev.
    assert question.instructions["field"]["value"] == "час"
    assert question.instructions["field"]["item"]["quantity"] == 4
    assert question.instructions["field"]["evidence"][0]["quote"] == messages[0].text
    assert "Prefer the quantity's unit" in question.model_dump_json()
    assert "-> часа" in ITEM_UNIT_RULE
    assert ITEM_UNIT_RULE in build_extraction_prompt("invoice_draft")
    assert ITEM_UNIT_RULE not in build_extraction_prompt("transfer_draft")
