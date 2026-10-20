"""Покрытие пропусков и фокусировка вопросов; не benchmark качества Jev."""

import asyncio
import json

import httpx2
import pytest

from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionProposal, prepare_fields
from src.llm import AsyncLLMExtractor, EXTRACTION_PROMPT
from src.pipeline import DocumentTypeDecision, process_selected_document
from src.schemas.document_schemas import InvoiceFields, TransferFields
from src.settings import LLMSettings
from src.verification import build_verification_questions


def dialogue():
    return [DocumentMessage(id="m1", role="user", text=(
        "Счёт за товары «Кабель» и «Адаптер», по 280 рублей каждый."
    ))]


def candidate(**attributes):
    fields = {name: {"status": "missing", "value": None, "evidence": []}
              for name in InvoiceFields.model_fields}
    items = [{"description": name, "quantity": None, "unit": None, "unit_price_rub": None,
              **attributes} for name in ("Кабель", "Адаптер")]
    fields["items"] = {"status": "fillable", "value": items,
                       "evidence": [{"message_id": "m1", "quote": dialogue()[0].text}]}
    return ExtractionProposal.model_validate({"fields": fields})


@pytest.mark.parametrize("attribute", ["description", "quantity", "unit", "unit_price_rub"])
def test_each_null_item_attribute_gets_its_own_supported_omission_check(attribute):
    prepared = prepare_fields(candidate(**{attribute: None}), dialogue(), "invoice_draft")
    questions = build_verification_questions(prepared).root
    key = f"items[0].{attribute}::missing_supported"
    assert key in questions
    assert questions[key].instructions["field"]["path"] == f"items[0].{attribute}"
    assert "unambiguously" in questions[key].model_dump_json()
    assert "null is correct" in questions[key].model_dump_json()
    assert f"items[0].{attribute}::value_wrong" not in questions


def test_missing_price_check_is_not_diluted_by_grammar_or_subject_rules():
    from src.field_rules import ITEM_UNIT_PRICE_RULE, ITEM_DESCRIPTION_RULE

    questions = build_verification_questions(prepare_fields(candidate(), dialogue(), "invoice_draft")).root
    text = questions["items[0].unit_price_rub::missing_supported"].model_dump_json()
    assert ITEM_UNIT_PRICE_RULE in text
    assert ITEM_UNIT_PRICE_RULE in EXTRACTION_PROMPT
    assert ITEM_DESCRIPTION_RULE not in text
    assert "Subject and items are not mutually exclusive" not in text
    assert "unknown quantity" in text
    assert "work fee" in text
    description = questions["items[0].description::value_wrong"].model_dump_json()
    assert ITEM_DESCRIPTION_RULE in description
    assert "total/count" not in description
    assert "outer quotation delimiters" in description


def test_missing_quantity_does_not_authorize_inventing_one_from_a_named_item():
    from src.field_rules import ITEM_QUANTITY_RULE

    questions = build_verification_questions(prepare_fields(candidate(), dialogue(), "invoice_draft")).root
    text = questions["items[0].quantity::missing_supported"].model_dump_json()
    assert ITEM_QUANTITY_RULE in text
    assert ITEM_QUANTITY_RULE in EXTRACTION_PROMPT
    assert "quantity=1" in text
    assert "number of named" in text


def test_new_item_checks_do_not_apply_to_transfers():
    fields = TransferFields.model_validate({
        name: {"status": "missing", "value": None, "evidence": [], "derivation": None}
        for name in TransferFields.model_fields
    })
    assert not any(key.startswith("items") for key in build_verification_questions(fields).root)


def test_real_partial_repair_fills_only_supported_item_prices_and_rechecks_them():
    initial = candidate()
    prepared = prepare_fields(initial, dialogue(), "invoice_draft")
    calls, checked = [], []

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            checked.append(state)
            if len(checked) == 1:
                assert "items[0].unit_price_rub::missing_supported" in questions.root
                assert "items[1].unit_price_rub::missing_supported" in questions.root
            else:
                assert "items[0].unit_price_rub::value_wrong" in questions.root
            return response_model.model_validate({"answers": {
                key: {"noul": 0.95 if len(checked) == 1 and key.endswith(
                    ".unit_price_rub::missing_supported") else 0.01}
                for key in questions.root
            }})

    def respond(request):
        calls.append(json.loads(request.content))
        reply = candidate(unit_price_rub=280).model_dump(mode="json")
        return httpx2.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"fields": {"items": reply["fields"]["items"]}}),
        }}]})

    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="fake", LLM_MODEL="small")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                return await process_selected_document(
                    "omitted-price", dialogue(),
                    DocumentTypeDecision(document_type="invoice_draft", confidence=0.9),
                    Verifier(), extractor, initial_proposal=initial,
                )

    result = asyncio.run(run())
    assert len(calls) == 1 and len(checked) == 2
    request_user = json.loads(calls[0]["messages"][-1]["content"])
    assert request_user["repair_fields"] == ["items"]
    assert request_user["repair_paths"] == ["items[0].unit_price_rub", "items[1].unit_price_rub"]
    assert any("unit_price_rub::missing_supported" in remark for remark in request_user["feedback"])
    assert [item.unit_price_rub for item in result.fields.items.value] == [280, 280]
    assert all(item.quantity is None and item.unit is None for item in result.fields.items.value)
    for name in InvoiceFields.model_fields:
        if name != "items":
            assert getattr(result.fields, name) == getattr(prepared, name)
