"""Python проверяет исходную форму единицы, не решая её смысл и принадлежность."""

import asyncio
import json

import httpx2
import pytest

from src.extraction import ExtractionError, ExtractionProposal, prepare_fields
from src.cases.document_type import DocumentMessage
from src.llm import AsyncLLMExtractor
from src.pipeline import DocumentTypeDecision, process_selected_document
from src.schemas.document_schemas import InvoiceFields
from src.settings import LLMSettings


def candidate(unit, quote):
    fields = {name: {"status": "missing", "value": None, "evidence": []}
              for name in InvoiceFields.model_fields}
    fields["items"] = {"status": "fillable", "value": [{
        "description": "товар", "quantity": None, "unit": unit, "unit_price_rub": None,
    }], "evidence": [{"message_id": "m1", "quote": quote}]}
    return ExtractionProposal.model_validate({"fields": fields})


@pytest.mark.parametrize("text,unit", [
    ("Товар: 3 штуки", "штуки"), ("Товар: 3штуки", "штуки"),
    ("Товар: 2,5кг", "кг"), ("Товар: 2м", "м"), ("Товар: 2м²", "м²"),
    ("Товар: 2 кв. м", "кв. м"), ("Товар: 3 шт.", "шт."),
    ("Единица — шт", "шт"), ("Единица — кг.", "кг"), ("Товар без единицы", None),
    ("Товар: 8 шт., стоимость партии 5600 рублей", "шт."),
    ("Товар: 2 компл. по 100 рублей", "компл."),
    ("Товар: 2 уп", "уп"), ("Товар: 2 уп.", "уп."),
    ("Масса 8 кг. Количество подтверждено.", "кг"),
    ("Поставили два комплекта.", "комплекта"),
    ("Бумага: 2 тыс. шт.", "тыс. шт."),
])
def test_exact_unit_is_preserved_with_word_boundaries_and_attached_quantity(text, unit):
    prepared = prepare_fields(candidate(unit, text), [DocumentMessage(id="m1", role="user", text=text)], "invoice_draft")
    assert prepared.items.value[0].unit == unit


@pytest.mark.parametrize("text,unit", [
    ("Товар: 3 штуки", "штук"), ("Товар: 3 штуки", "шт"),
    ("Товар: 3 штук", "штуки"), ("Товар: 5см", "м"),
    ("Товар: 2 комплекта", "коробки"),
    ("Товар: 5м²", "м"), ("Товар: 5м2", "м"),
    ("Товар: 5мкг", "кг"), ("Товар: 3 шт", "шт."),
    ("Товар без единицы", "кг"), ("Товар без единицы", ""),
    ("Товар: 8 шт., стоимость партии 5600 рублей", "шт"),
    ("Товар: 8шт.", "шт"), ("Товар: 8 ШТ.", "ШТ"),
    ("Товар: 2 компл. по 100 рублей", "компл"),
    ("Товар: 2 уп.", "уп"), ("Бумага: 2 тыс. шт.", "тыс. шт"),
])
def test_changed_or_partial_unit_is_rejected_without_silent_normalization(text, unit):
    proposal = candidate(unit, text)
    with pytest.raises(ExtractionError) as raised:
        prepare_fields(proposal, [DocumentMessage(id="m1", role="user", text=text)], "invoice_draft")
    assert raised.value.code == "invalid_source"
    assert "items[0].unit" in str(raised.value)
    assert "исходную форму" in str(raised.value)
    assert proposal.fields["items"].value[0].unit == unit


@pytest.mark.parametrize("text,unit", [
    ("Товар: 3 штуки", "штук"),
    ("Товар: 8 шт., стоимость партии 5600 рублей", "шт"),
])
def test_cropped_quote_cannot_turn_a_word_prefix_into_a_stated_unit(text, unit):
    with pytest.raises(ExtractionError, match=r"items\[0\].unit"):
        prepare_fields(candidate(unit, unit), [
            DocumentMessage(id="m1", role="user", text=text),
        ], "invoice_draft")


def test_units_in_unreferenced_or_assistant_messages_do_not_support_the_field():
    messages = [DocumentMessage(id="m1", role="user", text="Товар без единицы"),
                DocumentMessage(id="m2", role="assistant", text="Единица — кг"),
                DocumentMessage(id="m3", role="user", text="Другой товар: 3 кг")]
    with pytest.raises(ExtractionError, match=r"items\[0\].unit"):
        prepare_fields(candidate("кг", messages[0].text), messages, "invoice_draft")


def test_short_quote_can_reference_a_message_containing_the_exact_unit():
    # Evidence ultimately identifies messages; copying the entire unit-bearing sentence is not required.
    prepared = prepare_fields(candidate("штуки", "Товар"), [
        DocumentMessage(id="m1", role="user", text="Товар: 3 штуки"),
    ], "invoice_draft")
    assert prepared.items.value[0].unit == "штуки"


@pytest.mark.parametrize("use_verifier", [False, True])
@pytest.mark.parametrize("text,bad_unit,good_unit", [
    ("Товар: 3 штуки", "штук", "штуки"),
    ("Товар: 8 шт., стоимость партии 5600 рублей", "шт", "шт."),
])
def test_bad_unit_feedback_reaches_real_retry_before_any_semantic_verification(use_verifier, text, bad_unit, good_unit):
    messages = [DocumentMessage(id="m1", role="user", text=text)]
    bad, good = candidate(bad_unit, messages[0].text), candidate(good_unit, messages[0].text)
    requests, checks = [], []

    def respond(request):
        user = json.loads(json.loads(request.content)["messages"][-1]["content"])
        requests.append(user)
        return httpx2.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": (bad if len(requests) == 1 else good).model_dump_json(),
        }}]})

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            checks.append(state)
            assert len(requests) == 2
            return response_model.model_validate({"answers": {key: {"noul": .01} for key in questions.root}})

    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                return await process_selected_document(
                    "fixture", messages, DocumentTypeDecision(document_type="invoice_draft", confidence=1),
                    Verifier() if use_verifier else None, extractor, use_verifier=use_verifier,
                )

    result = asyncio.run(run())
    assert result.fields.items.value[0].unit == good_unit
    assert len(requests) == 2 and len(checks) == int(use_verifier)
    assert "items[0].unit" in requests[1]["feedback"][0]
    assert result.fields.items.value[0].quantity is None
