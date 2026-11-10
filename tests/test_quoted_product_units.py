"""Общий контракт названия и единицы комплектации; ответы моделей здесь — заглушки."""

import asyncio
import json
from pathlib import Path

import httpx2
import pytest

from scripts.build_dataset_v2 import build_contents, new_examples
from scripts.measure_request_sizes import jev_body
from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionError, ExtractionProposal, prepare_fields
from src.field_rules import INVOICE_SUBJECT_RULE, ITEM_UNIT_RULE, QUOTED_PRODUCT_NAME_RULE, ITEM_DESCRIPTION_SEMANTIC_RULE
from src.llm import AsyncLLMExtractor, build_extraction_prompt
from src.pipeline import DocumentTypeDecision, process_selected_document
from src.schemas.document_schemas import InvoiceFields
from src.settings import LLMSettings
from src.verification import VerificationResponse, VerificationState, _RULES, build_verification_questions


def proposal(messages, *, unit="комплекта", subject='Подарочный набор "Весна"'):
    fields = {name: {"status": "missing", "value": None, "evidence": []}
              for name in InvoiceFields.model_fields}
    source = [{"message_id": "m1", "quote": messages[0].text}]
    fields["subject"] = {"status": "fillable", "value": subject, "evidence": source}
    fields["items"] = {"status": "fillable", "value": [{
        "description": 'Подарочный набор "Весна"', "quantity": 2, "unit": unit,
        "unit_price_rub": 150,
    }], "evidence": source}
    fields["amount_rub"] = {
        "status": "fillable", "value": None, "evidence": source,
        "calculation": {"expression": "count * price", "operands": [
            {"name": "count", "source": {"message_id": "m1", "quote": "2"}},
            {"name": "price", "source": {"message_id": "m1", "quote": "150"}},
        ]},
    }
    return ExtractionProposal.model_validate({"fields": fields})


def dialogue(unit="комплекта"):
    return [DocumentMessage(id="m1", role="user", text=(
        f'Счёт за товар «Подарочный набор "Весна"»: 2 {unit} по 150 рублей.'
    ))]


def test_c063_source_and_generated_json_have_real_quotes_not_literal_backslashes():
    case, label = next(pair for pair in new_examples() if pair[0]["case_id"] == "C063")
    text = case["messages"][0]["text"]
    name = 'Набор "Лето"'
    assert name in text and "\\" not in text
    assert label["fields"]["subject"]["value"] == name
    assert label["fields"]["items"]["value"][0]["description"] == name
    assert label["fields"]["items"]["value"][0]["unit"] == "комплекта"
    for filename, generated in build_contents().items():
        saved = (Path(__file__).resolve().parents[1] / "data" / filename).read_text()
        assert saved == generated
        row = next(row for line in saved.splitlines()
                   if (row := json.loads(line))["case_id"] == "C063")
        value = row["messages"][0]["text"] if "messages" in row else row["fields"]["subject"]["value"]
        assert name in value and "\\" not in value


@pytest.mark.parametrize("unit", ["комплекта", "коробки", None])
def test_wire_and_extraction_share_packaging_vs_product_name_and_verbatim_unit_contract(unit):
    messages = dialogue(unit or "комплекта")
    initial = proposal(messages, unit=unit)
    fields = prepare_fields(initial, messages, "invoice_draft")
    questions = build_verification_questions(fields, proposal=initial, messages=messages)
    state = VerificationState(messages=messages, document_type="invoice_draft", rules=_RULES)
    payload = json.loads(jev_body(state, questions, VerificationResponse))
    key = "items[0].unit::" + ("missing_supported" if unit is None else "value_wrong")
    question = payload["questions"][key]
    assert ITEM_UNIT_RULE in question["instructions"]["rules"]
    assert "комплекта" in ITEM_UNIT_RULE
    assert "copy" in ITEM_UNIT_RULE and "stated form" in ITEM_UNIT_RULE
    assert "product name" in ITEM_UNIT_RULE and "separate unit label" in ITEM_UNIT_RULE
    assert "2 консультации" in ITEM_UNIT_RULE  # A counted service noun is still not a unit.
    claim = question["instructions"]["field"]
    assert claim["value"] == unit
    assert claim["item"]["description"] == 'Подарочный набор "Весна"'
    assert payload["state"]["messages"] == [m.model_dump(mode="json") for m in messages]
    prompt = build_extraction_prompt("invoice_draft")
    assert prompt.count(ITEM_UNIT_RULE) == 1


@pytest.mark.parametrize("subject", [
    'товар «Подарочный набор "Весна"»', 'Подарочный набор "Весна"', 'Товар года',
])
def test_quoted_subject_has_its_own_format_check_not_only_source_support(subject):
    messages = dialogue()
    initial = proposal(messages, subject=subject)
    fields = prepare_fields(initial, messages, "invoice_draft")
    questions = build_verification_questions(fields, proposal=initial, messages=messages).root
    question = questions["subject::noncanonical"]
    assert question.instructions["field"]["value"] == subject
    assert INVOICE_SUBJECT_RULE in question.instructions["rules"]
    assert QUOTED_PRODUCT_NAME_RULE in question.instructions["rules"]
    assert "outer quotation delimiters" in QUOTED_PRODUCT_NAME_RULE
    assert "embedded quotes" in QUOTED_PRODUCT_NAME_RULE
    assert "inside the name" in QUOTED_PRODUCT_NAME_RULE
    assert "verbatim" in QUOTED_PRODUCT_NAME_RULE
    assert "товар" in QUOTED_PRODUCT_NAME_RULE
    assert build_extraction_prompt("invoice_draft").count(INVOICE_SUBJECT_RULE) == 1
    assert build_extraction_prompt("invoice_draft").count(QUOTED_PRODUCT_NAME_RULE) == 1
    assert ITEM_DESCRIPTION_SEMANTIC_RULE in questions["items[0].description::value_wrong"].instructions["rules"]


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
def test_mock_repair_restores_explicit_unit_and_name_without_changing_quantity_or_formula(protocol):
    messages = dialogue()
    initial = proposal(messages, unit=None, subject='товар «Подарочный набор "Весна"»')
    corrected = proposal(messages)
    requests, rounds = [], []

    def respond(request):
        body = json.loads(request.content)
        user = json.loads(body["messages"][-1]["content"])
        system = body["system"] if protocol == "anthropic" else body["messages"][0]["content"]
        requests.append((system, user))
        data = corrected.model_dump(mode="json")["fields"]
        content = json.dumps({"fields": {name: data[name] for name in user["repair_fields"]}})
        return httpx2.Response(200, json=(
            {"content": [{"type": "text", "text": content}], "stop_reason": "end_turn"}
            if protocol == "anthropic" else
            {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        ))

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            rounds.append(questions)
            assert "subject::noncanonical" in questions.root
            # Explicit fixture scores test the repair plumbing, not real Jev accuracy.
            return response_model.model_validate({"answers": {
                key: {"noul": .95 if len(rounds) == 1 and key in (
                    "subject::noncanonical", "items[0].unit::missing_supported",
                ) else .01} for key in questions.root
            }})

    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture",
                               LLM_API_PROTOCOL=protocol, LLM_RESPONSE_FORMAT="text")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                return await process_selected_document(
                    "fixture", messages, DocumentTypeDecision(document_type="invoice_draft", confidence=1),
                    Verifier(), extractor, initial_proposal=initial,
                )

    result = asyncio.run(run())
    expected = prepare_fields(corrected, messages, "invoice_draft")
    assert result.fields == expected
    assert result.fields.amount_rub.value == 300
    assert len(requests) == 1 and len(rounds) == 2
    system, user = requests[0]
    assert ITEM_UNIT_RULE in system and INVOICE_SUBJECT_RULE in system
    assert QUOTED_PRODUCT_NAME_RULE in system
    assert "items[0].unit" in user["repair_paths"] and "subject" in user["repair_paths"]
    assert user["previous_checked_fields"]["items"]["value"][0]["unit"] is None


def test_name_normalization_does_not_relax_exact_evidence_membership():
    messages = dialogue()
    initial = proposal(messages)
    initial.fields["items"].evidence[0].quote = 'Набор Весна'
    with pytest.raises(ExtractionError, match="Цитата источника отсутствует"):
        prepare_fields(initial, messages, "invoice_draft")


def test_quoted_name_rules_do_not_dilute_price_quantity_unit_or_transfer_checks():
    messages = dialogue()
    initial = proposal(messages)
    prepared = prepare_fields(initial, messages, "invoice_draft")
    questions = build_verification_questions(prepared, proposal=initial, messages=messages).root
    for key, question in questions.items():
        expected = key.startswith(("subject::", "items::")) or ".description::" in key
        assert (QUOTED_PRODUCT_NAME_RULE in question.instructions["rules"]) == expected
    transfer_prompt = build_extraction_prompt("transfer_draft")
    assert QUOTED_PRODUCT_NAME_RULE not in transfer_prompt
    assert ITEM_UNIT_RULE not in transfer_prompt
