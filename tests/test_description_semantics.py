"""Проверки контракта смысла описаний; оценки Jev в тестах заданы явно."""

import asyncio
import json

import httpx2
import pytest

from scripts.measure_request_sizes import jev_body
from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionProposal, prepare_fields
from src.field_rules import ITEM_DESCRIPTION_RULE, QUOTED_PRODUCT_NAME_RULE
from src.llm import AsyncLLMExtractor, build_extraction_prompt
from src.pipeline import DocumentTypeDecision, process_selected_document
from src.schemas.document_schemas import InvoiceFields
from src.settings import LLMSettings
from src.verification import VerificationResponse, VerificationState, _RULES, build_verification_questions


def inputs(description):
    messages = [DocumentMessage(id="m1", role="user", text=(
        "Счёт за ручки и маркеры вместе на 1600 рублей, всего 8 штук. "
        "Для обеих позиций единица измерения — шт. Количество каждого товара и цены неизвестны."
    ))]
    fields = {name: {"status": "missing", "value": None, "evidence": []}
              for name in InvoiceFields.model_fields}
    fields["items"] = {"status": "fillable", "value": [
        {"description": name, "quantity": None, "unit": "шт.", "unit_price_rub": None}
        for name in (description, "маркер")
    ], "evidence": [{"message_id": "m1", "quote": messages[0].text}]}
    fields["amount_rub"] = {"status": "fillable", "value": 1600,
                            "evidence": [{"message_id": "m1", "quote": "1600 рублей"}]}
    proposal = ExtractionProposal.model_validate({"fields": fields})
    return messages, proposal, prepare_fields(proposal, messages, "invoice_draft")


def assert_semantic_question(question):
    text = question["instructions"]["question"]
    assert "different product/service" in text
    assert "one requested item" in text and "whole order" in text
    assert "corrections and withdrawals" in text and "source order" in text
    assert "items::incomplete" in text
    assert "or add unsupported" not in text
    assert "number/case" in text and "do not judge quantity, unit or price" in text
    assert "Singular nominative normalization preserves identity" in text
    assert "singular nominative normalization" in question["criteria"]["false"]
    assert "Singular nominative normalization" in " ".join(question["instructions"]["rules"])
    assert "Check the required singular" not in json.dumps(question)
    assert QUOTED_PRODUCT_NAME_RULE in question["instructions"]["rules"]
    assert "different product/service" in question["criteria"]["true"]
    assert "Same requested product/service" in question["criteria"]["false"]
    assert "number/case" in question["criteria"]["false"]


def assert_characteristics_question(question):
    text = question["instructions"]["question"]
    assert "unsupported characteristics" in text and "quoted product name" in text
    assert "corrections and withdrawals" in text and "items::incomplete" in text
    assert "product identity" in text and "separate" in text
    assert "do not judge quantity, unit or price" in text
    assert "characteristics" in question["criteria"]["true"]
    assert "not evidence" in text


@pytest.mark.parametrize("description", ["ручка", "ручки", "карандаш", "красная ручка"])
def test_wire_description_question_has_semantic_boundary_and_keeps_candidate_unchanged(description):
    messages, proposal, fields = inputs(description)
    questions = build_verification_questions(fields, proposal=proposal, messages=messages)
    payload = json.loads(jev_body(
        VerificationState(messages=messages, document_type="invoice_draft", rules=_RULES),
        questions, VerificationResponse,
    ))
    q = payload["questions"]["items[0].description::value_wrong"]
    assert_semantic_question(q)
    assert_characteristics_question(payload["questions"]["items[0].description::unsupported_characteristics"])
    assert q["instructions"]["field"]["value"] == description
    for index in (0, 1):
        for kind in ("value_wrong", "unsupported_characteristics"):
            claim = payload["questions"][f"items[{index}].description::{kind}"]["instructions"]["field"]
            assert claim["item_index"] == index
            assert claim["extracted_items"] == fields.model_dump(mode="json")["items"]["value"]
            assert claim["item"] == claim["extracted_items"][index]
            assert claim["evidence"] == proposal.fields["items"].model_dump(mode="json")["evidence"]
    assert "extracted_items" not in payload["questions"]["items[0].unit::value_wrong"]["instructions"]["field"]
    assert payload["state"]["messages"] == [message.model_dump(mode="json") for message in messages]
    assert "normalized to singular nominative" in payload["state"]["rules"]
    assert "source is plural" in payload["state"]["rules"]
    assert "Do not infer quantity=1 or change units/evidence" in payload["state"]["rules"]
    assert "plural-only nouns" in payload["state"]["rules"]
    for key, question in payload["questions"].items():
        if key.startswith("items::"):
            assert "Singular nominative normalization" in json.dumps(question)
            assert "Check the required singular" not in json.dumps(question)
    # Canonical wording remains an extraction instruction, not a semantic rejection criterion.
    assert ITEM_DESCRIPTION_RULE in build_extraction_prompt("invoice_draft")
    assert "singular nominative" in ITEM_DESCRIPTION_RULE


@pytest.mark.parametrize("initial_name", ["ручка", "ручки", "карандаш", "красная ручка", "красный карандаш"])
def test_mock_semantic_verdict_repairs_wrong_product_not_only_grammatical_number(initial_name):
    messages, initial, _ = inputs(initial_name)
    initial_data = initial.model_dump(mode="json")
    _, corrected, _ = inputs("ручка")
    requests, rounds = [], []

    def respond(request):
        user = json.loads(json.loads(request.content)["messages"][-1]["content"])
        requests.append(user)
        body = corrected.model_dump(mode="json")["fields"]
        return httpx2.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"fields": {name: body[name] for name in user["repair_fields"]}}),
        }}]})

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            q = questions.root["items[0].description::value_wrong"]
            assert_semantic_question(q.model_dump(mode="json"))
            assert_characteristics_question(questions.root[
                "items[0].description::unsupported_characteristics"].model_dump(mode="json"))
            name = q.instructions["field"]["value"]
            rounds.append(name)
            # Explicit fixture scores do not predict the real model's behavior.
            scores = {"items[0].description::value_wrong": .95 if name in ("карандаш", "красный карандаш") else .01,
                      "items[0].description::unsupported_characteristics": .95 if name in ("красная ручка", "красный карандаш") else .01}
            return response_model.model_validate({"answers": {
                key: {"noul": scores.get(key, .01)}
                for key in questions.root
            }})

    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                return await process_selected_document(
                    "fixture", messages, DocumentTypeDecision(document_type="invoice_draft", confidence=1),
                    Verifier(), extractor, initial_proposal=initial,
                )

    result = asyncio.run(run())
    if initial_name in ("ручка", "ручки"):
        assert result.fields.items.value[0].description == initial_name
        assert rounds == [initial_name] and requests == []
    else:
        assert result.fields.items.value[0].description == "ручка"
        assert rounds == [initial_name, "ручка"] and len(requests) == 1
        assert requests[0]["repair_paths"] == ["items[0].description"]
        feedback = requests[0]["feedback"]
        if initial_name in ("карандаш", "красный карандаш"):
            assert any("different product/service" in error for error in feedback)
            assert any("Singular nominative normalization preserves identity" in error for error in feedback)
        if initial_name in ("красная ручка", "красный карандаш"):
            assert any("unsupported_characteristics" in error for error in feedback)
            assert any("unsupported characteristics" in error for error in feedback)
        assert len(feedback) == (2 if initial_name == "красный карандаш" else 1)
    assert result.fields.items.value[1].description == "маркер"
    assert result.fields.items.value[0].quantity is None
    assert result.fields.items.value[0].unit_price_rub is None
    assert result.fields.items.value[0].unit == "шт."
    assert result.fields.amount_rub.value == 1600
    assert initial.model_dump(mode="json") == initial_data


def test_order_of_extracted_items_does_not_assign_them_to_source_list_positions():
    messages, proposal, _ = inputs("ручка")
    proposal.fields["items"].value.reverse()
    fields = prepare_fields(proposal, messages, "invoice_draft")
    questions = build_verification_questions(fields, proposal=proposal, messages=messages).root
    question = questions["items[0].description::value_wrong"]
    assert_semantic_question(question.model_dump(mode="json"))
    assert [item["description"] for item in question.instructions["field"]["extracted_items"]] == ["маркер", "ручка"]
    assert question.instructions["field"]["item_index"] == 0
    assert question.instructions["field"]["value"] == "маркер"


def test_missing_other_product_keeps_its_own_completeness_check():
    messages, proposal, _ = inputs("ручка")
    proposal.fields["items"].value.pop()
    fields = prepare_fields(proposal, messages, "invoice_draft")
    questions = build_verification_questions(fields, proposal=proposal, messages=messages).root
    question = questions["items[0].description::value_wrong"]
    assert_semantic_question(question.model_dump(mode="json"))
    assert len(question.instructions["field"]["extracted_items"]) == 1
    assert "active, unambiguously selected whole line item" in questions["items::incomplete"].instructions["question"]
    assert questions["items::incomplete"].instructions["field"]["value"] == fields.model_dump(mode="json")["items"]["value"]


def test_completeness_verdict_repairs_list_without_blaming_the_supported_description():
    messages, complete, _ = inputs("ручка")
    initial = complete.model_copy(deep=True)
    initial.fields["items"].value.pop()
    requests = []

    def respond(request):
        payload = json.loads(json.loads(request.content)["messages"][-1]["content"])
        requests.append(payload)
        assert payload["repair_paths"] == payload["repair_fields"] == ["items"]
        assert len(payload["feedback"]) == 1 and "items::incomplete" in payload["feedback"][0]
        return httpx2.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"fields": {"items": complete.fields["items"].model_dump(mode="json")}}),
        }}]})

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            question = questions.root["items[0].description::value_wrong"]
            assert_semantic_question(question.model_dump(mode="json"))
            count = len(question.instructions["field"]["extracted_items"])
            return response_model.model_validate({"answers": {
                key: {"noul": .95 if key == "items::incomplete" and count == 1 else .01}
                for key in questions.root
            }})

    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                return await process_selected_document(
                    "fixture", messages, DocumentTypeDecision(document_type="invoice_draft", confidence=1),
                    Verifier(), extractor, initial_proposal=initial,
                )

    result = asyncio.run(run())
    assert len(requests) == 1
    assert result.fields == prepare_fields(complete, messages, "invoice_draft")
