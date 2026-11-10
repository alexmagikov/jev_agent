"""Полнота списка и пропуск атрибута задают разные вопросы и области ремонта."""

import asyncio

import pytest

from src.cases.document_type import DocumentMessage
from src.extraction import prepare_fields
from src.pipeline import DocumentTypeDecision, process_selected_document
from src.verification import build_verification_questions
from tests.test_item_attribute_checks import candidate


def inputs():
    text = "Счёт: Кабель — 2 шт. по 280 рублей; Адаптер — 1 шт. по 280 рублей."
    messages = [DocumentMessage(id="m1", role="user", text=text)]
    proposal = candidate(quantity=2, unit="шт.", unit_price_rub=280)
    proposal.fields["items"].value[1].quantity = 1
    proposal.fields["items"].evidence[0].quote = text
    return messages, proposal


@pytest.mark.parametrize("attribute", ["description", "quantity", "unit", "unit_price_rub"])
def test_list_completeness_excludes_attributes_that_have_separate_omission_checks(attribute):
    messages, proposal = inputs()
    setattr(proposal.fields["items"].value[0], attribute, None)
    prepared = prepare_fields(proposal, messages, "invoice_draft")
    questions = build_verification_questions(prepared, proposal=proposal, messages=messages).root
    completeness = questions["items::incomplete"]
    text = completeness.instructions["question"]
    assert "whole line item" in text
    assert "unambiguously selected" in text
    assert "do not assess attributes" in text
    assert "withdrawn" in text
    assert "whole line item" in completeness.criteria["true"]
    assert "attributes" in completeness.criteria["false"]
    assert "items already present" in completeness.criteria["false"]
    assert f"items[0].{attribute}::missing_supported" in questions


@pytest.mark.parametrize("missing_whole_item", [False, True])
def test_omission_feedback_preserves_attribute_scope_and_allows_adding_a_whole_item(missing_whole_item):
    messages, correct = inputs()
    initial = correct.model_copy(deep=True)
    if missing_whole_item:
        initial.fields["items"].value.pop()
    else:
        initial.fields["items"].value[0].quantity = None
    target = "items::incomplete" if missing_whole_item else "items[0].quantity::missing_supported"
    calls, rounds = [], []

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            rounds.append(questions)
            question = questions.root["items::incomplete"]
            assert "do not assess attributes" in question.instructions["question"]
            return response_model.model_validate({"answers": {
                key: {"noul": .95 if len(rounds) == 1 and key == target else .01}
                for key in questions.root
            }})

    class Extractor:
        async def extract(self, *args, **kwargs):
            calls.append(kwargs)
            reply = correct.model_copy(deep=True)
            if not missing_whole_item:
                # Ответ модели нарушает область ремонта; Python обязан сохранить соседние значения.
                reply.fields["items"].value[0].unit = "кг"
                reply.fields["items"].value[1].unit_price_rub = 999
            return reply

    result = asyncio.run(process_selected_document(
        "completeness", messages, DocumentTypeDecision(document_type="invoice_draft", confidence=1),
        Verifier(), Extractor(), initial_proposal=initial,
    ))
    assert len(calls) == 1 and len(rounds) == 2
    assert calls[0]["repair_paths"] == (["items"] if missing_whole_item else ["items[0].quantity"])
    assert result.fields == prepare_fields(correct, messages, "invoice_draft")
    assert result.validation.status == "passed"
