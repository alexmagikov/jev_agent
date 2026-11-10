"""Один план, два независимых смысловых вопроса; арифметика выполняется Python.

Оценки верификатора в тестах заданы явно и не измеряют качество реального Jev.
"""

import asyncio
import json

import pytest

from scripts.measure_request_sizes import jev_body
from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionProposal, prepare_fields
from src.pipeline import DocumentTypeDecision, process_selected_document
from src.schemas.document_schemas import InvoiceFields
from src.verification import VerificationResponse, VerificationState, _RULES, build_verification_questions


KINDS = ("calculation_operands_wrong", "calculation_operation_wrong")


def plan(expression, **sources):
    return {"expression": expression, "operands": [
        {"name": name, "source": {"message_id": message_id, "quote": quote}}
        for name, (message_id, quote) in sources.items()
    ]}


def fixture():
    messages = [
        DocumentMessage(id="m1", role="user", text="Датчик: 7 шт., партия 5600 рублей. Кабель: 2 упаковки по 5 м, партия 100 рублей."),
        DocumentMessage(id="m3", role="user", text="Исправление: датчиков 8 шт., партия всё ещё 5600 рублей. Скидка на общий итог 10%."),
        DocumentMessage(id="m7", role="user", text="800 — прежняя цена датчика, не новое согласование."),
    ]
    fields = {name: {"status": "missing", "value": None, "evidence": []} for name in InvoiceFields.model_fields}
    fields["items"] = {"status": "fillable", "value": [
        {"description": "датчик", "quantity": 8, "unit": "шт.", "unit_price_rub": None,
         "unit_price_calculation": plan("total / count", total=("m3", "5600"), count=("m3", "8"))},
        {"description": "кабель", "quantity": None, "unit": "м", "unit_price_rub": 10,
         "quantity_calculation": plan("packs * size", packs=("m1", "2"), size=("m1", "5"))},
    ], "evidence": [{"message_id": m.id, "quote": m.text} for m in messages]}
    fields["amount_rub"] = {"status": "fillable", "value": None,
        "calculation": plan("(sensors + cable) * (1 - discount / 100)",
                            sensors=("m3", "5600"), cable=("m1", "100"), discount=("m3", "10")),
        "evidence": [{"message_id": m.id, "quote": m.text} for m in messages[:2]]}
    return messages, ExtractionProposal.model_validate({"fields": fields})


def test_wire_questions_each_contain_only_their_own_plan_and_goal():
    messages, proposal = fixture()
    before = proposal.model_dump(mode="json")
    fields = prepare_fields(proposal, messages, "invoice_draft")
    questions = build_verification_questions(fields, proposal=proposal, messages=messages)
    body = json.loads(jev_body(VerificationState(messages=messages, document_type="invoice_draft", rules=_RULES),
                               questions, VerificationResponse))
    targets = {"items[0].unit_price_rub": proposal.fields["items"].value[0].unit_price_calculation,
               "items[1].quantity": proposal.fields["items"].value[1].quantity_calculation,
               "amount_rub": proposal.fields["amount_rub"].calculation}
    assert {key for key in body["questions"] if "::calculation_" in key} == {
        f"{path}::{kind}" for path in targets for kind in KINDS}
    assert "items::formula_wrong" not in body["questions"]
    assert "amount_rub::formula_wrong" not in body["questions"]
    for path, calculation in targets.items():
        for kind in KINDS:
            q = body["questions"][f"{path}::{kind}"]
            claim = q["instructions"]["field"]
            assert claim["path"] == path
            assert claim["meaning"]
            assert claim["calculation"] == calculation.model_dump(mode="json")
            assert "item_calculations" not in claim
            assert path in q["instructions"]["question"]
            assert "instructions.field.meaning" in q["instructions"]["question"]
            assert "Python" in q["instructions"]["question"]
        if path.startswith("items["):
            assert f"{path}::value_wrong" not in body["questions"]
    assert body["questions"]["items[0].quantity::value_wrong"]
    assert body["questions"]["items[1].unit_price_rub::value_wrong"]
    assert body["state"]["messages"] == [m.model_dump(mode="json") for m in messages]
    assert proposal.model_dump(mode="json") == before


def test_operand_and_operation_criteria_are_independent():
    messages, proposal = fixture()
    fields = prepare_fields(proposal, messages, "invoice_draft")
    q = build_verification_questions(fields, proposal=proposal, messages=messages).root
    source = q[f"items[0].unit_price_rub::{KINDS[0]}"]
    operation = q[f"items[0].unit_price_rub::{KINDS[1]}"]
    assert "source selection only" in source.instructions["question"]
    assert "superseded" in source.criteria["true"]
    assert "different" in source.criteria["true"]
    assert "operation is judged separately" in source.criteria["false"]
    assert "expression only" in operation.instructions["question"]
    assert "wrong operand source alone" in operation.criteria["false"]


@pytest.mark.parametrize("fault,expected_value", [("source", 800), ("operation", 44800), ("both", 39200)])
def test_mock_verdict_repairs_only_the_target_and_existing_dependent_calculations(fault, expected_value):
    messages, correct = fixture()
    initial = correct.model_copy(deep=True)
    calculation = initial.fields["items"].value[0].unit_price_calculation
    kinds = KINDS if fault == "both" else (KINDS[0] if fault == "source" else KINDS[1],)
    if fault in ("source", "both"):
        calculation.operands[1].source.message_id, calculation.operands[1].source.quote = "m1", "7"
    if fault in ("operation", "both"):
        calculation.expression = "total * count"
    before = initial.model_dump(mode="json")
    fields = prepare_fields(initial, messages, "invoice_draft")
    assert fields.items.value[0].unit_price_rub == expected_value
    calls, rounds = [], []

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            rounds.append(questions)
            return response_model.model_validate({"answers": {
                key: {"noul": .99 if len(rounds) == 1 and key in {f"items[0].unit_price_rub::{kind}" for kind in kinds} else .01}
                for key in questions.root}})

    class Extractor:
        async def extract(self, *args, **kwargs):
            calls.append(kwargs)
            assert kwargs["repair_paths"] == ["amount_rub", "items[0].unit_price_rub", "items[1].quantity"]
            return correct.model_copy(deep=True)

    result = asyncio.run(process_selected_document(
        "formula", messages, DocumentTypeDecision(document_type="invoice_draft", confidence=1),
        Verifier(), Extractor(), initial_proposal=initial))
    assert len(calls) == 1 and len(rounds) == 2
    assert result.fields == prepare_fields(correct, messages, "invoice_draft")
    assert initial.model_dump(mode="json") == before


def test_without_source_plans_no_formula_verdict_is_fabricated_from_derivation_text():
    messages, proposal = fixture()
    fields = prepare_fields(proposal, messages, "invoice_draft")
    questions = build_verification_questions(fields, messages=messages).root
    assert not any("::calculation_" in key or key.endswith("::formula_wrong") for key in questions)
    assert "items[0].unit_price_rub::value_wrong" in questions
