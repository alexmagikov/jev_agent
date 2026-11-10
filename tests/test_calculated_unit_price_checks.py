"""Контракт вопросов о расчётной цене; оценки качества Jev здесь не моделируются."""

from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionProposal, prepare_fields
from src.schemas.document_schemas import InvoiceFields
from src.verification import build_verification_questions


TARGET = "items[0].unit_price_rub::calculation_operands_wrong"


def example():
    messages = [
        DocumentMessage(id="m1", role="user", text="Датчик: 7 шт., стоимость партии 5600 рублей."),
        DocumentMessage(id="m3", role="user", text="Исправление: 8 шт., стоимость партии остаётся 5600 рублей."),
        DocumentMessage(id="m7", role="user", text="800 — прежняя цена, не новое согласование."),
    ]
    fields = {name: {"status": "missing", "value": None, "evidence": []}
              for name in InvoiceFields.model_fields}
    fields["items"] = {
        "status": "fillable", "value": [{
            "description": "датчик", "quantity": 8, "unit": "шт.", "unit_price_rub": None,
            "unit_price_calculation": {"expression": "total / count", "operands": [
                {"name": "total", "source": {"message_id": "m3", "quote": "5600"}},
                {"name": "count", "source": {"message_id": "m3", "quote": "8"}},
            ]},
        }], "evidence": [{"message_id": m.id, "quote": m.text} for m in messages],
    }
    return messages, ExtractionProposal.model_validate({"fields": fields})


def questions_for(messages, proposal):
    checked = prepare_fields(proposal, messages, "invoice_draft")
    return checked, build_verification_questions(checked, proposal=proposal, messages=messages).root


def test_calculated_price_operand_question_checks_latest_source_roles():
    messages, proposal = example()
    checked, questions = questions_for(messages, proposal)
    question = questions[TARGET]
    assert checked.items.value[0].unit_price_rub == 700
    assert "source selection only" in question.instructions["question"]
    assert "latest applicable" in question.instructions["question"]
    assert "arithmetic" in question.instructions["question"]
    assert "superseded" in question.criteria["true"]
    assert "different item" in question.criteria["true"]
    assert "positive count" in " ".join(question.instructions["rules"])
    assert "historical" in question.criteria["false"]
    assert question.instructions["field"]["calculation"] == proposal.fields["items"].value[0].unit_price_calculation.model_dump(mode="json")
    assert question.instructions["field"]["evidence"] == proposal.fields["items"].model_dump(mode="json")["evidence"]


def test_direct_price_and_other_attributes_keep_their_existing_questions():
    messages, calculated = example()
    _, calculation_questions = questions_for(messages, calculated)
    direct = calculated.model_copy(deep=True)
    direct.fields["items"].value[0].unit_price_calculation = None
    direct.fields["items"].value[0].unit_price_rub = 800
    _, direct_questions = questions_for(messages, direct)
    direct_target = "items[0].unit_price_rub::value_wrong"
    assert "Check this attribute individually" in direct_questions[direct_target].instructions["question"]
    assert direct_questions[direct_target].criteria["true"] == "Yes. The specific error in instructions.question is present."
    assert TARGET not in direct_questions
    for attribute in ("description", "quantity", "unit"):
        key = f"items[0].{attribute}::value_wrong"
        assert calculation_questions[key].instructions["question"] == direct_questions[key].instructions["question"]
        assert calculation_questions[key].criteria == direct_questions[key].criteria


def test_stale_operand_is_preserved_for_jev_to_judge_even_when_arithmetic_is_valid():
    messages, proposal = example()
    stale = proposal.model_copy(deep=True)
    stale.fields["items"].value[0].unit_price_calculation.operands[1].source.message_id = "m1"
    stale.fields["items"].value[0].unit_price_calculation.operands[1].source.quote = "7"
    checked, questions = questions_for(messages, stale)
    assert checked.items.value[0].unit_price_rub == 800
    claim = questions[TARGET].instructions["field"]
    assert claim["item"]["quantity"] == 8
    assert claim["calculation"]["operands"][1]["source"] == {"message_id": "m1", "quote": "7"}
    assert "superseded" in questions[TARGET].criteria["true"]
