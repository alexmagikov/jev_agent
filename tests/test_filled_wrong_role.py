"""Контракт wrong_role для значений, чьи цитаты содержат несколько владельцев."""

import pytest

from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionProposal, prepare_fields
from src.schemas.document_schemas import InvoiceFields, TransferFields
from src.verification import build_verification_questions


BUYER = {
    "recipient_name": "ООО Бета",
    "recipient_inn": "2222222222",
    "recipient_email": "buyer@example.test",
    "recipient_account": "22222222222222222222",
    "recipient_bik": "222222222",
}
SELLER = {
    "recipient_name": "ООО Альфа",
    "recipient_inn": "1111111111",
    "recipient_email": "seller@example.test",
    "recipient_account": "11111111111111111111",
    "recipient_bik": "111111111",
}
TEXT = (
    "Выставь счёт покупателю ООО Бета: ИНН 2222222222, buyer@example.test, "
    "счёт 22222222222222222222, БИК 222222222. "
    "Продавец ООО Альфа: ИНН 1111111111, seller@example.test, "
    "счёт 11111111111111111111, БИК 111111111. Сумма 1500 рублей."
)


def question_for(name, value, document_type="invoice_draft"):
    model = InvoiceFields if document_type == "invoice_draft" else TransferFields
    fields = {key: {"status": "missing", "value": None, "evidence": []}
              for key in model.model_fields}
    fields[name] = {"status": "fillable", "value": value,
                    "evidence": [{"message_id": "m1", "quote": TEXT}]}
    proposal = ExtractionProposal.model_validate({"fields": fields})
    messages = [DocumentMessage(id="m1", role="user", text=TEXT)]
    prepared = prepare_fields(proposal, messages, document_type)
    questions = build_verification_questions(
        prepared, proposal=proposal, messages=messages, document_type=document_type,
    )
    question = questions.root[name + "::wrong_role"]
    return question.instructions["question"], question


@pytest.mark.parametrize("name", BUYER)
@pytest.mark.parametrize("owner", [BUYER, SELLER], ids=["buyer", "seller"])
def test_invoice_role_question_targets_value_support_with_both_owners_in_one_quote(name, owner):
    check, question = question_for(name, owner[name])
    claim = question.instructions["field"]
    assert claim["value"] == owner[name]
    assert claim["evidence"] == [{"message_id": "m1", "quote": TEXT}]
    assert "valid_source_candidates" not in claim
    assert question.instructions["question"] == check
    assert "instructions.field.evidence" in check
    assert "instructions.field.value" in check
    assert "instructions.field.meaning" in check
    assert "buyer/customer being billed" in check
    assert "same quote" in check
    assert "used to justify" in question.criteria["true"]
    assert "seller" in question.criteria["false"] and "buyer" in question.criteria["false"]
    assert "same quote" in question.criteria["false"]


@pytest.mark.parametrize("document_type", ["invoice_draft", "transfer_draft"])
def test_numerical_role_question_uses_meaning_without_assigning_buyer_ownership(document_type):
    check, question = question_for("amount_rub", 1500, document_type)
    assert "instructions.field.meaning" in check
    assert "source operands" in check
    assert "buyer/customer being billed" not in check
    assert "numerical role" in question.criteria["true"]
    assert "format" in question.criteria["false"].lower()
