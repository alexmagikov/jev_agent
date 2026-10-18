import pytest

from src.schemas.document_schemas import InvoiceFields, TransferFields
from src.llm import EXTRACTION_PROMPT
from src.verification import build_verification_questions


def fields(model, *, items_present=False):
    data = {name: {"status": "missing", "value": None, "evidence": [], "derivation": None}
            for name in model.model_fields}
    if items_present:
        data["items"] = {
            "status": "fillable", "value": [{
                "description": "консультационные услуги", "quantity": None,
                "unit": None, "unit_price_rub": None,
            }], "evidence": ["m1"], "derivation": None,
        }
        data["subject"] = {
            "status": "fillable", "value": "консультационные услуги",
            "evidence": ["m1"], "derivation": None,
        }
    return model.model_validate(data)


def test_extractor_prompt_explicitly_separates_item_presence_from_completeness():
    assert "Определи items отдельно от subject" in EXTRACTION_PROMPT
    assert "может подтверждать и subject, и description позиции" in EXTRACTION_PROMPT
    assert "items.status='fillable', даже когда известно только" in EXTRACTION_PROMPT
    assert "Не создавай позицию только по общей сумме, названию получателя или ссылке на договор" in EXTRACTION_PROMPT
    assert "Счёт на 5000 рублей" in EXTRACTION_PROMPT
    assert "Счёт за консультационные услуги на 5000 рублей" in EXTRACTION_PROMPT


@pytest.mark.parametrize("items_present", [False, True])
def test_jev_gets_same_item_rules_for_missing_and_filled_items(items_present):
    from src.field_rules import INVOICE_SUBJECT_RULE

    questions = build_verification_questions(fields(InvoiceFields, items_present=items_present)).root
    relevant = {key: value for key, value in questions.items()
                if key.startswith("items::")}
    assert relevant
    for question in relevant.values():
        assert "Subject and items are not mutually exclusive" in question.instructions
        assert "An item with only a supported description is fillable" in question.instructions
        assert "Do not create an item from only a total amount, a recipient name or a contract reference" in question.instructions
    for key, question in questions.items():
        if key.startswith("subject::"):
            assert INVOICE_SUBJECT_RULE in question.instructions
            assert "total/count" not in question.instructions
    if items_present:
        assert "Only flag parts supplied by the user" in questions["items::incomplete"].instructions
    else:
        assert "items::status_wrong" in questions


def test_item_specific_rules_do_not_bloat_unrelated_or_transfer_questions():
    for model in (InvoiceFields, TransferFields):
        questions = build_verification_questions(fields(model)).root
        for key, question in questions.items():
            if not key.startswith(("items::", "items[", "subject::")):
                assert "Subject and items are not mutually exclusive" not in question.instructions
