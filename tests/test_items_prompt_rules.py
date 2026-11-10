import pytest

from src.schemas.document_schemas import InvoiceFields, TransferFields
from src.llm import build_extraction_prompt
from src.verification import build_verification_questions

EXTRACTION_PROMPT = build_extraction_prompt("invoice_draft")


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
    from src.field_rules import ITEMS_PRESENCE_RULE, ITEMS_EXAMPLES, INVOICE_SUBJECT_RULE

    for rule in (ITEMS_PRESENCE_RULE, ITEMS_EXAMPLES, INVOICE_SUBJECT_RULE):
        assert EXTRACTION_PROMPT.count(rule) == 1
    assert "only a supported description" in ITEMS_PRESENCE_RULE
    assert "leave all other attributes null" in ITEMS_PRESENCE_RULE
    assert "Счёт на 5000 рублей" in ITEMS_EXAMPLES
    assert "Счёт за консультационные услуги на 5000 рублей" in ITEMS_EXAMPLES


@pytest.mark.parametrize("items_present", [False, True])
def test_jev_gets_same_item_rules_for_missing_and_filled_items(items_present):
    from src.field_rules import INVOICE_SUBJECT_RULE

    questions = build_verification_questions(fields(InvoiceFields, items_present=items_present)).root
    relevant = {key: value for key, value in questions.items()
                if key.startswith("items::")}
    assert relevant
    for question in relevant.values():
        assert "Subject and items are not mutually exclusive" in question.model_dump_json()
        assert "An item with only a supported description is fillable" in question.model_dump_json()
        assert "Do not create an item from only a total amount, a recipient name or a contract reference" in question.model_dump_json()
    for key, question in questions.items():
        if key.startswith("subject::"):
            assert INVOICE_SUBJECT_RULE in question.model_dump_json()
            assert "total/count" not in question.model_dump_json()
    if items_present:
        assert "Judge only missing line items; do not assess attributes" in questions["items::incomplete"].model_dump_json()
    else:
        assert "items::missing_supported" in questions


def test_item_specific_rules_do_not_bloat_unrelated_or_transfer_questions():
    for model in (InvoiceFields, TransferFields):
        questions = build_verification_questions(fields(model)).root
        for key, question in questions.items():
            if not key.startswith(("items::", "items[", "subject::")):
                assert "Subject and items are not mutually exclusive" not in question.model_dump_json()


@pytest.mark.parametrize("items_present", [False, True])
def test_llm_and_verifier_share_one_definition_of_item_semantics(items_present):
    from src.field_rules import (
        ITEMS_PRESENCE_RULE, ITEMS_EXAMPLES, INVOICE_SUBJECT_RULE,
        ITEM_QUANTITY_RULE, ITEM_UNIT_PRICE_RULE,
    )
    shared = (ITEMS_PRESENCE_RULE, ITEMS_EXAMPLES, INVOICE_SUBJECT_RULE,
              ITEM_QUANTITY_RULE, ITEM_UNIT_PRICE_RULE)
    questions = build_verification_questions(fields(InvoiceFields, items_present=items_present)).root
    for rule in shared:
        assert EXTRACTION_PROMPT.count(rule) == 1
        for key, question in questions.items():
            if key.startswith("items::"):
                assert question.model_dump_json().count(rule) == 1
    assert "Определи items отдельно от subject" not in EXTRACTION_PROMPT
    assert "Не дели общую сумму нескольких разных товаров" not in EXTRACTION_PROMPT
