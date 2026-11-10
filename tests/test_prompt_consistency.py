"""Согласованность правил LLM и Jev после ревью промптов; без запросов к моделям."""

import asyncio

import pytest

from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionError, ExtractionProposal, prepare_fields
from src.field_rules import INVOICE_SUBJECT_RULE, ITEMS_EXAMPLES, PERMITTED_ALIASES, VAT_MODE_RULE
from src.llm import build_extraction_prompt
from src.schemas.document_schemas import InvoiceFields, TransferFields
from src.verification import _RULES, build_verification_questions, verify_fields

INVOICE_PROMPT = build_extraction_prompt("invoice_draft")
TRANSFER_PROMPT = build_extraction_prompt("transfer_draft")


def test_subject_is_written_in_nominative_case_keeping_the_users_number():
    assert "nominative" in INVOICE_SUBJECT_RULE and "'за консультацию' -> 'консультация'" in INVOICE_SUBJECT_RULE
    assert INVOICE_PROMPT.count(INVOICE_SUBJECT_RULE) == 1
    assert "'Выставь счёт за консультацию': description=консультация, subject=консультация" in ITEMS_EXAMPLES


def test_extractor_knows_the_invoice_recipient_is_the_buyer_and_all_requisite_lengths():
    assert "покупатель" in INVOICE_PROMPT
    assert "БИК — ровно 9 цифр" in INVOICE_PROMPT and "расчётный счёт — ровно 20 цифр" in INVOICE_PROMPT


def test_jev_status_question_accepts_ambiguity_of_meaning_or_ownership():
    fields = {name: {"status": "missing", "value": None, "evidence": [], "derivation": None}
              for name in TransferFields.model_fields}
    questions = build_verification_questions(TransferFields.model_validate(fields)).root
    assert "unclear meaning or ownership of supplied data" in questions["amount_rub::status_wrong"].model_dump_json()


def test_output_example_uses_a_concrete_service_not_a_generic_label():
    assert '"description":"Услуга"' not in INVOICE_PROMPT
    assert '"description":"консультация"' in INVOICE_PROMPT


def calculated_amount(expression, text, **operands):
    fields = {name: {"status": "missing", "value": None, "evidence": []} for name in InvoiceFields.model_fields}
    fields["amount_rub"] = {"status": "fillable", "value": None, "evidence": [{"message_id": "m1", "quote": text}],
                            "calculation": {"expression": expression, "operands": [
                                {"name": name, "source": {"message_id": "m1", "quote": quote}}
                                for name, quote in operands.items()]}}
    return prepare_fields(ExtractionProposal.model_validate({"fields": fields}),
                          [DocumentMessage(id="m1", role="user", text=text)], "invoice_draft").amount_rub.value


def test_unit_conversion_constants_are_allowed_and_documented():
    assert "0, 1, 60, 100 и 1000" in INVOICE_PROMPT
    assert calculated_amount("grams * price / 1000", "500 граммов по 300 рублей за кг", grams="500", price="300") == 150
    assert calculated_amount("minutes * price / 60", "90 минут по 2000 рублей за час", minutes="90", price="2000") == 3000
    with pytest.raises(ExtractionError):
        calculated_amount("grams * price / 7", "500 граммов по 300 рублей за кг", grams="500", price="300")


def test_genitive_bank_form_is_no_longer_a_permitted_alias():
    assert ("Сбер", "Сбербанк") in PERMITTED_ALIASES[("transfer_draft", "destination_bank")]
    assert "Сбера" not in TRANSFER_PROMPT


def test_vat_included_without_a_rate_is_rate_mode_with_missing_percentage():
    assert "'С НДС' without a percentage means vat_mode='rate' and vat_rate_percent is missing" in VAT_MODE_RULE
    assert INVOICE_PROMPT.count(VAT_MODE_RULE) == 1


def test_minor_prompt_wording_is_explicit():
    assert "Для missing evidence=[]" in INVOICE_PROMPT and "обычно []" not in INVOICE_PROMPT
    assert "Источник: меняй только указанную часть." not in INVOICE_PROMPT


@pytest.mark.parametrize("document_type,has_item_rules", [("invoice_draft", True), ("transfer_draft", False)])
def test_item_description_note_is_sent_to_jev_only_for_invoices(document_type, has_item_rules):
    model = InvoiceFields if document_type == "invoice_draft" else TransferFields
    fields = model.model_validate({name: {"status": "missing", "value": None, "evidence": [], "derivation": None}
                                   for name in model.model_fields})
    proposal = ExtractionProposal.model_validate({"fields": {
        name: {"status": "missing", "value": None, "evidence": []} for name in model.model_fields}})
    captured = []

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            captured.append(state.rules)
            return response_model.model_validate({"answers": {key: {"noul": .01} for key in questions.root}})

    messages = [DocumentMessage(id="m1", role="user", text="Привет")]
    asyncio.run(verify_fields(messages, document_type, proposal, fields, Verifier()))
    assert ("Item descriptions" in captured[0]) is has_item_rules
    assert captured[0].startswith(_RULES[:40])


def test_none_reason_placeholder_is_only_for_jev_not_for_llm():
    from src.cases.document_type import NoneReasonQuestion
    from src.llm import CLASSIFICATION_PROMPT

    jev_criterion = NoneReasonQuestion().criteria.no_request
    assert jev_criterion.endswith("Also the ignored placeholder when a supported request is active.")
    assert "placeholder" not in CLASSIFICATION_PROMPT
    assert "merely discusses a past operation." in CLASSIFICATION_PROMPT
    assert "For invoice_draft and transfer_draft, none_reason must be null." in CLASSIFICATION_PROMPT


def invoice_with_description(description):
    fields = {name: {"status": "missing", "value": None, "evidence": [], "derivation": None}
              for name in InvoiceFields.model_fields}
    fields["items"] = {"status": "fillable", "evidence": ["m1"], "derivation": None, "value": [
        {"description": description, "quantity": None, "unit": None, "unit_price_rub": None}]}
    return InvoiceFields.model_validate(fields)


def test_jev_checks_the_grammatical_form_of_item_descriptions():
    from src.field_rules import ITEM_DESCRIPTION_RULE, ITEM_DESCRIPTION_SEMANTIC_RULE

    questions = build_verification_questions(invoice_with_description("услуги")).root
    question = questions["items[0].description::noncanonical"]
    assert "singular nominative" in question.instructions["question"]
    assert ITEM_DESCRIPTION_RULE in question.instructions["rules"]
    assert ITEM_DESCRIPTION_SEMANTIC_RULE not in question.instructions["rules"]
    assert "singular nominative" in question.criteria["true"]
    assert "plural-only" in question.criteria["false"]
    assert "items[0].description::noncanonical" not in build_verification_questions(
        invoice_with_description(None)).root


@pytest.mark.parametrize("name,expected,actual,aliases,matches", [
    ("source_account_kind", "расчётный счёт", "расчетный счёт", False, True),
    ("source_account_kind", "расчётный счёт", "расчетный", True, True),
    ("items", [{"description": "ёлка"}], [{"description": "елка"}], False, True),
    ("items", [{"description": "услуга"}], [{"description": "услуги"}], False, False),
    # Реквизиты и идентификаторы копируются буквально: «ё» в них — другой символ.
    ("recipient_email", "семён@example.com", "семен@example.com", False, False),
    ("recipient_email", "семён@example.com", "семен@example.com", True, False),
    ("basis_number", "ДОГ-Ё-1", "ДОГ-Е-1", False, False),
    ("basis_number", "ДОГ-Ё-1", "ДОГ-Е-1", True, False),
    ("basis_number", "ДОГ-Ё-1", "ДОГ-Ё-1", False, True),
])
def test_metrics_treat_yo_and_ye_as_the_same_letter(name, expected, actual, aliases, matches):
    from src.metrics import field_matches

    document_type = "transfer_draft" if name == "source_account_kind" else "invoice_draft"
    assert field_matches(document_type, name,
                         {"status": "fillable", "value": expected},
                         {"status": "fillable", "value": actual}, allow_aliases=aliases) is matches


def test_metrics_version_marks_the_yo_rule():
    from src.metrics import MetricsReport

    assert MetricsReport.model_fields["metrics_version"].default == 6


def test_generic_category_words_keep_the_users_form():
    from src.field_rules import ITEM_DESCRIPTION_RULE

    assert "'за услуги' -> 'услуги'" in ITEM_DESCRIPTION_RULE
    assert "'за консультационные услуги' -> 'консультационные услуги'" in ITEM_DESCRIPTION_RULE
    assert "'5 услуг настройки' -> 'услуга настройки'" in ITEM_DESCRIPTION_RULE
    assert "description=консультационные услуги" in ITEMS_EXAMPLES
    assert INVOICE_PROMPT.count(ITEM_DESCRIPTION_RULE) == 1
    question = build_verification_questions(invoice_with_description("услуги")).root[
        "items[0].description::noncanonical"]
    assert ITEM_DESCRIPTION_RULE in question.instructions["rules"]
    assert "'за услуги' -> 'услуга'" not in question.model_dump_json()
    assert "plural services or works without a count" in question.criteria["false"]


@pytest.mark.parametrize("path", ["data/labels.jsonl", "data/labels_v2.jsonl"])
def test_bare_services_are_labeled_as_written(path):
    import json
    from pathlib import Path

    rows = {row["case_id"]: row for row in map(json.loads, (Path(__file__).resolve().parents[1] / path).read_text(
        encoding="utf-8").splitlines())}
    for case_id, description in (("C010", "услуги"), ("C012", "услуги"), ("C001", "консультационные услуги")):
        assert rows[case_id]["fields"]["items"]["value"][0]["description"] == description
    assert rows["C007"]["fields"]["items"]["value"][0]["description"] == "услуга"


def test_extraction_prompt_explains_multi_line_totals_and_returned_values():
    prompt = build_extraction_prompt("invoice_draft")
    # Промахи dev-выборки challenge: сложенные количества и фраза вместо числа в операнде.
    assert "q1 * p1 + q2 * p2" in prompt
    assert "вернул" in prompt
