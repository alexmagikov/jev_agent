"""Пути в вопросах совпадают с JSON запроса; критерии учитывают смысл и расчёты."""

import json
import re

import pytest

from scripts.measure_request_sizes import jev_body
from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionProposal, prepare_fields
from src.schemas.extraction_schemas import SourceQuote
from src.verification import (
    VerificationResponse, VerificationState, _RULES, build_verification_questions,
)
from tests.test_source_feedback import inputs


def payload(proposal, messages):
    fields = prepare_fields(proposal, messages, "invoice_draft")
    questions = build_verification_questions(fields, proposal=proposal, messages=messages)
    return json.loads(jev_body(
        VerificationState(messages=messages, document_type="invoice_draft", rules=_RULES),
        questions, VerificationResponse,
    ))


@pytest.mark.parametrize("name,value,quote", [
    ("recipient_name", "ИП Смирнов", "ИП Смирнову"),
    ("subject", "папки", "папки"),
    ("basis_number", "07/25", "07/25"),
])
def test_canonical_question_uses_actual_value_path_and_explicit_rule(name, value, quote):
    messages, proposal = inputs()
    messages = [DocumentMessage(id="m1", role="user",
        text=messages[0].text + " ИП Смирнову, папки, договор № 07/25.")]
    proposal.fields[name].status = "fillable"
    proposal.fields[name].value = value
    proposal.fields[name].evidence = [SourceQuote(message_id="m1", quote=quote)]
    wire = payload(proposal, messages)
    question = wire["questions"][name + "::noncanonical"]
    assert question["instructions"]["field"]["path"] == name
    assert question["instructions"]["field"]["value"] == value
    text = question["instructions"]["question"]
    assert "instructions.field.value" in text
    assert "instructions.rules" in text
    assert not re.search(r"`(?:subject|recipient_name|basis_number)`\.value", text)
    assert "explicit canonical-format rule" in question["criteria"]["true"]
    assert "permitted normalization" in question["criteria"]["false"]


def test_calculated_value_is_supported_without_a_verbatim_result():
    messages, proposal = inputs()
    wire = payload(proposal, messages)
    question = wire["questions"]["amount_rub::unsupported"]
    claim = question["instructions"]["field"]
    assert claim["value"] == 300
    assert "300" not in messages[0].text
    assert claim["calculation"]["expression"] == "count * price"
    assert claim["evidence"] == proposal.fields["amount_rub"].model_dump(mode="json")["evidence"]
    text = question["instructions"]["question"]
    assert "instructions.field.value" in text
    assert "instructions.field.evidence" in text
    assert "instructions.field.calculation" in text
    assert "source operands" in text
    assert "justified calculation" in question["criteria"]["false"]
    assert "need not appear verbatim" in question["criteria"]["false"]
    assert "source operands" in question["criteria"]["true"]


def test_item_list_support_uses_its_own_calculation_container():
    messages, proposal = inputs()
    data = proposal.model_dump(mode="json")
    item = data["fields"]["items"]["value"][0]
    item["unit_price_rub"] = None
    item["unit_price_calculation"] = data["fields"]["amount_rub"]["calculation"]
    proposal = ExtractionProposal.model_validate(data)
    wire = payload(proposal, messages)
    question = wire["questions"]["items::unsupported"]
    claim = question["instructions"]["field"]
    assert claim["calculation"] is None
    assert claim["item_calculations"][0]["unit_price_calculation"]["expression"] == "count * price"
    assert claim["value"][0]["unit_price_rub"] == 300
    assert "instructions.field.item_calculations" in question["instructions"]["question"]


def test_permitted_name_normalization_is_not_missing_support():
    messages, proposal = inputs()
    messages[0].text += " Выставить ИП Смирнову."
    proposal.fields["recipient_name"].status = "fillable"
    proposal.fields["recipient_name"].value = "ИП Смирнов"
    proposal.fields["recipient_name"].evidence = [SourceQuote(message_id="m1", quote="ИП Смирнову")]
    question = payload(proposal, messages)["questions"]["recipient_name::unsupported"]
    assert "permitted normalization" in question["criteria"]["false"]
    assert "supports" in question["criteria"]["false"]


def test_rules_distinguish_logical_field_label_from_json_location():
    messages, proposal = inputs()
    rules = payload(proposal, messages)["state"]["rules"]
    assert "instructions.field.path is a logical field label" in rules
    assert "instructions.field.value" in rules and "instructions.field.evidence" in rules
