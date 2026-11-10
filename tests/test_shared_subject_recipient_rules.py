"""Правила извлечения self и восстановленные правила назначения счёта из Jev v1."""

import asyncio
import json

import pytest

import src.field_rules as rules
from scripts.measure_request_sizes import jev_body
from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionProposal, prepare_fields
from src.llm import build_extraction_prompt
from src.pipeline import DocumentTypeDecision, process_selected_document
from src.schemas.document_schemas import InvoiceFields, TransferFields
from src.verification import VerificationResponse, VerificationState, _RULES, build_verification_questions


def candidate(document_type, text, name, value, *, status="fillable"):
    messages = [DocumentMessage(id="m1", role="user", text=text)]
    model = InvoiceFields if document_type == "invoice_draft" else TransferFields
    fields = {key: {"status": "missing", "value": None, "evidence": []} for key in model.model_fields}
    fields[name] = {"status": status, "value": value,
                    "evidence": [{"message_id": "m1", "quote": text}] if status != "missing" else []}
    proposal = ExtractionProposal.model_validate({"fields": fields})
    prepared = prepare_fields(proposal, messages, document_type)
    return messages, proposal, prepared


@pytest.mark.parametrize("text,status,value", [
    ("Переведи себе 900 рублей, карту сообщу позже", "missing", None),
    ("Переведи на мою карту, банк пока неизвестен", "fillable", "self"),
    ("Переведи 900 рублей, получателя сообщу позже", "missing", None),
    ("Переведи сотруднику на его карту", "fillable", "self"),
    ("Получатель пока неизвестен", "invalid", None),
    ("Либо себе, либо сотруднику — ещё решу", "ambiguous", None),
])
def test_self_rule_reaches_every_relation_check_and_no_other_field(text, status, value):
    messages, proposal, fields = candidate("transfer_draft", text, "recipient_relation", value, status=status)
    questions = build_verification_questions(fields, proposal=proposal, messages=messages)
    payload = json.loads(jev_body(
        VerificationState(messages=messages, document_type="transfer_draft", rules=_RULES),
        questions, VerificationResponse,
    ))
    rule = rules.TRANSFER_RECIPIENT_RULE
    prompt = build_extraction_prompt("transfer_draft")
    assert prompt.count(rule) == 1
    assert "recipient_relation='self'" in rule
    assert "себе" in rule and "card/bank details" in rule
    assert "document type alone" in rule
    assert "только если пользователь указал собственную карту" not in prompt
    assert payload["state"]["messages"] == [message.model_dump(mode="json") for message in messages]
    assert "recipient_relation::status_wrong" in payload["questions"]
    for key, question in payload["questions"].items():
        if key.startswith("recipient_relation::"):
            assert question["instructions"]["rules"].count(rule) == 1
            assert question["instructions"]["field"]["status"] == status
            assert question["instructions"]["field"]["value"] == value
        else:
            assert rule not in question["instructions"]["rules"]


@pytest.mark.parametrize("text,value,status", [
    ("Счёт за товары: кабели и адаптеры", "товары", "fillable"),
    ("Счёт за товары: кабели и адаптеры", None, "missing"),
    ("Счёт за кабели и адаптеры. Назначение: товары", "товары", "fillable"),
    ("Счёт за кабели и адаптеры. Назначение: закупка материалов", "закупка материалов", "fillable"),
    ("Счёт за консультацию", "консультация", "fillable"),
])
def test_v1_subject_rule_requires_a_common_purpose_for_different_goods(text, value, status):
    messages, proposal, fields = candidate("invoice_draft", text, "subject", value, status=status)
    questions = build_verification_questions(fields, proposal=proposal, messages=messages).root
    rule = rules.INVOICE_SUBJECT_RULE
    assert "For different goods require a stated common purpose" in rule
    assert "never concatenate names" in rule
    assert "one billed category" in rule.lower()
    assert build_extraction_prompt("invoice_draft").count(rule) == 1
    for key, question in questions.items():
        assert rules.TRANSFER_RECIPIENT_RULE not in question.instructions["rules"]
        if key.startswith("subject::"):
            assert rule in question.instructions["rules"]


def test_mock_omission_verdict_can_repair_self_without_inventing_card_or_bank():
    messages, initial, _ = candidate("transfer_draft", "Переведи себе 900 рублей", "recipient_relation", None, status="missing")
    calls = []

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            for key, question in questions.root.items():
                if key.startswith("recipient_relation::"):
                    assert question.instructions["rules"].count(rules.TRANSFER_RECIPIENT_RULE) == 1
            q = questions.root.get("recipient_relation::missing_supported")
            if q is not None:
                assert q.instructions["field"]["path"] == "recipient_relation"
                assert "unambiguous usable value" in q.instructions["question"]
            return response_model.model_validate({"answers": {
                key: {"noul": .95 if key == "recipient_relation::missing_supported" else .01}
                for key in questions.root
            }})

    class Extractor:
        async def extract(self, *args, **kwargs):
            calls.append(kwargs)
            assert kwargs["repair_paths"] == ["recipient_relation"]
            return ExtractionProposal.model_validate({"fields": {"recipient_relation": {
                "status": "fillable", "value": "self",
                "evidence": [{"message_id": "m1", "quote": "себе"}],
            }}})

    result = asyncio.run(process_selected_document(
        "fixture", messages, DocumentTypeDecision(document_type="transfer_draft", confidence=1),
        Verifier(), Extractor(), initial_proposal=initial,
    ))
    assert len(calls) == 1 and result.fields.recipient_relation.value == "self"
    assert result.fields.destination_card_masked.status == "missing"
    assert result.fields.destination_bank.status == "missing"
