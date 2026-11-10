"""Контракт запросов, не проверка качества живой модели Jev."""

import asyncio
import json

import pytest

from scripts.measure_request_sizes import c001_proposal, jev_body
from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionProposal, prepare_fields, requisite_format_context
from src.schemas.document_schemas import InvoiceFields
from src.verification import (
    VerificationResponse, VerificationState, _RULES, build_verification_questions, verify_fields,
)


def invoice(messages, values):
    fields = {name: {"status": "missing", "value": None, "evidence": []}
              for name in InvoiceFields.model_fields}
    for name, value in values.items():
        fields[name] = {"status": "fillable", "value": value, "evidence": [
            {"message_id": messages[0].id, "quote": value},
        ]}
    return ExtractionProposal.model_validate({"fields": fields})


def test_c046_wire_payload_explains_invalid_source_not_incorrect_status():
    messages = [DocumentMessage(id="m1", role="user", text=(
        "Выставь счёт ООО Формат на 2600 рублей за ремонт. ИНН 77012345678, "
        "расчётный счёт 4070281000000000046, БИК 04452546. Цифры не дополняй."
    ))]
    values = {"recipient_inn": "77012345678", "recipient_account": "4070281000000000046",
              "recipient_bik": "04452546"}
    proposal = invoice(messages, values)
    prepared = prepare_fields(proposal, messages, "invoice_draft")
    questions = build_verification_questions(prepared, proposal=proposal, messages=messages)
    state = VerificationState(messages=messages, document_type="invoice_draft", rules=_RULES)
    payload = json.loads(jev_body(state, questions, VerificationResponse))
    assert payload["state"]["messages"] == [m.model_dump(mode="json") for m in messages]
    for name, value in values.items():
        status_claim = payload["questions"][name + "::status_wrong"]["instructions"]["field"]
        assert status_claim["status"] == "invalid"
        assert status_claim["format_check"]["matches_format"] is False
        assert status_claim["format_check"]["checked_by"] == "python"
        assert name + "::missing_supported" not in payload["questions"]
        q = payload["questions"][name + "::wrong_role"]
        assert q["instructions"]["question"]
        assert "cited sources" in q["instructions"]["question"]
        claim = q["instructions"]["field"]
        assert claim["status"] == "invalid" and claim["value"] is None
        assert "format_check" not in claim
        assert "valid_source_candidates" not in claim
        check = requisite_format_context(name, value)
        assert check["input_value"] == value
        assert check["input_length"] == len(value)
        assert check["matches_format"] is False
        assert check["checked_by"] == "python"
        assert "invalid/null is correct" in " ".join(q["instructions"]["rules"])
        assert q["criteria"]["true"] != q["instructions"]["question"]


def test_missing_contact_fields_do_not_invent_a_python_format_verdict():
    messages = [DocumentMessage(id="m1", role="user", text=(
        "Выставь счет ООО Ромашка, ИНН 7701234567, за консультационные услуги "
        "5000 рублей, без НДС."
    ))]
    proposal = c001_proposal()
    prepared = prepare_fields(proposal, messages, "invoice_draft")
    for with_proposal in (proposal, None):
        questions = build_verification_questions(prepared, proposal=with_proposal, messages=messages).root
        for name in ("recipient_email", "recipient_account"):
            assert {key for key in questions if key.startswith(name + "::")} == {name + "::status_wrong"}
            check = questions[name + "::status_wrong"].instructions["field"]["format_check"]
            assert check == requisite_format_context(name)
            assert set(check) == {"rule"}  # Правило не выдаётся за выполненную проверку.


@pytest.mark.parametrize("name,value", [
    ("recipient_inn", "12345678901"),
    ("recipient_account", "4070281000000000046"),
    ("recipient_bik", "04452546"),
    ("recipient_email", "mail@example"),
])
@pytest.mark.parametrize("status", ["missing", "invalid", "ambiguous"])
@pytest.mark.parametrize("with_proposal", [False, True])
def test_null_status_wire_has_format_rule_without_fabricating_a_python_verdict(name, value, status, with_proposal):
    from src.schemas.extraction_schemas import ProposedField

    messages = [DocumentMessage(id="m1", role="user", text=f"Реквизит покупателя: {value}")]
    proposal = invoice(messages, {})
    proposal.fields[name] = ProposedField(
        status=status, value=None,
        evidence=[] if status == "missing" else [{"message_id": "m1", "quote": value}],
    )
    prepared = prepare_fields(proposal, messages, "invoice_draft")
    before = proposal.model_dump(mode="json")
    questions = build_verification_questions(prepared, proposal=proposal if with_proposal else None,
                                             messages=messages)
    payload = json.loads(jev_body(
        VerificationState(messages=messages, document_type="invoice_draft", rules=_RULES),
        questions, VerificationResponse,
    ))
    claim = payload["questions"][f"{name}::status_wrong"]["instructions"]["field"]
    assert claim["status"] == status and claim["value"] is None
    assert claim["format_check"] == requisite_format_context(name)
    assert set(claim["format_check"]) == {"rule"}
    assert "valid_source_candidates" not in claim
    assert "not checked every value in messages" in payload["state"]["rules"]
    assert "when only a rule is supplied, apply it to relevant user data" in payload["state"]["rules"]
    assert payload["state"]["messages"] == [message.model_dump(mode="json") for message in messages]
    assert proposal.model_dump(mode="json") == before
    for key, question in payload["questions"].items():
        if key.startswith(name + "::") and not key.endswith("::status_wrong"):
            assert "format_check" not in question["instructions"]["field"]


@pytest.mark.parametrize("later_correction", [False, True])
def test_invalid_status_keeps_semantic_check_and_can_still_be_rejected(later_correction):
    messages = [DocumentMessage(id="m1", role="user", text="Счёт, БИК 04452546")]
    if later_correction:
        messages.append(DocumentMessage(id="m2", role="user", text="Исправь БИК на 044525225"))
        value = "04452546"
    else:
        messages[0].text = "Счёт, БИК 044525225"
        value = "044525225"
    proposal = invoice(messages, {"recipient_bik": value})
    if not later_correction:
        # A model claims invalid despite citing a well-formed value.
        proposal.fields["recipient_bik"].status = "invalid"
        proposal.fields["recipient_bik"].value = None
    prepared = prepare_fields(proposal, messages, "invoice_draft")

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            q = questions.root["recipient_bik::missing_supported"]
            assert q.instructions["field"]["status"] == "invalid"
            assert "latest" in " ".join(q.instructions["rules"])
            claim = q.instructions["field"]
            assert "format_check" not in claim
            assert "expected_status" not in claim  # No automatic acceptance of invalid.
            if later_correction:
                assert claim["valid_source_candidates"] == [{"message_id": "m2", "quote": "044525225"}]
                assert len(state.messages) == 2
            else:
                assert claim["valid_source_candidates"] == [{"message_id": "m1", "quote": "044525225"}]
            return response_model.model_validate({"answers": {
                key: {"noul": .95 if key == "recipient_bik::missing_supported" else .01}
                for key in questions.root
            }})

    remarks = asyncio.run(verify_fields(messages, "invoice_draft", proposal, prepared, Verifier()))
    assert [(remark.path, remark.check) for remark in remarks] == [("recipient_bik", "missing_supported")]


def test_format_context_is_not_a_claim_that_valid_input_is_invalid():
    messages = [DocumentMessage(id="m1", role="user", text="Счёт, БИК 044525225")]
    proposal = invoice(messages, {"recipient_bik": "044525225"})
    prepared = prepare_fields(proposal, messages, "invoice_draft")
    q = build_verification_questions(prepared, proposal=proposal).root["recipient_bik::unsupported"]
    assert q.instructions["field"]["format_check"]["matches_format"] is True
    assert q.instructions["field"]["status"] == "fillable"


@pytest.mark.parametrize("name,value,valid", [
    ("recipient_inn", "7701234567", True),
    ("recipient_inn", "770123456789", True),
    ("recipient_inn", "77012345678", False),
    ("recipient_account", "40702810000000000046", True),
    ("recipient_account", "4070281000000000046", False),
    ("recipient_bik", "044525225", True),
    ("recipient_bik", "04452546", False),
    ("recipient_bik", "０４４５２５２２５", False),
    ("recipient_email", "mail@example.test", True),
    ("recipient_email", "mail@example", False),
])
def test_reported_format_matches_the_python_validator(name, value, valid):
    from src.extraction import invalid_format, requisite_format_context

    check = requisite_format_context(name, value)
    assert check["matches_format"] is valid
    assert invalid_format(name, value) is not valid
