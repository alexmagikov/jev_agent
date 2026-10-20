import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.cases.document_type import DocumentMessage
from src.schemas.document_schemas import InvoiceFields, TransferFields
from src.extraction import ExtractionProposal, prepare_fields
import src.verification as verification
from src.verification import VerificationError, verify_fields


def _inputs():
    messages = [DocumentMessage(id="m1", role="user", text="Переведи себе 100 рублей")]
    proposal_fields = {
        name: {"status": "missing", "value": None, "evidence": []}
        for name in TransferFields.model_fields
    }
    proposal_fields["amount_rub"] = {
        "status": "fillable",
        "value": 100,
        "evidence": [{"message_id": "m1", "quote": "100 рублей"}],
    }
    proposal = ExtractionProposal.model_validate({"fields": proposal_fields})
    fields = prepare_fields(proposal, messages, "transfer_draft")
    return messages, proposal, fields


class FakeJev:
    def __init__(self, *, probability=0.01, key_change=None, error=None):
        self.probability = probability
        self.key_change = key_change
        self.error = error

    async def ainvoke(self, state, questions, response_model):
        if self.error is not None:
            raise self.error
        answers = {key: {"noul": self.probability} for key in questions.root}
        if self.key_change == "missing":
            answers.pop(next(iter(answers)))
        elif self.key_change == "extra":
            answers["unexpected::check"] = {"noul": 0.01}
        return response_model.model_validate({"answers": answers})


def _verify(client, *, threshold=0.7):
    messages, proposal, fields = _inputs()
    return asyncio.run(
        verify_fields(messages, "transfer_draft", proposal, fields, client, threshold)
    )


@pytest.mark.parametrize("key_change", ["missing", "extra"])
def test_verifier_requires_exactly_the_requested_answer_keys(key_change):
    with pytest.raises(VerificationError, match="неполный или неожиданный"):
        _verify(FakeJev(key_change=key_change))


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_verifier_rejects_probability_outside_zero_to_one(probability):
    with pytest.raises(ValidationError):
        _verify(FakeJev(probability=probability))


@pytest.mark.parametrize(
    ("probability", "has_feedback"),
    [(0.699999, False), (0.7, True), (0.700001, True)],
)
def test_wrong_threshold_includes_exact_boundary_and_values_above_it(probability, has_feedback):
    feedback = _verify(FakeJev(probability=probability))
    assert bool(feedback) is has_feedback


def test_jev_failure_is_propagated_without_being_replaced():
    error = RuntimeError("Jev unavailable")
    with pytest.raises(RuntimeError) as raised:
        _verify(FakeJev(error=error))
    assert raised.value is error


def test_invoice_price_ambiguity_rule_does_not_apply_to_transfer_amount():
    from src.field_rules import INVOICE_AMOUNT_RULE

    _, _, fields = _inputs()
    questions = verification.build_verification_questions(fields).root
    assert "amount_rub::ambiguous" not in questions
    assert all(INVOICE_AMOUNT_RULE not in q.model_dump_json() for q in questions.values())


@pytest.mark.parametrize("items_present", [False, True])
def test_feedback_contains_specific_check_without_general_or_item_rules(monkeypatch, items_present):
    monkeypatch.setattr(verification, "_RULES", "GENERAL_RULES_FOR_JEV_ONLY")
    monkeypatch.setattr(verification, "_ITEMS_RULES", "ITEM_RULES_FOR_JEV_ONLY")
    messages = [DocumentMessage(id="m1", role="user", text="Счёт за консультацию")]
    raw_fields = {
        name: {"status": "missing", "value": None, "evidence": []}
        for name in InvoiceFields.model_fields
    }
    if items_present:
        raw_fields["items"] = {
            "status": "fillable",
            "value": [{"description": "консультация", "quantity": None,
                       "unit": None, "unit_price_rub": None}],
            "evidence": [{"message_id": "m1", "quote": "консультацию"}],
        }
    proposal = ExtractionProposal.model_validate({"fields": raw_fields})
    fields = prepare_fields(proposal, messages, "invoice_draft")
    target = "items::unsupported" if items_present else "items::status_wrong"
    captured = []

    class SelectiveJev:
        async def ainvoke(self, state, questions, response_model):
            captured.append((state, questions.root[target]))
            return response_model.model_validate({"answers": {
                key: {"noul": 0.89 if key == target else 0.01}
                for key in questions.root
            }})

    feedback = asyncio.run(verify_fields(messages, "invoice_draft", proposal, fields, SelectiveJev()))
    assert "GENERAL_RULES_FOR_JEV_ONLY" in captured[0][0].rules
    assert "ITEM_RULES_FOR_JEV_ONLY" in captured[0][1].instructions["rules"]
    assert len(feedback) == 1
    assert feedback[0].startswith(target)
    assert "0.89" in feedback[0]
    assert "`items`" in feedback[0]
    assert "GENERAL_RULES_FOR_JEV_ONLY" not in feedback[0]
    assert "ITEM_RULES_FOR_JEV_ONLY" not in feedback[0]
    assert ("unsupported by its cited user messages" if items_present else "Is the status") in feedback[0]
