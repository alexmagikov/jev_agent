"""Контекст проверки принадлежности цитаты; оценки Jev здесь заданы вручную."""

import asyncio
import json
from pathlib import Path

import pytest

from scripts.measure_request_sizes import jev_body
from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionProposal, prepare_fields
from src.pipeline import DocumentTypeDecision, PipelineRejectionError, process_selected_document
from src.schemas.document_schemas import InvoiceFields
from src.verification import VerificationResponse, VerificationState, _RULES, build_verification_questions


def t003(*, cite_seller=False):
    data = Path(__file__).resolve().parents[1] / "data/cases_tricky35.jsonl"
    case = next(row for line in data.read_text().splitlines()
                if (row := json.loads(line))["case_id"] == "T003")
    messages = [DocumentMessage.model_validate(message) for message in case["messages"]]
    fields = {name: {"status": "missing", "value": None, "evidence": []}
              for name in InvoiceFields.model_fields}
    fields["recipient_name"] = {"status": "fillable", "value": "ООО Формат Плюс", "evidence": [
        {"message_id": "m1", "quote": "ООО Формат Плюс"},
    ]}
    sources = [{"message_id": "m3", "quote": "Письмо продавца с ИНН 7704100998"}] if cite_seller else [
        {"message_id": "m1", "quote": "ИНН 77041001011"},
        {"message_id": "m3", "quote": "ИНН пока тот же, верного другого номера у меня нет"},
    ]
    fields["recipient_inn"] = {"status": "invalid", "value": None, "evidence": sources}
    proposal = ExtractionProposal.model_validate({"fields": fields})
    return messages, proposal, prepare_fields(proposal, messages, "invoice_draft")


def test_t003_wire_keeps_cited_buyer_inn_separate_from_uncited_seller_candidate():
    messages, proposal, fields = t003()
    questions = build_verification_questions(fields, proposal=proposal, messages=messages)
    payload = json.loads(jev_body(
        VerificationState(messages=messages, document_type="invoice_draft", rules=_RULES),
        questions, VerificationResponse,
    ))
    role = payload["questions"]["recipient_inn::wrong_role"]
    claim = role["instructions"]["field"]
    assert claim["status"] == "invalid" and claim["value"] is None
    assert claim["evidence"] == proposal.fields["recipient_inn"].model_dump(mode="json")["evidence"]
    assert "valid_source_candidates" not in claim
    assert "instructions.field.evidence" in role["instructions"]["question"]
    assert "elsewhere in the dialogue" in role["instructions"]["question"]
    assert "instructions.field.evidence" in role["criteria"]["true"]
    assert "unrelated details elsewhere" in role["criteria"]["false"]
    omitted = payload["questions"]["recipient_inn::missing_supported"]
    assert omitted["instructions"]["field"]["valid_source_candidates"] == [
        {"message_id": "m3", "quote": "7704100998"},
    ]
    assert "valid_source_candidates" not in payload["questions"]["recipient_inn::superseded"]["instructions"]["field"]
    assert payload["state"]["messages"] == [m.model_dump(mode="json") for m in messages]


@pytest.mark.parametrize("status", ["missing", "invalid", "ambiguous"])
def test_format_candidates_are_available_only_to_omission_and_alternative_checks(status):
    messages, proposal, _ = t003()
    messages.append(DocumentMessage(id="m8", role="user", text=(
        "У другого продавца ИНН 7704100888. Это ещё один чужой реквизит, не исправление ИНН покупателя."
    )))
    proposal.fields["recipient_inn"].status = status
    fields = prepare_fields(proposal, messages, "invoice_draft")
    questions = build_verification_questions(fields, proposal=proposal, messages=messages).root
    expected = [{"message_id": "m3", "quote": "7704100998"}, {"message_id": "m8", "quote": "7704100888"}]
    for key, question in questions.items():
        claim = question.instructions["field"]
        if key in ("recipient_inn::missing_supported", "recipient_inn::unresolved_alternatives"):
            assert claim["valid_source_candidates"] == expected
        elif not key.endswith(("::missing_supported", "::unresolved_alternatives")):
            assert "valid_source_candidates" not in claim
    assert "recipient_inn::wrong_role" in questions
    assert ("recipient_inn::unresolved_alternatives" in questions) == (status != "ambiguous")


@pytest.mark.parametrize("cite_seller", [False, True])
def test_mock_role_verdict_preserves_buyer_invalid_but_still_rejects_cited_seller(cite_seller):
    messages, proposal, _ = t003(cite_seller=cite_seller)

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            claim = questions.root["recipient_inn::wrong_role"].instructions["field"]
            assert "valid_source_candidates" not in claim
            assert claim["evidence"] == proposal.fields["recipient_inn"].model_dump(mode="json")["evidence"]
            # Fixed verdicts verify routing; they do not predict live Jev scores.
            return response_model.model_validate({"answers": {
                key: {"noul": .95 if cite_seller and key == "recipient_inn::wrong_role" else .01}
                for key in questions.root
            }})

    async def run():
        return await process_selected_document(
            "T003", messages, DocumentTypeDecision(document_type="invoice_draft", confidence=1),
            Verifier(), None, initial_proposal=proposal, max_attempts=1,
            allow_partial=False,
        )

    if cite_seller:
        with pytest.raises(PipelineRejectionError, match="recipient_inn::wrong_role"):
            asyncio.run(run())
    else:
        result = asyncio.run(run())
        assert result.fields.recipient_inn.status == "invalid"
        assert result.fields.recipient_inn.value is None
        assert result.fields.recipient_name.value == "ООО Формат Плюс"
