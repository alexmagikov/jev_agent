"""Контракт проверки актуальности кандидата, а не наличия исправлений в истории.

Ответы верификатора в тестах заданы явно: качество реального Jev они не измеряют.
"""

import asyncio
import json
from pathlib import Path

import httpx2
import pytest

from scripts.measure_request_sizes import jev_body
from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionProposal, prepare_fields
from src.llm import AsyncLLMExtractor
from src.pipeline import DocumentTypeDecision, PipelineRejectionError, process_selected_document
from src.schemas.document_schemas import InvoiceFields, TransferFields
from src.settings import LLMSettings
from src.verification import (
    VerificationResponse, VerificationState, _RULES, build_verification_questions,
)


def inputs(value, quote=None, *, document_type="invoice_draft"):
    if document_type == "invoice_draft":
        path = Path(__file__).resolve().parents[1] / "data/cases_v2.jsonl"
        case = next(row for line in path.read_text().splitlines()
                    if (row := json.loads(line))["case_id"] == "C004")
        messages = [DocumentMessage.model_validate(m) for m in case["messages"]]
        model = InvoiceFields
    else:
        messages = [
            DocumentMessage(id="m1", role="user", text="Переведи себе 3000 рублей"),
            DocumentMessage(id="m3", role="user", text="Исправь: не 3000, а 3500 рублей"),
        ]
        model = TransferFields
    fields = {name: {"status": "missing", "value": None, "evidence": []}
              for name in model.model_fields}
    fields["amount_rub"] = {
        "status": "fillable", "value": value,
        "evidence": [{"message_id": "m1" if value == 3000 else "m3",
                      "quote": quote or f"{value} рублей"}],
    }
    proposal = ExtractionProposal.model_validate({"fields": fields})
    return messages, proposal, prepare_fields(proposal, messages, document_type)


def assert_candidate_contract(question):
    text = question["instructions"]["question"]
    assert "instructions.field.value" in text
    assert "latest explicit user correction" in text
    assert "already reflects" in text and "false" in text
    assert "withdrawal" in text and "other fields" in text
    assert "different value" in question["criteria"]["true"]
    assert "already reflects" in question["criteria"]["false"]
    assert "historical correction alone" in question["criteria"]["false"]


@pytest.mark.parametrize("document_type", ["invoice_draft", "transfer_draft"])
@pytest.mark.parametrize("value,quote", [
    (3000, "3000 рублей"),
    (3500, "3500 рублей"),
    (3500, "не 3000, а 3500 рублей"),
])
def test_wire_superseded_question_anchors_the_candidate_not_any_historical_change(document_type, value, quote):
    messages, proposal, fields = inputs(value, quote, document_type=document_type)
    questions = build_verification_questions(
        fields, proposal=proposal, document_type=document_type, messages=messages,
    )
    state = VerificationState(messages=messages, document_type=document_type, rules=_RULES)
    payload = json.loads(jev_body(state, questions, VerificationResponse))
    question = payload["questions"]["amount_rub::superseded"]
    assert_candidate_contract(question)
    claim = question["instructions"]["field"]
    assert claim["value"] == value
    assert claim["evidence"] == proposal.fields["amount_rub"].model_dump(mode="json")["evidence"]
    assert payload["state"]["messages"] == [m.model_dump(mode="json") for m in messages]
    assert "labels" not in payload["state"]


@pytest.mark.parametrize("initial_value", [3000, 3500])
def test_mock_verdict_repairs_old_candidate_but_keeps_current_candidate_without_repair(initial_value):
    messages, initial, _ = inputs(initial_value)
    requests, rounds = [], []

    def respond(request):
        requests.append(json.loads(json.loads(request.content)["messages"][1]["content"]))
        return httpx2.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"fields": {"amount_rub": {
                "status": "fillable", "value": 3500,
                "evidence": [{"message_id": "m3", "quote": "3500 рублей"}],
                "calculation": None,
            }}}),
        }}]})

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            q = questions.root["amount_rub::superseded"]
            assert_candidate_contract(q.model_dump(mode="json"))
            value = q.instructions["field"]["value"]
            rounds.append(value)
            # Explicit mock verdicts, not a simulation of Jev's understanding.
            return response_model.model_validate({"answers": {
                key: {"noul": {3000: .95, 3500: .01}[value]
                      if key == "amount_rub::superseded" else .01}
                for key in questions.root
            }})

    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                return await process_selected_document(
                    "fixture", messages, DocumentTypeDecision(document_type="invoice_draft", confidence=1),
                    Verifier(), extractor, initial_proposal=initial,
                )

    result = asyncio.run(run())
    assert result.fields.amount_rub.value == 3500
    assert result.fields.amount_rub.evidence == ["m3"]
    assert result.fields.subject.status == "missing"
    if initial_value == 3000:
        assert rounds == [3000, 3500] and len(requests) == 1
        assert requests[0]["repair_fields"] == ["amount_rub"]
        assert requests[0]["repair_paths"] == ["amount_rub"]
        assert requests[0]["previous_checked_fields"]["amount_rub"]["value"] == 3000
        assert "latest explicit user correction" in requests[0]["feedback"][0]
    else:
        assert rounds == [3500] and requests == []


def test_high_superseded_score_is_not_bypassed_even_for_latest_candidate():
    messages, initial, _ = inputs(3500)

    class RejectingVerifier:
        async def ainvoke(self, state, questions, response_model):
            return response_model.model_validate({"answers": {
                key: {"noul": .99 if key == "amount_rub::superseded" else .01}
                for key in questions.root
            }})

    with pytest.raises(PipelineRejectionError, match="amount_rub::superseded"):
        asyncio.run(process_selected_document(
            "fixture", messages, DocumentTypeDecision(document_type="invoice_draft", confidence=1),
            RejectingVerifier(), None, initial_proposal=initial, max_attempts=1,
            allow_partial=False,
        ))
