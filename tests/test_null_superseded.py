"""Проверки актуального null-статуса и маршрутизация заданных вручную ответов модели.

Ответы Jev в тестах заданы явно: качество реальной модели они не измеряют.
"""

import asyncio
import json

import httpx2
import pytest

from scripts.measure_request_sizes import jev_body
from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionProposal, prepare_fields
from src.llm import AsyncLLMExtractor
from src.pipeline import DocumentTypeDecision, PipelineRejectionError, process_selected_document
from src.schemas.document_schemas import InvoiceFields, TransferFields
from src.settings import LLMSettings
from src.verification import VerificationResponse, VerificationState, _RULES, build_verification_questions


def inputs(status="missing", *, replacement=False, value=None, evidence="history"):
    messages = [
        DocumentMessage(id="m1", role="user", text="Переведи себе 6000 рублей, банк ВТБ, карта ****7777"),
        DocumentMessage(id="m2", role="assistant", text="Карта записана"),
        DocumentMessage(id="m3", role="user", text=(
            "Замени карту на ****8888, банк и сумму оставь" if replacement else
            "Карту ****7777 удали, новую пока не назову. Сам перевод не отменяю"
        )),
    ]
    fields = {name: {"status": "missing", "value": None, "evidence": []}
              for name in TransferFields.model_fields}
    fields["amount_rub"] = {"status": "fillable", "value": 6000,
                            "evidence": [{"message_id": "m1", "quote": "6000 рублей"}]}
    fields["destination_bank"] = {"status": "fillable", "value": "ВТБ",
                                  "evidence": [{"message_id": "m1", "quote": "банк ВТБ"}]}
    sources = [] if evidence == "empty" else [
        {"message_id": "m1", "quote": "карта ****7777"},
    ]
    if evidence == "history":
        sources.append({"message_id": "m3", "quote": messages[-1].text})
    if value == "****8888":
        sources = [{"message_id": "m3", "quote": "****8888"}]
    fields["destination_card_masked"] = {"status": status, "value": value, "evidence": sources}
    proposal = ExtractionProposal.model_validate({"fields": fields})
    return messages, proposal, prepare_fields(proposal, messages, "transfer_draft")


@pytest.mark.parametrize("evidence", ["empty", "old", "history"])
@pytest.mark.parametrize("replacement", [False, True])
def test_wire_missing_card_checks_omitted_replacement_not_source_history(evidence, replacement):
    messages, proposal, fields = inputs(evidence=evidence, replacement=replacement)
    questions = build_verification_questions(fields, proposal=proposal, messages=messages)
    payload = json.loads(jev_body(
        VerificationState(messages=messages, document_type="transfer_draft", rules=_RULES),
        questions, VerificationResponse,
    ))
    assert "destination_card_masked::superseded" not in payload["questions"]
    q = payload["questions"]["destination_card_masked::missing_supported"]
    assert q["instructions"]["field"]["status"] == "missing"
    assert q["instructions"]["field"]["value"] is None
    assert q["instructions"]["field"]["evidence"] == proposal.fields["destination_card_masked"].model_dump(mode="json")["evidence"]
    assert payload["state"]["messages"] == [m.model_dump(mode="json") for m in messages]


def assert_current_null_status_contract(question):
    text = question.instructions["question"]
    assert "instructions.field.status" in text
    assert "latest explicit user correction for this field" in text
    assert "Withdrawal without replacement requires missing/null" in text
    assert "still justify invalid/null or ambiguous/null, answer false" in text
    assert "current null status" in question.criteria["true"]
    assert "historical source change alone is not an error" in question.criteria["false"]


@pytest.mark.parametrize("status", ["invalid", "ambiguous"])
@pytest.mark.parametrize("replacement", [False, True])
def test_null_status_checks_current_candidate_against_correction(status, replacement):
    messages, proposal, fields = inputs(status, replacement=replacement)
    questions = build_verification_questions(fields, proposal=proposal, messages=messages).root
    q = questions["destination_card_masked::superseded"]
    assert_current_null_status_contract(q)
    assert q.instructions["field"]["status"] == status
    assert "destination_card_masked::missing_supported" in questions


@pytest.mark.parametrize("evidence", ["old", "history"])
def test_withdrawn_card_does_not_trigger_repair_for_changed_source_history(evidence):
    messages, initial, _ = inputs(evidence=evidence)

    class HistoryVerifier:
        async def ainvoke(self, state, questions, response_model):
            # A correct answer to the old history question was high. The fixed
            # request must not ask it for missing/null, even with old evidence.
            return response_model.model_validate({"answers": {
                key: {"noul": .95 if key == "destination_card_masked::superseded" else .01}
                for key in questions.root
            }})

    class NoRepair:
        async def extract(self, *args, **kwargs):
            pytest.fail("Withdrawal already reflected as missing/null must not cause repair")

    result = asyncio.run(process_selected_document(
        "fixture", messages, DocumentTypeDecision(document_type="transfer_draft", confidence=1),
        HistoryVerifier(), NoRepair(), initial_proposal=initial,
    ))
    assert result.fields.destination_card_masked.status == "missing"
    assert result.fields.destination_card_masked.value is None
    assert result.fields.amount_rub.value == 6000
    assert result.fields.destination_bank.value == "ВТБ"


@pytest.mark.parametrize("initial_status,initial_value,replacement,expected_value,repair_check", [
    ("missing", None, False, None, None),
    ("missing", None, True, "****8888", "missing_supported"),
    ("fillable", "****7777", False, None, "superseded"),
    ("fillable", "****7777", True, "****8888", "superseded"),
    ("fillable", "****8888", True, "****8888", None),
    ("invalid", None, False, None, "superseded"),
    ("ambiguous", None, False, None, "superseded"),
])
def test_mock_temporal_verdict_repairs_only_an_incorrect_final_card_state(
    initial_status, initial_value, replacement, expected_value, repair_check,
):
    messages, initial, _ = inputs(initial_status, replacement=replacement, value=initial_value)
    requests, rounds = [], []

    def respond(request):
        user = json.loads(json.loads(request.content)["messages"][-1]["content"])
        requests.append(user)
        card = {"status": "fillable" if expected_value else "missing", "value": expected_value,
                "evidence": [{"message_id": "m3", "quote": "****8888"}] if expected_value else [],
                "calculation": None}
        return httpx2.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"fields": {"destination_card_masked": card}}),
        }}]})

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            rounds.append(questions)
            q = questions.root.get("destination_card_masked::missing_supported")
            if q is not None and q.instructions["field"]["status"] == "missing":
                assert "destination_card_masked::superseded" not in questions.root
            q = questions.root.get("destination_card_masked::superseded")
            if q is not None and q.instructions["field"]["status"] != "fillable":
                assert_current_null_status_contract(q)
            # Explicit mock verdicts test routing and repair, not the live model's accuracy.
            target = "destination_card_masked::" + repair_check if repair_check and len(rounds) == 1 else None
            return response_model.model_validate({"answers": {
                key: {"noul": .95 if key == target else .01} for key in questions.root
            }})

    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                return await process_selected_document(
                    "fixture", messages, DocumentTypeDecision(document_type="transfer_draft", confidence=1),
                    Verifier(), extractor, initial_proposal=initial,
                )

    result = asyncio.run(run())
    assert result.fields.destination_card_masked.value == expected_value
    assert result.fields.destination_card_masked.status == ("fillable" if expected_value else "missing")
    assert result.fields.amount_rub.value == 6000
    assert result.fields.destination_bank.value == "ВТБ"
    assert len(requests) == int(repair_check is not None)
    assert len(rounds) == len(requests) + 1
    if requests:
        assert requests[0]["repair_paths"] == ["destination_card_masked"]
        assert requests[0]["previous_checked_fields"]["destination_card_masked"]["status"] == initial_status
        assert "destination_card_masked::" + repair_check in requests[0]["feedback"][0]


@pytest.mark.parametrize("status,original,correction", [
    ("invalid", "БИК 04452546", "Исправь БИК на 12345678, цифры не дополняй"),
    ("ambiguous", "БИК 044525225 или 044525226, пока не выбирай",
     "Вместо них БИК 044525227 или 044525228, всё ещё не выбрал"),
])
def test_low_mock_verdict_preserves_null_when_latest_data_still_justify_status(status, original, correction):
    messages = [DocumentMessage(id="m1", role="user", text=original),
                DocumentMessage(id="m2", role="user", text=correction)]
    fields = {name: {"status": "missing", "value": None, "evidence": []}
              for name in InvoiceFields.model_fields}
    fields["recipient_bik"] = {"status": status, "value": None, "evidence": [
        {"message_id": m.id, "quote": m.text} for m in messages
    ]}
    proposal = ExtractionProposal.model_validate({"fields": fields})

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            q = questions.root["recipient_bik::superseded"]
            assert q.instructions["field"]["status"] == status
            assert_current_null_status_contract(q)
            # Explicit mock verdict for a still-justified invalid/ambiguous status.
            return response_model.model_validate({"answers": {
                key: {"noul": .01} for key in questions.root
            }})

    class NoRepair:
        async def extract(self, *args, **kwargs):
            pytest.fail("A justified final status must not require repair")

    result = asyncio.run(process_selected_document(
        "fixture", messages, DocumentTypeDecision(document_type="invoice_draft", confidence=1),
        Verifier(), NoRepair(), initial_proposal=proposal,
    ))
    assert result.fields.recipient_bik.status == status
    assert result.fields.recipient_bik.value is None


@pytest.mark.parametrize("status", ["invalid", "ambiguous"])
def test_high_null_status_superseded_score_is_not_bypassed(status):
    messages, initial, _ = inputs(status)

    class RejectingVerifier:
        async def ainvoke(self, state, questions, response_model):
            assert_current_null_status_contract(questions.root["destination_card_masked::superseded"])
            return response_model.model_validate({"answers": {
                key: {"noul": .99 if key == "destination_card_masked::superseded" else .01}
                for key in questions.root
            }})

    with pytest.raises(PipelineRejectionError, match="destination_card_masked::superseded"):
        asyncio.run(process_selected_document(
            "fixture", messages, DocumentTypeDecision(document_type="transfer_draft", confidence=1),
            RejectingVerifier(), None, initial_proposal=initial, max_attempts=1,
            allow_partial=False,
        ))
