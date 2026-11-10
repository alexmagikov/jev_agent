"""Ставка НДС: только явный процент; no_vat не означает пропущенные 0%."""

import asyncio
import json
from pathlib import Path

import httpx2
import pytest

from scripts.measure_request_sizes import jev_body
from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionProposal, prepare_fields
from src.field_rules import VAT_MODE_RULE, VAT_RATE_RULE
from src.llm import AsyncLLMExtractor
from src.pipeline import DocumentTypeDecision, process_selected_document
from src.schemas.document_schemas import InvoiceFields
from src.settings import LLMSettings
from src.verification import (
    VerificationResponse, VerificationState, _RULES, build_verification_questions,
)


def inputs(messages, *, mode="no_vat", status="missing", rate=None):
    fields = {name: {"status": "missing", "value": None, "evidence": []}
              for name in InvoiceFields.model_fields}
    source = {"message_id": messages[-1].id, "quote": messages[-1].text}
    fields["vat_mode"] = {"status": "fillable", "value": mode, "evidence": [source]}
    fields["vat_rate_percent"] = {
        "status": status, "value": rate,
        "evidence": [] if status == "missing" else [source],
    }
    proposal = ExtractionProposal.model_validate({"fields": fields})
    return proposal, prepare_fields(proposal, messages, "invoice_draft")


@pytest.mark.parametrize("status", ["missing", "ambiguous", "invalid"])
def test_c065_wire_omission_question_does_not_allow_a_computed_vat_rate(status):
    path = Path(__file__).resolve().parents[1] / "data/cases_v2.jsonl"
    row = next(row for line in path.read_text().splitlines()
               if (row := json.loads(line))["case_id"] == "C065")
    messages = [DocumentMessage.model_validate(m) for m in row["messages"]]
    proposal, fields = inputs(messages, status=status)
    questions = build_verification_questions(fields, proposal=proposal, messages=messages)
    state = VerificationState(messages=messages, document_type="invoice_draft", rules=_RULES)
    payload = json.loads(jev_body(state, questions, VerificationResponse))
    question = payload["questions"]["vat_rate_percent::missing_supported"]
    text = question["instructions"]["question"]
    assert "explicitly stated" in text and "percentage" in text
    assert "computable" not in text
    assert "0%" in text and "без НДС" in text
    assert VAT_RATE_RULE in question["instructions"]["rules"]
    assert VAT_MODE_RULE in question["instructions"]["rules"]
    assert "explicit" in question["criteria"]["true"]
    assert "explicit" in question["criteria"]["false"]
    assert "no_vat" in question["criteria"]["false"]
    assert payload["state"]["messages"] == [m.model_dump(mode="json") for m in messages]


@pytest.mark.parametrize("text,mode,rate", [
    ("Счёт 1000 рублей, без НДС", "no_vat", None),
    ("Счёт 1000 рублей, НДС 0%", "rate", None),
    ("Счёт 1000 рублей, НДС 20%", "rate", None),
    ("Счёт 1200 рублей, в том числе НДС 200 рублей", "rate", None),
    ("Счёт 1000 рублей, НДС 0%", "rate", 0),
])
def test_rate_question_keeps_explicit_zero_and_never_imputes_a_rate(text, mode, rate):
    messages = [DocumentMessage(id="m1", role="user", text=text)]
    proposal, fields = inputs(messages, mode=mode, status="fillable" if rate is not None else "missing", rate=rate)
    questions = build_verification_questions(fields, proposal=proposal, messages=messages).root
    if rate is not None:
        assert fields.vat_rate_percent.value == 0
        assert "vat_rate_percent::missing_supported" not in questions
        assert questions["vat_rate_percent::unsupported"].instructions["field"]["value"] == 0
    else:
        assert fields.vat_rate_percent.value is None
        q = questions["vat_rate_percent::missing_supported"]
        assert "explicitly stated" in q.instructions["question"]
        assert "do not infer" in q.instructions["question"].lower()


def test_a_real_repair_can_restore_an_explicit_zero_rate_without_changing_vat_mode():
    messages = [DocumentMessage(id="m1", role="user", text="Подготовь счёт 1000 рублей, НДС 0%")]
    proposal, _ = inputs(messages, mode="rate")
    requests, rounds = [], []

    def respond(request):
        requests.append(json.loads(json.loads(request.content)["messages"][1]["content"]))
        return httpx2.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"fields": {"vat_rate_percent": {
                "status": "fillable", "value": 0, "evidence": [{"message_id": "m1", "quote": "НДС 0%"}],
                "calculation": None,
            }}}),
        }}]})

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            rounds.append(questions)
            return response_model.model_validate({"answers": {
                key: {"noul": .95 if key == "vat_rate_percent::missing_supported" else .01}
                for key in questions.root
            }})

    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                return await process_selected_document(
                    "fixture", messages, DocumentTypeDecision(document_type="invoice_draft", confidence=1),
                    Verifier(), extractor, initial_proposal=proposal,
                )

    result = asyncio.run(run())
    assert result.fields.vat_mode.value == "rate"
    assert result.fields.vat_rate_percent.value == 0
    assert len(requests) == 1 and len(rounds) == 2
    assert requests[0]["repair_fields"] == ["vat_rate_percent"]
    assert requests[0]["previous_checked_fields"]["vat_rate_percent"]["status"] == "missing"
