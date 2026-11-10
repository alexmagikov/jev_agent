"""Общее правило даты передаётся в реальный запрос извлечения без догадок о годе.

Ответы LLM заданы вручную: тесты проверяют промпт и контракт, а не качество модели.
"""

import asyncio
import json

import httpx2
import pytest

from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionProposal, prepare_fields
from src.llm import AsyncLLMExtractor, build_extraction_prompt
from src.schemas.document_schemas import InvoiceFields
from src.settings import LLMSettings
import src.field_rules as rules


def test_basis_date_rule_is_explicit_and_shared_by_invoice_prompt():
    rule = rules.BASIS_DATE_RULE
    assert "YYYY-MM-DD" in rule
    assert "current year" in rule and "contract number" in rule
    assert "same contract or basis" in rule
    assert "another user message" in rule
    assert "12 мая" in rule
    assert "status='invalid', value=null" in rule
    assert "status='missing', value=null" in rule
    assert "verbatim" in rule
    assert build_extraction_prompt("invoice_draft").count(rule) == 1
    assert rule not in build_extraction_prompt("transfer_draft")


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("texts,status,value,evidence", [
    (["Выставь счёт по договору от 12 мая."], "invalid", None, [0]),
    (["Выставь счёт по договору от 12 мая 2026 года."], "fillable", "2026-05-12", [0]),
    (["Выставь счёт по договору от 12 мая.", "Год этого договора — 2026."],
     "fillable", "2026-05-12", [0, 1]),
    (["Выставь счёт по договору № 15."], "missing", None, []),
    (["Выставь счёт по договору № 15/2026 от 12 мая."], "invalid", None, [0]),
])
def test_date_rule_reaches_request_and_date_contract_is_preserved(protocol, texts, status, value, evidence):
    messages = [DocumentMessage(id=f"m{i + 1}", role="user", text=text) for i, text in enumerate(texts)]
    fields = {name: {"status": "missing", "value": None, "evidence": [], "calculation": None}
              for name in InvoiceFields.model_fields}
    fields["basis_date"] = {
        "status": status, "value": value, "calculation": None,
        "evidence": [{"message_id": messages[i].id, "quote": messages[i].text} for i in evidence],
    }
    captured = []

    def respond(request):
        body = json.loads(request.content)
        captured.append(body)
        content = json.dumps({"fields": fields}, ensure_ascii=False)
        return httpx2.Response(200, json=(
            {"content": [{"type": "text", "text": content}], "stop_reason": "end_turn"}
            if protocol == "anthropic" else
            {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        ))

    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture",
                               LLM_API_PROTOCOL=protocol, LLM_RESPONSE_FORMAT="text")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                return await extractor.extract(messages, "invoice_draft")

    proposal = asyncio.run(run())
    assert isinstance(proposal, ExtractionProposal)
    system = captured[0]["system"] if protocol == "anthropic" else captured[0]["messages"][0]["content"]
    assert system.count(rules.BASIS_DATE_RULE) == 1
    user = json.loads(captured[0]["messages"][-1]["content"])
    assert user["messages"] == [message.model_dump() for message in messages]
    checked = prepare_fields(proposal, messages, "invoice_draft")
    assert checked.basis_date.status == status
    assert checked.basis_date.value == value
    assert checked.basis_date.evidence == [messages[i].id for i in evidence]


@pytest.mark.parametrize("text,status,value", [
    ("По договору от 12 мая.", "invalid", None),
    ("По договору от 12 мая 2026 года.", "fillable", "2026-05-12"),
    ("По договору № 15.", "missing", None),
    ("По договору от 12 мая: либо 2026, либо 2027 год — уточню.", "ambiguous", None),
])
def test_same_date_rule_reaches_every_jev_date_question(text, status, value):
    from src.verification import build_verification_questions

    messages = [DocumentMessage(id="m1", role="user", text=text)]
    fields = {name: {"status": "missing", "value": None, "evidence": []}
              for name in InvoiceFields.model_fields}
    fields["basis_date"] = {
        "status": status, "value": value,
        "evidence": [{"message_id": "m1", "quote": text}] if status != "missing" else [],
    }
    proposal = ExtractionProposal.model_validate({"fields": fields})
    prepared = prepare_fields(proposal, messages, "invoice_draft")
    questions = build_verification_questions(prepared, proposal=proposal, messages=messages).root
    assert "basis_date::status_wrong" in questions
    for key, question in questions.items():
        if key.startswith("basis_date::"):
            assert question.instructions["rules"].count(rules.BASIS_DATE_RULE) == 1
            assert question.instructions["field"]["status"] == status
            assert question.instructions["field"]["value"] == value
        else:
            assert rules.BASIS_DATE_RULE not in question.instructions["rules"]
