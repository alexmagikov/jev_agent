"""Эквивалентное написание не заменяет подтверждение значения и выбор пользователя.

Ответы верификатора — заданные фикстуры; это не оценка качества реального Jev.
"""

import asyncio
import json

import httpx2
import pytest

from scripts.measure_request_sizes import jev_body
from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionProposal, prepare_fields
from src.field_rules import TRANSFER_ALIAS_RULES
from src.llm import AsyncLLMExtractor, build_extraction_prompt
from src.pipeline import DocumentTypeDecision, process_selected_document
from src.schemas.document_schemas import TransferFields
from src.settings import LLMSettings
from src.verification import VerificationResponse, VerificationState, _RULES, build_verification_questions


def candidate(name, text, value, *, later=None, status="fillable", evidence_id="m1"):
    messages = [DocumentMessage(id="m1", role="user", text="Переведи себе 100 рублей на мою карту ****0007. " + text)]
    if later:
        messages.append(DocumentMessage(id="m3", role="user", text=later))
    fields = {key: {"status": "missing", "value": None, "evidence": []} for key in TransferFields.model_fields}
    for key, field_value, quote in (("amount_rub", 100, "100 рублей"), ("destination_card_masked", "****0007", "****0007"),
                                    ("recipient_relation", "self", "себе")):
        fields[key] = {"status": "fillable", "value": field_value, "evidence": [{"message_id": "m1", "quote": quote}]}
    source = next(message for message in messages if message.id == evidence_id)
    fields[name] = {"status": status, "value": value,
                    "evidence": [{"message_id": evidence_id, "quote": source.text}] if status != "missing" else []}
    return messages, ExtractionProposal.model_validate({"fields": fields})


@pytest.mark.parametrize("name", ["destination_bank", "source_account_kind"])
def test_wire_rule_is_shared_and_conditional_on_current_supported_value(name):
    messages, initial = candidate(name, "Банк ВТБ; счёт накопительный.", "Сбербанк" if name == "destination_bank" else "расчётный")
    rule = TRANSFER_ALIAS_RULES[name]
    assert "only if" in rule and "same bank or account type" in rule
    assert "current applicable user messages" in rule
    assert "source support" in rule and "ownership" in rule and "later corrections" in rule
    assert "withdrawals" in rule and "unresolved alternatives" in rule
    assert "not evidence or a whitelist" in rule
    assert "Do not flag a permitted form as unsupported or incorrect" not in rule
    assert build_extraction_prompt("transfer_draft").count(rule) == 1
    fields = prepare_fields(initial, messages, "transfer_draft")
    questions = build_verification_questions(fields, proposal=initial, messages=messages)
    payload = json.loads(jev_body(VerificationState(messages=messages, document_type="transfer_draft", rules=_RULES), questions, VerificationResponse))
    for kind in ("status_wrong", "unsupported", "wrong_role", "superseded", "unresolved_alternatives"):
        question = payload["questions"][f"{name}::{kind}"]
        assert question["instructions"]["rules"].count(rule) == 1
        assert question["instructions"]["field"]["value"] == initial.fields[name].value
    assert payload["state"]["messages"] == [message.model_dump(mode="json") for message in messages]


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("name,text,initial_value,later,status,value,check,evidence_id", [
    ("destination_bank", "Банк получателя Сбера.", "Сбербанк", None, "fillable", "Сбербанк", None, "m1"),
    ("destination_bank", "Банк получателя ВТБ.", "Сбербанк", None, "fillable", "ВТБ", "unsupported", "m1"),
    ("destination_bank", "Банк получателя Сбер.", "Сбербанк", "Исправление: банк получателя ВТБ, прежний банк не используй.",
     "fillable", "ВТБ", "superseded", "m3"),
    ("destination_bank", "Банк получателя Сбер.", "Сбербанк", "Убери банк, оставь поле пустым до уточнения.",
     "missing", None, "superseded", "m3"),
    ("destination_bank", "Либо Сбер, либо ВТБ — ещё не выбрал банк получателя.", "Сбербанк", None,
     "ambiguous", None, "unresolved_alternatives", "m1"),
    ("source_account_kind", "Счёт списания расчётный счёт.", "расчётный", None, "fillable", "расчётный", None, "m1"),
    ("source_account_kind", "Счёт списания накопительный.", "расчётный", None, "fillable", "накопительный", "unsupported", "m1"),
])
def test_mock_verdict_preserves_spelling_alias_but_repairs_wrong_or_superseded_choice(
    protocol, name, text, initial_value, later, status, value, check, evidence_id,
):
    messages, initial = candidate(name, text, initial_value, later=later)
    _, corrected = candidate(name, text, value, later=later, status=status, evidence_id=evidence_id)
    before = initial.model_dump(mode="json")
    calls = []

    def respond(request):
        body = json.loads(request.content)
        system = body["system"] if protocol == "anthropic" else body["messages"][0]["content"]
        assert TRANSFER_ALIAS_RULES[name] in system
        user = json.loads(body["messages"][-1]["content"])
        calls.append(user)
        assert user["repair_fields"] == user["repair_paths"] == [name]
        assert f"{name}::{check}" in user["feedback"][0]
        text = json.dumps({"fields": {name: corrected.fields[name].model_dump(mode="json")}})
        if protocol == "anthropic":
            return httpx2.Response(200, json={"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"})
        return httpx2.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": text}}]})

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            claim = questions.root[f"{name}::status_wrong"].instructions["field"]
            assert TRANSFER_ALIAS_RULES[name] in questions.root[f"{name}::status_wrong"].instructions["rules"]
            wrong = (claim["status"], claim["value"]) != (status, value)
            return response_model.model_validate({"answers": {
                key: {"noul": .95 if check and wrong and key == f"{name}::{check}" else .01}
                for key in questions.root
            }})

    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture", LLM_API_PROTOCOL=protocol,
                               LLM_RESPONSE_FORMAT="text")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                return await process_selected_document(
                    "alias-scope", messages, DocumentTypeDecision(document_type="transfer_draft", confidence=1),
                    Verifier(), extractor, initial_proposal=initial,
                )

    result = asyncio.run(run())
    assert len(calls) == int(check is not None)
    assert result.fields == prepare_fields(corrected, messages, "transfer_draft")
    assert initial.model_dump(mode="json") == before


def test_destination_bank_nominative_rule_reaches_extractor_and_verifier_only_for_bank():
    from src.field_rules import DESTINATION_BANK_RULE

    assert "nominative" in DESTINATION_BANK_RULE and "'Альфа-Банк'" in DESTINATION_BANK_RULE
    assert build_extraction_prompt("transfer_draft").count(DESTINATION_BANK_RULE) == 1
    assert DESTINATION_BANK_RULE not in build_extraction_prompt("invoice_draft")
    messages, proposal = candidate("destination_bank", "Банк получателя — в Альфа-Банке.", "Альфа-Банк")
    fields = prepare_fields(proposal, messages, "transfer_draft")
    questions = build_verification_questions(fields, proposal=proposal, messages=messages).root
    for key, question in questions.items():
        rules = question.instructions["rules"]
        assert rules.count(DESTINATION_BANK_RULE) == (1 if key.startswith("destination_bank::") else 0), key
