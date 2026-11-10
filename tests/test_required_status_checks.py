"""Каждое поле получает проверку статуса, даже без значения и валидных кандидатов."""

import asyncio
import json

import httpx2
import pytest

from src.cases.document_type import DocumentMessage
from src.client import DecisionModelClient, decision_sdk_options
from src.extraction import ExtractionProposal, prepare_fields
from src.llm import AsyncLLMExtractor
from src.pipeline import DocumentTypeDecision, process_selected_document
from src.schemas.document_schemas import InvoiceFields, TransferFields
from src.schemas.extraction_schemas import ProposedField, SourceQuote
from src.settings import DecisionModelSettings, LLMSettings
from src.verification import build_verification_questions


def empty_proposal(model=InvoiceFields):
    return ExtractionProposal.model_validate({"fields": {
        name: {"status": "missing", "value": None, "evidence": []}
        for name in model.model_fields
    }})


@pytest.mark.parametrize("document_type,model,name,value", [
    ("invoice_draft", InvoiceFields, "recipient_inn", "7701234567"),
    ("transfer_draft", TransferFields, "destination_bank", "Банк"),
])
@pytest.mark.parametrize("status", ["fillable", "missing", "invalid", "ambiguous"])
def test_every_template_field_has_a_status_check_for_every_current_status(document_type, model, name, value, status):
    messages = [DocumentMessage(id="m1", role="user", text=value)]
    proposal = empty_proposal(model)
    field = proposal.fields[name]
    field.status, field.value = status, value if status == "fillable" else None
    if status != "missing":
        field.evidence = [SourceQuote(message_id="m1", quote=value)]
    prepared = prepare_fields(proposal, messages, document_type)
    questions = build_verification_questions(prepared, proposal=proposal, messages=messages).root
    assert {key for key in questions if key.endswith("::status_wrong")} == {
        f"{name}::status_wrong" for name in model.model_fields
    }
    check = questions[f"{name}::status_wrong"]
    assert check.instructions["field"]["status"] == status
    assert "Judge status" in check.instructions["question"]
    assert "same correct status" in check.criteria["false"]


def test_explicit_malformed_inn_cannot_remove_the_status_check_for_missing_field():
    messages = [DocumentMessage(id="m1", role="user", text="ИНН покупателя 12345678901")]
    proposal = empty_proposal()
    prepared = prepare_fields(proposal, messages, "invoice_draft")
    questions = build_verification_questions(prepared, proposal=proposal, messages=messages).root
    # Поиск пригодного значения по-прежнему не обязан предлагать некорректный ИНН.
    assert "recipient_inn::missing_supported" not in questions
    check = questions["recipient_inn::status_wrong"]
    assert check.instructions["field"]["status"] == "missing"
    assert check.instructions["field"]["evidence"] == []
    assert "malformed" in check.instructions["question"]
    assert "missing" in check.instructions["question"]
    assert "invalid" in check.instructions["question"]
    assert "valid_source_candidates" not in check.instructions["field"]


def test_status_check_uses_prepared_status_and_python_format_fact_for_bad_proposed_value():
    messages = [DocumentMessage(id="m1", role="user", text="ИНН покупателя 12345678901")]
    proposal = empty_proposal()
    proposal.fields["recipient_inn"] = ProposedField.model_validate({
        "status": "fillable", "value": "12345678901", "evidence": [{"message_id": "m1", "quote": "12345678901"}],
    })
    prepared = prepare_fields(proposal, messages, "invoice_draft")
    check = build_verification_questions(prepared, proposal=proposal, messages=messages).root["recipient_inn::status_wrong"]
    assert check.instructions["field"]["status"] == "invalid"
    assert check.instructions["field"]["value"] is None
    assert check.instructions["field"]["format_check"]["checked_by"] == "python"
    assert check.instructions["field"]["format_check"]["matches_format"] is False


def test_status_verdict_reaches_real_sdk_and_repairs_only_the_missing_inn(monkeypatch):
    import src.client as client_module

    messages = [DocumentMessage(id="m1", role="user", text="ИНН покупателя 12345678901")]
    initial = empty_proposal()
    checks, repairs = [], []

    def verdict(request):
        body = json.loads(request.content)
        checks.append(body)
        key = "recipient_inn::status_wrong"
        assert key in body["questions"]
        assert body["state"]["messages"] == [m.model_dump(mode="json") for m in messages]
        assert body["questions"][key]["instructions"]["field"]["status"] == ("missing" if len(checks) == 1 else "invalid")
        check = body["questions"][key]["instructions"]["field"]["format_check"]
        assert check == {"rule": "Exactly 10 or 12 ASCII digits."}
        return httpx2.Response(200, json={"model": "jev-fixture", "answers": {
            name: {"type": "noul", "noul": .95 if name == key and len(checks) == 1 else .01}
            for name in body["questions"]
        }, "usage": {"input_tokens": 10, "output_tokens": 10}})

    def extraction(request):
        body = json.loads(request.content)
        user = json.loads(body["messages"][-1]["content"])
        repairs.append(user)
        assert user["repair_fields"] == user["repair_paths"] == ["recipient_inn"]
        assert set(user["template"]["fields"]) == {"recipient_inn"}
        assert user["previous_checked_fields"]["recipient_inn"]["status"] == "missing"
        assert "recipient_inn::status_wrong" in user["feedback"][0]
        reply = {"fields": {"recipient_inn": {
            "status": "invalid", "value": None, "evidence": [{"message_id": "m1", "quote": "12345678901"}],
            "calculation": None,
        }}}
        return httpx2.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(reply)}}]})

    real_sdk = client_module.AsyncTypeSafeClient
    monkeypatch.setattr(client_module, "AsyncTypeSafeClient", lambda **kwargs: real_sdk(
        **kwargs, transport=httpx2.MockTransport(verdict),
    ))
    settings = DecisionModelSettings(_env_file=None, DECISION_API_KEY="fixture", DECISION_BASE_URL="https://fixture.invalid")
    options = decision_sdk_options(settings, max_retries=0)
    verifier = DecisionModelClient(options.pop("api_key"), sdk_options=options)

    async def run():
        llm = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture",
                          LLM_API_PROTOCOL="openai", LLM_RESPONSE_FORMAT="text")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(extraction)) as http:
            async with AsyncLLMExtractor(llm, client=http) as extractor:
                return await process_selected_document(
                    "status-regression", messages, DocumentTypeDecision(document_type="invoice_draft", confidence=1),
                    verifier, extractor, initial_proposal=initial,
                )
    result = asyncio.run(run())
    assert len(checks) == 2 and len(repairs) == 1
    assert result.fields.recipient_inn.status == "invalid"
    for name in InvoiceFields.model_fields:
        if name != "recipient_inn":
            assert getattr(result.fields, name).status == "missing"
