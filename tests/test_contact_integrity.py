"""Email и маска копируются целиком; пустые заполненные строки не являются результатом."""

import asyncio
import json

import httpx2
import pytest
from pydantic import ValidationError

from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionError, ExtractionProposal, prepare_fields
from src.llm import AsyncLLMExtractor
from src.pipeline import DocumentTypeDecision, process_selected_document
from src.schemas.document_schemas import FieldResult, InvoiceFields, TransferFields
from src.settings import LLMSettings


def candidate(name, value, text, quote=None):
    kind, model = ("invoice_draft", InvoiceFields) if name.startswith("recipient_") or name in ("subject", "basis_number") else ("transfer_draft", TransferFields)
    fields = {key: {"status": "missing", "value": None, "evidence": []} for key in model.model_fields}
    fields[name] = {"status": "fillable", "value": value,
                    "evidence": [{"message_id": "m1", "quote": quote if quote is not None else text}]}
    return kind, ExtractionProposal.model_validate({"fields": fields}), [DocumentMessage(id="m1", role="user", text=text)]


@pytest.mark.parametrize("name,text,value", [
    ("recipient_email", "Почта ivan@example.com", "van@example.com"),
    ("recipient_email", "Почта billing+archive@example.com", "archive@example.com"),
    ("recipient_email", "Почта ivan@example.com.ru", "ivan@example.com"),
    ("recipient_email", "Почта ivan@example.com", "ivan@example.co"),
    ("recipient_email", "Почта ivan@example.com", "ivan@example"),
    ("recipient_email", "Почта orders@company.example", "orders@"),
    ("destination_card_masked", "Карта 2202 **** **** 9090", "9090"),
    ("destination_card_masked", "Карта 2202 **** **** 9090", "**** **** 9090"),
    ("destination_card_masked", "Карта 2202 **** **** 9090", "2202"),
    ("destination_card_masked", "Карта ****9090", "***9090"),
    ("destination_card_masked", "Карта ****9090", "****909"),
    ("destination_card_masked", "Карта ****9090", "****"),
    ("destination_card_masked", "Карта ****0007", "0007"),
    ("destination_card_masked", "Карта XXXX-XXXX-XXXX-9090", "9090"),
    ("destination_card_masked", "Карта •••• 9090", "9090"),
    # После скрытой группы может идти несколько открытых групп по 4 цифры.
    ("destination_card_masked", "Карта 2202 **** 1234 9090", "2202 **** 1234"),
    ("destination_card_masked", "Карта **** 1234 5678 9090", "**** 1234 5678"),
    ("destination_card_masked", "Карта **** 1234 5678 9090", "**** 1234"),
])
@pytest.mark.parametrize("crop_quote", [False, True])
def test_truncation_is_rejected_even_if_quote_is_cropped(name, text, value, crop_quote):
    kind, proposal, messages = candidate(name, value, text, value if crop_quote else None)
    before = proposal.model_dump(mode="json")
    with pytest.raises(ExtractionError) as raised:
        prepare_fields(proposal, messages, kind)
    assert raised.value.code == "invalid_source"
    assert raised.value.repair_paths == (name,)
    assert name in str(raised.value) and value not in str(raised.value)
    assert proposal.model_dump(mode="json") == before


@pytest.mark.parametrize("name,text,value", [
    ("recipient_email", "Почта ivan@example.com", "ivan@example.com"),
    ("recipient_email", "Почта: ivan@example.com.", "ivan@example.com"),
    ("recipient_email", "Почта <ivan@example.com>, подтверждена.", "ivan@example.com"),
    ("recipient_email", 'Почта «ivan@example.com»', "ivan@example.com"),
    ("recipient_email", "Почта 'ivan@example.com'", "ivan@example.com"),
    ("recipient_email", "Почта o'brien+archive@example.co.uk", "o'brien+archive@example.co.uk"),
    ("recipient_email", "Почта имя@пример.рф", "имя@пример.рф"),
    ("destination_card_masked", "На ****9090 переведи 3000 рублей", "****9090"),
    ("destination_card_masked", "На ****9090 3000 рублей", "****9090"),
    ("destination_card_masked", "Карта 2202 **** **** 9090.", "2202 **** **** 9090"),
    ("destination_card_masked", "Карта 2202 12** **** 9090", "2202 12** **** 9090"),
    ("destination_card_masked", "Карта XXXX-XXXX-XXXX-9090", "XXXX-XXXX-XXXX-9090"),
    ("destination_card_masked", "Карта •••• 9090", "•••• 9090"),
    ("destination_card_masked", "Карта 2202\u00a0****\u00a0****\u00a00007", "2202\u00a0****\u00a0****\u00a00007"),
    ("destination_card_masked", "Выбрана карта 9090", "9090"),
    ("destination_card_masked", "Передана маска ****", "****"),
    ("destination_card_masked", "Карта 2202 **** 1234 9090", "2202 **** 1234 9090"),
    ("destination_card_masked", "Карта **** 1234 5678 9090", "**** 1234 5678 9090"),
    ("destination_card_masked", "На карту **** 1234, 5000 рублей", "**** 1234"),
    ("destination_card_masked", "Карта **** 1234 50000 рублей", "**** 1234"),
    ("destination_card_masked", "Карта **** 123", "**** 123"),
])
def test_whole_stated_contacts_are_preserved(name, text, value):
    kind, proposal, messages = candidate(name, value, text, value)
    result = prepare_fields(proposal, messages, kind)
    assert getattr(result, name).status == "fillable"
    assert getattr(result, name).value == value


@pytest.mark.parametrize("value", ["orders@", "orders@company", "@example.com", "orders@@example.com", "orders @example.com", "orders@."])
def test_whole_malformed_email_remains_invalid_without_guessing_a_replacement(value):
    kind, proposal, messages = candidate("recipient_email", value, "Почта: " + value, value)
    result = prepare_fields(proposal, messages, kind)
    assert result.recipient_email.status == "invalid"
    assert result.recipient_email.value is None


@pytest.mark.parametrize("name,value,text,quote", [
    ("recipient_email", "van@example.com", "Почта van@example.com. Старый адрес ivan@example.com", "Старый адрес ivan@example.com"),
    ("destination_card_masked", "9090", "Карта 9090. Старый шаблон 2202 **** **** 9090", "Старый шаблон 2202 **** **** 9090"),
])
def test_whole_contact_elsewhere_does_not_rescue_a_truncated_cited_contact(name, value, text, quote):
    kind, proposal, messages = candidate(name, value, text, quote)
    with pytest.raises(ExtractionError):
        prepare_fields(proposal, messages, kind)


@pytest.mark.parametrize("value", ["", " ", "\t\n\u00a0"])
def test_result_schema_rejects_blank_fillable_strings(value):
    with pytest.raises(ValidationError):
        FieldResult[str].model_validate({"status": "fillable", "value": value, "evidence": ["m1"], "derivation": None})


@pytest.mark.parametrize("name", ["recipient_name", "recipient_email", "subject", "basis_number", "destination_card_masked", "destination_bank"])
@pytest.mark.parametrize("value", ["", " \t"])
def test_blank_fillable_string_gets_a_precise_python_repair_path(name, value):
    kind, proposal, messages = candidate(name, value, "Известные данные пользователя")
    with pytest.raises(ExtractionError) as raised:
        prepare_fields(proposal, messages, kind)
    assert raised.value.code == "missing_value"
    assert raised.value.repair_paths == (name,)


@pytest.mark.parametrize("use_verifier", [False, True])
@pytest.mark.parametrize("name,bad,good,text", [
    ("recipient_email", "van@example.com", "ivan@example.com", "Почта ivan@example.com"),
    ("destination_card_masked", "9090", "2202 **** **** 9090", "Карта 2202 **** **** 9090"),
    ("destination_card_masked", "", "****9090", "Карта ****9090"),
])
def test_integrity_error_is_repaired_before_semantic_verification(use_verifier, name, bad, good, text):
    kind, initial, messages = candidate(name, bad, text)
    _, corrected, _ = candidate(name, good, text)
    requests, verified = [], []
    def respond(request):
        user = json.loads(json.loads(request.content)["messages"][-1]["content"])
        requests.append(user)
        assert user["repair_paths"] == user["repair_fields"] == [name]
        reply = {"fields": {name: corrected.fields[name].model_dump(mode="json")}}
        return httpx2.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(reply)}}]})
    class Verifier:
        async def ainvoke(self, *, state, questions, response_model):
            assert len(requests) == 1
            verified.append(questions)
            return response_model.model_validate({"answers": {key: {"noul": .01} for key in questions.root}})
    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture",
                               LLM_API_PROTOCOL="openai", LLM_RESPONSE_FORMAT="text")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http:
            async with AsyncLLMExtractor(settings, client=http) as extractor:
                return await process_selected_document(
                    "contact-integrity", messages, DocumentTypeDecision(document_type=kind, confidence=1),
                    Verifier() if use_verifier else None, extractor, initial_proposal=initial, use_verifier=use_verifier,
                )
    result = asyncio.run(run())
    assert getattr(result.fields, name).value == good
    assert len(requests) == 1 and len(verified) == int(use_verifier)
    assert result.fields == prepare_fields(corrected, messages, kind)
