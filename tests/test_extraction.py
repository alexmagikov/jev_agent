import asyncio
import json
from decimal import Decimal

import pytest
import httpx2

from src.cases.document_type import DocumentMessage
from src.schemas.document_schemas import InvoiceFields, TransferFields
from src.extraction import ExtractionError, ExtractionProposal, get_proposal_model, prepare_fields
from src.llm import AsyncLLMExtractor, LLMError
from src.settings import LLMSettings


def proposal(document_type="invoice_draft", **overrides):
    model = InvoiceFields if document_type == "invoice_draft" else TransferFields
    fields = {
        name: {"status": "missing", "value": None, "evidence": []}
        for name in model.model_fields
    }
    fields.update(overrides)
    return {"fields": fields}


def filled(value, quote, message_id="m1"):
    return {
        "status": "fillable", "value": value,
        "evidence": [{"message_id": message_id, "quote": quote}],
    }


def prepare(data, text, document_type="invoice_draft", role="user"):
    return prepare_fields(
        ExtractionProposal.model_validate(data),
        [DocumentMessage(id="m1", role=role, text=text)],
        document_type,
    )


def test_python_computes_total_from_sourced_operands():
    data = proposal(amount_rub=filled(None, "3 стула по 4000"))
    data["fields"]["amount_rub"]["calculation"] = {
        "expression": "quantity * price",
        "operands": [
            {"name": "quantity", "source": {"message_id": "m1", "quote": "3"}},
            {"name": "price", "source": {"message_id": "m1", "quote": "4000"}},
        ],
    }
    result = prepare(data, "3 стула по 4000")
    assert result.amount_rub.value == 12000
    assert result.amount_rub.evidence == ["m1"]
    assert "quantity * price" in result.amount_rub.derivation
    assert "4000" in result.amount_rub.derivation


def test_python_computes_unit_price_without_changing_total():
    data = proposal(
        amount_rub=filled(100, "100 рублей за 5 штук"),
        items=filled([{
            "description": "Абрикосы", "quantity": 5,
            "unit": "штук", "unit_price_rub": None,
        }], "Абрикосы: 100 рублей за 5 штук"),
    )
    data["fields"]["items"]["value"][0]["unit_price_calculation"] = {
        "expression": "total / quantity",
        "operands": [
            {"name": "total", "source": {"message_id": "m1", "quote": "100"}},
            {"name": "quantity", "source": {"message_id": "m1", "quote": "5"}},
        ],
    }
    result = prepare(data, "Абрикосы: 100 рублей за 5 штук")
    assert result.items.value[0].unit_price_rub == 20
    assert result.amount_rub.value == 100
    assert result.items.derivation is not None


@pytest.mark.parametrize("expression", [
    "__import__('os').system('id')", "a ** 1000000", "a / 0", "a + unknown", "a + 999",
])
def test_rejects_unsafe_or_unsupported_calculation(expression):
    data = proposal(amount_rub=filled(None, "10"))
    data["fields"]["amount_rub"]["calculation"] = {
        "expression": expression,
        "operands": [{"name": "a", "source": {"message_id": "m1", "quote": "10"}}],
    }
    with pytest.raises(ExtractionError):
        prepare(data, "Счёт на 10 рублей")


@pytest.mark.parametrize("quote,role", [("выдумано", "user"), ("100", "assistant")])
def test_invalid_evidence_is_extractor_error_not_missing(quote, role):
    with pytest.raises(ExtractionError):
        prepare(proposal(amount_rub=filled(100, quote)), "100 рублей", role=role)


def test_invalid_inn_is_invalid_data_with_evidence():
    result = prepare(proposal(recipient_inn=filled("12345", "ИНН 12345")), "ИНН 12345")
    assert result.recipient_inn.status == "invalid"
    assert result.recipient_inn.value is None
    assert result.recipient_inn.evidence == ["m1"]


def test_synthetic_inn_is_not_checked_against_registry_or_checksum():
    result = prepare(
        proposal(recipient_inn=filled("7701234567", "ИНН 7701234567")), "ИНН 7701234567"
    )
    assert result.recipient_inn.status == "fillable"


def test_card_cannot_be_invented_or_unmasked():
    with pytest.raises(ExtractionError):
        prepare(
            proposal("transfer_draft", destination_card_masked=filled("123456789012", "****9012")),
            "На ****9012", "transfer_draft",
        )


@pytest.mark.parametrize("rate", [0, 20])
def test_explicit_vat_rate_is_preserved_and_nulls_survive(rate):
    text = f"НДС {rate}%"
    data = proposal(
        vat_mode=filled("rate", text), vat_rate_percent=filled(rate, text),
    )
    get_proposal_model("invoice_draft").model_validate(data)
    result = prepare(data, text)
    assert result.vat_mode.value == "rate"
    assert result.vat_rate_percent.value == rate
    assert result.vat_rate_percent.derivation is None
    assert result.recipient_name.model_dump() == {
        "status": "missing", "value": None, "evidence": [], "derivation": None,
    }


@pytest.mark.parametrize("boundary", ["llm_schema", "python_preparation"])
def test_vat_rate_cannot_be_calculated_from_tax_and_base_amounts(boundary):
    from pydantic import ValidationError

    text = "Сумма без НДС 100 рублей, сумма НДС 20 рублей"
    data = proposal(vat_rate_percent=filled(None, text))
    data["fields"]["vat_rate_percent"]["calculation"] = {
        "expression": "tax / base * 100",
        "operands": [
            {"name": "tax", "source": {"message_id": "m1", "quote": "20"}},
            {"name": "base", "source": {"message_id": "m1", "quote": "100"}},
        ],
    }
    expected_error = ValidationError if boundary == "llm_schema" else ExtractionError
    with pytest.raises(expected_error, match="vat_rate_percent"):
        if boundary == "llm_schema":
            get_proposal_model("invoice_draft").model_validate(data)
        else:
            prepare(data, text)


def test_missing_template_field_is_not_silently_filled():
    data = proposal()
    del data["fields"]["recipient_name"]
    with pytest.raises(ExtractionError):
        prepare(data, "Сделай счёт")


def test_unsourced_numeric_value_is_rejected():
    with pytest.raises(ExtractionError):
        prepare(proposal(amount_rub=filled(500, "100 рублей")), "100 рублей")


def test_rubles_and_kopecks_are_normalized():
    result = prepare(
        proposal("transfer_draft", amount_rub=filled(1000.21, "1000 рублей 21 копейку")),
        "Переведи 1000 рублей 21 копейку", "transfer_draft",
    )
    assert Decimal(str(result.amount_rub.value)) == Decimal("1000.21")


def test_operand_cannot_be_a_truncated_number():
    data = proposal(amount_rub=filled(None, "1200 рублей"))
    data["fields"]["amount_rub"]["calculation"] = {
        "expression": "a",
        "operands": [{"name": "a", "source": {"message_id": "m1", "quote": "200"}}],
    }
    with pytest.raises(ExtractionError):
        prepare(data, "Счёт на 1200 рублей")


@pytest.mark.parametrize("base_url", [
    "http://example.com/v1", "https://user:password@example.com/v1",
    "https://example.com/v1?key=example", "file:///tmp/model",
])
def test_llm_url_rejects_unsafe_configuration(base_url):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        LLMSettings(_env_file=None, LLM_API_KEY="test", LLM_MODEL="small", LLM_BASE_URL=base_url)


def test_llm_sends_only_selected_template_and_parses_json():
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json={"choices": [{
            "finish_reason": "stop", "message": {"content": json.dumps(proposal("transfer_draft"))},
        }]})

    settings = LLMSettings(_env_file=None, LLM_API_KEY="test", LLM_MODEL="small-model")
    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                return await extractor.extract(
                    [DocumentMessage(id="m1", role="user", text="Переведи себе")], "transfer_draft"
                )

    result = asyncio.run(run())
    assert set(result.fields) == set(TransferFields.model_fields)
    request = requests[0]
    assert request["model"] == "small-model"
    assert request["response_format"] == {"type": "json_object"}
    user = json.loads(request["messages"][1]["content"])
    assert set(user["template"]["fields"]) == set(TransferFields.model_fields)
    assert "labels" not in user


@pytest.mark.parametrize("choice", [
    {"finish_reason": "length", "message": {"content": "{}"}},
    {"finish_reason": "stop", "message": {"refusal": "Refused", "content": None}},
    {"finish_reason": "stop", "message": {"content": "not json"}},
    {"finish_reason": "stop", "message": None},
])
def test_llm_failures_are_explicit(choice):
    settings = LLMSettings(_env_file=None, LLM_API_KEY="test", LLM_MODEL="small-model")
    async def run():
        transport = httpx2.MockTransport(lambda request: httpx2.Response(200, json={"choices": [choice]}))
        async with httpx2.AsyncClient(transport=transport) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                return await extractor.extract(
                    [DocumentMessage(id="m1", role="user", text="Счёт")], "invoice_draft"
                )

    with pytest.raises(LLMError):
        asyncio.run(run())
