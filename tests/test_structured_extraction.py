import asyncio
import json

import httpx2
import pytest
from pydantic import ValidationError

from src.cases.document_type import DocumentMessage
from src.schemas.document_schemas import InvoiceFields, TransferFields
from src.extraction import ExtractionError, ExtractionProposal, prepare_fields
from src.llm import AsyncLLMExtractor, LLMError, LLMResponseError
from src.settings import LLMSettings


def settings(**kwargs):
    return LLMSettings(
        _env_file=None, LLM_API_KEY="test", LLM_MODEL="small",
        LLM_BASE_URL="http://localhost:11434/v1", **kwargs,
    )


def proposal(document_type="invoice_draft"):
    model = InvoiceFields if document_type == "invoice_draft" else TransferFields
    return {"fields": {name: {"status": "missing", "value": None, "evidence": []}
                       for name in model.model_fields}}


def envelope(data):
    return {"choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps(data),
    }}]}


def extract(data, document_type="invoice_draft", **kwargs):
    async def run():
        transport = httpx2.MockTransport(lambda request: httpx2.Response(200, json=envelope(data)))
        async with httpx2.AsyncClient(transport=transport) as client:
            async with AsyncLLMExtractor(settings(**kwargs), client=client) as extractor:
                return await extractor.extract([], document_type)

    return asyncio.run(run())


@pytest.mark.parametrize("document_type", ["invoice_draft", "transfer_draft"])
def test_schema_is_document_specific_and_matches_prompt(document_type):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json=envelope(proposal(document_type)))

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings(
                LLM_RESPONSE_FORMAT="json_schema", LLM_TEMPERATURE=0,
            ), client=client) as extractor:
                result = await extractor.extract([], document_type)
                assert isinstance(result, ExtractionProposal)
                assert set(result.fields) == set(proposal(document_type)["fields"])

    asyncio.run(run())
    payload = requests[0]
    output = payload["response_format"]
    assert output["type"] == "json_schema"
    assert output["json_schema"]["strict"] is True
    schema = output["json_schema"]["schema"]
    assert schema == json.loads(payload["messages"][1]["content"])["output_schema"]
    fields = schema["$defs"][schema["properties"]["fields"]["$ref"].split("/")[-1]]
    assert fields["additionalProperties"] is False
    assert set(fields["required"]) == set(proposal(document_type)["fields"])
    assert payload["temperature"] == 0

    def check_objects(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                check_objects(value)
        elif isinstance(node, list):
            for value in node:
                check_objects(value)

    check_objects(schema)


@pytest.mark.parametrize("response_format,json_mode,expected", [
    (None, True, "json_object"), (None, False, None),
    ("json_object", False, "json_object"), ("text", True, None),
])
def test_legacy_modes_and_explicit_override(response_format, json_mode, expected):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json=envelope(proposal()))

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings(
                LLM_RESPONSE_FORMAT=response_format, LLM_JSON_MODE=json_mode,
            ), client=client) as extractor:
                await extractor.extract([], "invoice_draft")

    asyncio.run(run())
    assert requests[0].get("response_format") == ({"type": expected} if expected else None)
    assert "temperature" not in requests[0]


@pytest.mark.parametrize("change,path", [
    (lambda fields: fields.update(items=[]), "fields.items"),
    (lambda fields: fields["items"].update(status="ammable"), "fields.items.status"),
    (lambda fields: fields.pop("subject"), "fields.subject"),
    (lambda fields: fields["recipient_inn"].update(value=123), "fields.recipient_inn.value"),
])
def test_invalid_output_gets_specific_safe_feedback(change, path):
    data = proposal()
    change(data["fields"])
    with pytest.raises(LLMResponseError) as error:
        extract(data)
    assert path in str(error.value)


def test_formula_feedback_covers_operand_paths_without_echoing_input():
    data = proposal()
    data["fields"]["amount_rub"]["calculation"] = {
        "expression": "Quantity * UnitPriceRub",
        "operands": [{"name": "Quantity", "source": {"message_id": ""}},
                     {"name": "UnitPriceRub", "source": {"message_id": ""}}],
    }
    # Неизвестные ключи и значения модели не должны попадать в текст ошибки.
    marker = "PRIVATE_INPUT_MARKER"
    data["fields"][marker] = {"status": marker, "value": marker, "evidence": []}
    with pytest.raises(LLMResponseError) as error:
        extract(data)
    feedback = str(error.value)
    assert "fields.amount_rub.calculation.operands.0.name" in feedback
    assert "fields.amount_rub.calculation.operands.0.source.message_id" in feedback
    assert "fields.amount_rub.calculation.operands.0.source.quote" in feedback
    assert marker not in feedback


def test_schema_rejection_does_not_silently_fallback_or_retry():
    calls = []

    def respond(request):
        calls.append(request)
        return httpx2.Response(400)

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings(LLM_RESPONSE_FORMAT="json_schema"), client=client) as extractor:
                with pytest.raises(LLMError, match="json_schema"):
                    await extractor.extract([], "invoice_draft")

    asyncio.run(run())
    assert len(calls) == 1


def test_prompt_contains_valid_example_without_invented_quantity():
    from src.llm import EXTRACTION_PROMPT

    example = EXTRACTION_PROMPT.split("<example_output>")[1].split("</example_output>")[0]
    data = json.loads(example)
    # Фрагмент поясняет вложенность, а не заменяет полный шаблон.
    from src.schemas.extraction_schemas import ProposedField
    items = ProposedField.model_validate(data["fields"]["items"])
    assert items.value[0].quantity is None
    assert items.value[0].unit_price_rub is None
    assert data["fields"]["amount_rub"]["calculation"] is None


def test_schema_does_not_replace_numeric_grounding_checks():
    data = proposal()
    data["fields"]["items"] = {
        "status": "fillable", "value": [{
            "description": "услуга", "quantity": 1, "unit": None, "unit_price_rub": 100,
        }], "evidence": [{"message_id": "m1", "quote": "Услуга на 100 рублей"}],
    }
    parsed = extract(data)
    with pytest.raises(ExtractionError, match="неподтверждённое число"):
        prepare_fields(parsed, [DocumentMessage(id="m1", role="user", text="Услуга на 100 рублей")], "invoice_draft")


def test_schema_feedback_reaches_next_request_and_recovers_pipeline(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    import src.pipeline as pipeline

    good = proposal()
    good["fields"]["amount_rub"] = {
        "status": "fillable", "value": 100,
        "evidence": [{"message_id": "m1", "quote": "100 рублей"}],
    }
    bad = proposal()
    bad["fields"]["items"] = []
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json=envelope(bad if len(requests) == 1 else good))

    classify = AsyncMock(return_value=SimpleNamespace(document_type="invoice_draft", confidence=0.9))
    verify = AsyncMock(return_value=[])
    monkeypatch.setattr(pipeline, "classify_document", classify)
    monkeypatch.setattr(pipeline, "verify_fields", verify)

    async def run():
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings(LLM_RESPONSE_FORMAT="json_schema"), client=client) as extractor:
                return await pipeline.process_document(
                    "example", [DocumentMessage(id="m1", role="user", text="Услуга на 100 рублей")],
                    object(), extractor,
                )

    result = asyncio.run(run())
    assert result.fields.amount_rub.value == 100
    assert len(requests) == 2
    feedback = json.loads(requests[1]["messages"][1]["content"])["feedback"]
    assert "fields.items: ожидается объект" in feedback[0]
    classify.assert_awaited_once()
    verify.assert_awaited_once()


@pytest.mark.parametrize("kwargs", [
    {"LLM_RESPONSE_FORMAT": "unsupported"}, {"LLM_TEMPERATURE": -1},
    {"LLM_TEMPERATURE": float("nan")},
])
def test_invalid_output_settings_are_rejected(kwargs):
    with pytest.raises(ValidationError):
        settings(**kwargs)
