import asyncio
import json

import httpx2
import pytest

from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionProposal, get_proposal_schema, get_template
from src.llm import AsyncLLMExtractor, LLMResponseError
from src.schemas.document_schemas import InvoiceFields
from src.settings import LLMSettings


def candidate():
    fields = {name: {"status": "missing", "value": None, "evidence": []}
              for name in InvoiceFields.model_fields}
    fields["recipient_name"] = {"status": "fillable", "value": "ИП Петрову",
                                "evidence": [{"message_id": "m1", "quote": "ИП Петрову"}]}
    fields["amount_rub"] = {"status": "fillable", "value": 1500,
                            "evidence": [{"message_id": "m1", "quote": "1500 рублей"}]}
    return ExtractionProposal.model_validate({"fields": fields})


def field(value, **kwargs):
    return {"status": "fillable", "value": value, "evidence": [], **kwargs}


def invoke(reply, *, protocol="anthropic", repair_fields=None, initial=False):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        text = json.dumps(reply)
        if protocol == "anthropic":
            data = {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 5}}
        else:
            data = {"choices": [{"finish_reason": "stop", "message": {"content": text}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        return httpx2.Response(200, json=data)

    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="fake", LLM_MODEL="small",
                               LLM_API_PROTOCOL=protocol, LLM_BASE_URL="https://example.com/v1",
                               LLM_RESPONSE_FORMAT="text" if protocol == "anthropic" else "json_schema")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                kwargs = {} if initial else {"previous_proposal": candidate(),
                                            "repair_fields": repair_fields or ["recipient_name"]}
                result = await extractor.extract([
                    DocumentMessage(id="m1", role="user", text="Счёт ИП Петрову на 1500 рублей"),
                ], "invoice_draft", **kwargs)
                assert extractor.usage.requests == 1
                assert extractor.usage.output_tokens == 5
                return result

    result = asyncio.run(run())
    return result, requests


def fields_definition(schema):
    return schema["$defs"][schema["properties"]["fields"]["$ref"].split("/")[-1]]


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
def test_repair_can_return_one_field_and_preserves_the_complete_candidate(protocol):
    old = candidate().model_dump()
    result, requests = invoke({"fields": {"recipient_name": field("ИП Петров")}}, protocol=protocol)
    assert set(result.fields) == set(InvoiceFields.model_fields)
    assert result.fields["recipient_name"].value == "ИП Петров"
    assert result.fields["amount_rub"].value == 1500
    assert candidate().model_dump() == old
    user = json.loads(requests[0]["messages"][-1]["content"])
    assert set(user["template"]["fields"]) == {"recipient_name"}
    output = fields_definition(user["output_schema"])
    assert set(output["properties"]) == set(output["required"]) == {"recipient_name"}
    assert output["additionalProperties"] is False
    if protocol == "openai":
        assert requests[0]["response_format"]["json_schema"]["schema"] == user["output_schema"]
    assert set(get_template("invoice_draft")["fields"]) == set(InvoiceFields.model_fields)


@pytest.mark.parametrize(("name", "value"), [
    ("recipient_name", 123), ("vat_mode", "invalid-option"),
    ("basis_date", "2026-99-99"), ("vat_rate_percent", "bad-number"),
])
def test_partial_response_still_has_document_specific_type_validation(name, value):
    with pytest.raises(LLMResponseError):
        invoke({"fields": {name: field(value)}}, repair_fields=[name])


def test_noncalculable_field_cannot_gain_a_formula_through_a_partial_response():
    with pytest.raises(LLMResponseError):
        invoke({"fields": {"recipient_name": field(None, calculation={
            "expression": "x", "operands": [{"name": "x", "source": {"message_id": "m1", "quote": "1500"}}],
        })}})


def test_missing_requested_repair_is_not_silently_replaced_by_the_old_value():
    with pytest.raises(LLMResponseError, match="recipient_name"):
        invoke({"fields": {}})


def test_unrequested_changes_are_ignored_if_gateway_returns_a_full_legacy_response():
    reply = candidate().model_dump()
    reply["fields"]["recipient_name"]["value"] = "ИП Петров"
    reply["fields"]["amount_rub"]["value"] = 99999
    result, _ = invoke(reply)
    assert result.fields["amount_rub"].value == 1500


def test_initial_extraction_still_requires_all_document_fields():
    with pytest.raises(LLMResponseError):
        invoke({"fields": {"recipient_name": field("ИП Петров")}}, initial=True)


def test_projected_schema_does_not_change_the_normal_schema_or_hide_unknown_fields():
    normal = get_proposal_schema("invoice_draft")
    projected = get_proposal_schema("invoice_draft", field_names=["amount_rub", "items"])
    assert fields_definition(projected)["required"] == ["amount_rub", "items"]
    assert get_proposal_schema("invoice_draft") == normal
    for names in ([], ["not_a_field"]):
        with pytest.raises(ValueError):
            get_proposal_schema("invoice_draft", field_names=names)


def test_real_extractor_partial_repair_is_merged_and_rechecked_by_pipeline():
    from src.pipeline import DocumentTypeDecision, process_selected_document

    requests = []
    verified = []

    def respond(request):
        requests.append(json.loads(request.content))
        data = candidate().model_dump()
        if len(requests) == 2:
            data = {"fields": {"recipient_name": {
                **data["fields"]["recipient_name"], "value": "ИП Петров",
            }}}
        return httpx2.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(data),
        }}]})

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            verified.append(questions)
            return response_model.model_validate({"answers": {
                key: {"noul": 0.9 if key == "recipient_name::noncanonical" and len(verified) == 1 else 0.01}
                for key in questions.root
            }})

    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="fake", LLM_MODEL="small",
                               LLM_BASE_URL="https://example.com/v1", LLM_RESPONSE_FORMAT="json_schema")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                return await process_selected_document("local", [DocumentMessage(
                    id="m1", role="user", text="Счёт ИП Петрову на 1500 рублей",
                )], DocumentTypeDecision(document_type="invoice_draft", confidence=0.9), Verifier(), extractor)

    result = asyncio.run(run())
    assert len(requests) == len(verified) == 2
    assert result.fields.recipient_name.value == "ИП Петров"
    assert result.fields.amount_rub.value == 1500
    assert verified[-1].root["amount_rub::unsupported"].instructions["field"]["value"] == 1500
    patch = json.loads(requests[-1]["messages"][-1]["content"])
    assert set(patch["template"]["fields"]) == {"recipient_name"}
