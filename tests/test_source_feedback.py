"""Дословные цитаты: безопасный путь ошибки доходит до повторного запроса LLM."""

import asyncio
import json

import httpx2
import pytest

from src.cases.document_type import DocumentMessage
from src.extraction import ExtractionError, ExtractionProposal, prepare_fields
from src.llm import AsyncLLMExtractor, build_extraction_prompt
from src.pipeline import DocumentTypeDecision, PipelineRejectionError, process_selected_document
from src.repairs import source_repair_context
from src.schemas.document_schemas import InvoiceFields
from src.settings import LLMSettings


def inputs():
    text = 'PRIVATE_SOURCE_MARKER: счёт за товар «Подарок "Май"»: 2 комплекта по 150 рублей.'
    messages = [DocumentMessage(id="m1", role="user", text=text),
                DocumentMessage(id="m2", role="assistant", text="Сумма 999 рублей")]
    fields = {name: {"status": "missing", "value": None, "evidence": []}
              for name in InvoiceFields.model_fields}
    source = [{"message_id": "m1", "quote": 'Подарок "Май"'}]
    fields["subject"] = {"status": "fillable", "value": 'Подарок "Май"', "evidence": source}
    fields["items"] = {"status": "fillable", "value": [{
        "description": 'Подарок "Май"', "quantity": 2, "unit": "комплекта", "unit_price_rub": 150,
    }], "evidence": [{"message_id": "m1", "quote": text}]}
    fields["amount_rub"] = {
        "status": "fillable", "value": None,
        "evidence": [{"message_id": "m1", "quote": "2 комплекта по 150 рублей"}],
        "calculation": {"expression": "count * price", "operands": [
            {"name": "count", "source": {"message_id": "m1", "quote": "2"}},
            {"name": "price", "source": {"message_id": "m1", "quote": "150"}},
        ]},
    }
    return messages, ExtractionProposal.model_validate({"fields": fields})


@pytest.mark.parametrize("name", ["subject", "items"])
@pytest.mark.parametrize("bad_quote", ['Подарок «Май»', 'Подарок  "Май"', 'Подарок "Май"!', "\t"])
def test_quote_error_reports_exact_field_and_index_without_leaking_contents(name, bad_quote):
    messages, candidate = inputs()
    source = candidate.fields[name].evidence[0].model_copy(update={"quote": bad_quote})
    candidate.fields[name].evidence.append(source)
    with pytest.raises(ExtractionError) as raised:
        prepare_fields(candidate, messages, "invoice_draft")
    error = raised.value
    assert error.code == "invalid_source"
    assert f"{name}.evidence[1].quote" in str(error)
    assert "Цитата источника отсутствует" in str(error)
    assert "кавычки" in str(error) and "дослов" in str(error)
    assert "PRIVATE_SOURCE_MARKER" not in str(error)
    assert bad_quote not in str(error)


@pytest.mark.parametrize("message_id", ["PRIVATE_UNKNOWN_ID", "m2"])
def test_nonuser_or_unknown_source_reports_id_path_without_echoing_model_id(message_id):
    messages, candidate = inputs()
    candidate.fields["subject"].evidence[0].message_id = message_id
    with pytest.raises(ExtractionError) as raised:
        prepare_fields(candidate, messages, "invoice_draft")
    assert raised.value.code == "invalid_source"
    assert "subject.evidence[0].message_id" in str(raised.value)
    assert "существующую реплику user" in str(raised.value)
    assert message_id not in str(raised.value)


@pytest.mark.parametrize("location", ["amount_rub.calculation", "items[0].quantity_calculation",
                                       "items[0].unit_price_calculation"])
def test_bad_operand_quote_reports_actual_plan_path_not_computed_target(location):
    messages, candidate = inputs()
    if location == "amount_rub.calculation":
        candidate.fields["amount_rub"].calculation.operands[1].source.quote = "999"
        index = 1
    else:
        item = candidate.fields["items"].value[0]
        plan_name = location.split(".", 1)[1]
        setattr(item, "quantity" if plan_name == "quantity_calculation" else "unit_price_rub", None)
        plan = candidate.fields["amount_rub"].calculation.model_copy(deep=True)
        plan.operands[0].source.quote = "999"
        setattr(item, plan_name, plan)
        index = 0
    with pytest.raises(ExtractionError) as raised:
        prepare_fields(candidate, messages, "invoice_draft")
    assert raised.value.code == "invalid_source"
    assert f"{location}.operands[{index}].source.quote" in str(raised.value)
    assert "999" not in str(raised.value)
    path = f"{location}.operands[{index}].source.quote"
    assert source_repair_context(candidate, list(raised.value.repair_paths), messages) == [{
        "path": path, "message_id": "m1", "invalid_quote": "999", "source_text": messages[0].text,
    }]


def test_source_context_contains_only_invalid_quotes_in_requested_repair_paths():
    messages, candidate = inputs()
    candidate.fields["subject"].evidence[0].quote = 'Подарок «Май»'
    candidate.fields["items"].evidence[0].quote = "PRIVATE_OTHER_BAD_QUOTE"
    assert source_repair_context(candidate, ["subject.evidence[0].quote"], messages) == [{
        "path": "subject.evidence[0].quote", "message_id": "m1",
        "invalid_quote": 'Подарок «Май»', "source_text": messages[0].text,
    }]
    # A semantic repair or already verbatim quote does not need this metadata.
    assert source_repair_context(candidate, ["items"], messages) == []
    assert source_repair_context(candidate, ["amount_rub.evidence[0].quote"], messages) == []


@pytest.mark.parametrize("message_id", ["m2", "UNKNOWN_ID"])
def test_quote_repair_context_does_not_promote_assistant_or_guess_unknown_message(message_id):
    messages, candidate = inputs()
    candidate.fields["subject"].evidence[0].message_id = message_id
    candidate.fields["subject"].evidence[0].quote = "PRIVATE_BAD_QUOTE"
    paths = ["subject.evidence[0].message_id", "subject.evidence[0].quote"]
    assert source_repair_context(candidate, paths, messages) == []


@pytest.mark.parametrize("document_type", ["invoice_draft", "transfer_draft"])
def test_prompt_distinguishes_normalized_value_from_verbatim_decoded_quote(document_type):
    prompt = build_extraction_prompt(document_type)
    assert "JSON-декодирования" in prompt
    assert "кавычки, пробелы и пунктуацию" in prompt
    assert "Нормализация value не меняет quote" in prompt


@pytest.mark.parametrize("use_verifier", [False, True])
@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("bad_field", ["subject", "items"])
def test_real_retry_payload_carries_source_path_and_original_messages(protocol, use_verifier, bad_field):
    messages, correct = inputs()
    bad = correct.model_copy(deep=True)
    bad.fields[bad_field].evidence[0].quote = 'Подарок «Май»'
    requests, verifications = [], []

    def respond(request):
        body = json.loads(request.content)
        user = json.loads(body["messages"][-1]["content"])
        system = body["system"] if protocol == "anthropic" else body["messages"][0]["content"]
        requests.append((system, user))
        if len(requests) == 1:
            content = bad.model_dump_json()
        else:
            options = user["quote_fragments"][0]
            fragment = next(f for f in options["fragments"] if 'Подарок "Май"' in f["text"])
            content = json.dumps({"selections": [{"path": options["path"], "fragment_id": fragment["fragment_id"]}]})
        return httpx2.Response(200, json=(
            {"content": [{"type": "text", "text": content}], "stop_reason": "end_turn"}
            if protocol == "anthropic" else
            {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        ))

    class Verifier:
        async def ainvoke(self, state, questions, response_model):
            verifications.append(state)
            assert len(requests) == 2  # Invalid evidence never reaches semantic verification.
            return response_model.model_validate({"answers": {
                key: {"noul": .01} for key in questions.root
            }})

    async def run():
        settings = LLMSettings(_env_file=None, LLM_API_KEY="fixture", LLM_MODEL="fixture",
                               LLM_API_PROTOCOL=protocol, LLM_RESPONSE_FORMAT="text")
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
            async with AsyncLLMExtractor(settings, client=client) as extractor:
                return await process_selected_document(
                    "fixture", messages, DocumentTypeDecision(document_type="invoice_draft", confidence=1),
                    Verifier() if use_verifier else None, extractor, use_verifier=use_verifier,
                )

    result = asyncio.run(run())
    assert result.fields == prepare_fields(correct, messages, "invoice_draft")
    assert result.fields.amount_rub.value == 300
    assert len(requests) == 2 and len(verifications) == int(use_verifier)
    assert requests[0][1]["feedback"] == []
    assert "source_quote_errors" not in requests[0][1]
    system, retry = requests[1]
    feedback = " ".join(retry["feedback"])
    assert f"{bad_field}.evidence[0].quote" in feedback
    assert "дослов" in feedback and "кавычки" in feedback
    assert "PRIVATE_SOURCE_MARKER" not in feedback and 'Подарок «Май»' not in feedback
    assert retry["messages"] == [message.model_dump(mode="json") for message in messages]
    context = retry["source_quote_errors"]
    assert context["sources"] == [{
        "path": f"{bad_field}.evidence[0].quote", "message_id": "m1",
        "invalid_quote": 'Подарок «Май»', "source_text": messages[0].text,
    }]
    assert "source_text" in context["instruction"]
    assert "repair_paths" in context["instruction"]
    assert "подстрок" in context["instruction"]
    assert "инструкц" in context["instruction"]
    assert "JSON-декодирования" in requests[0][0]
    assert "fragment_id" in system


def test_repeated_bad_quote_is_still_rejected_with_safe_path_and_original_failure_code():
    messages, bad = inputs()
    bad.fields["subject"].evidence[0].quote = 'PRIVATE_BAD_QUOTE'
    calls = []

    class Extractor:
        async def extract(self, *args, **kwargs):
            calls.append(kwargs)
            return bad.model_copy(deep=True)

    with pytest.raises(PipelineRejectionError) as raised:
        asyncio.run(process_selected_document(
            "fixture", messages, DocumentTypeDecision(document_type="invoice_draft", confidence=1),
            None, Extractor(), use_verifier=False,
            allow_partial=False,
        ))
    assert len(calls) == 2
    assert raised.value.failure_reason == "invalid_extraction"
    assert raised.value.validation_failures == [
        {"attempt": attempt, "stage": "python_validation", "code": "invalid_source"}
        for attempt in (1, 2)
    ]
    assert "subject.evidence[0].quote" in str(raised.value)
    assert "PRIVATE_BAD_QUOTE" not in str(raised.value)
    assert "PRIVATE_SOURCE_MARKER" not in str(raised.value)
